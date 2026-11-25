# Abstract
# Each signal is stored the way the overfitted neural codecs COOL-CHIC [3] and C3 [4] store an image: a
# pyramid of integer latent grids, a small decoder, and nothing else. A level is a dense grid over the sample
# lattice with cells of 1 to 64 samples; its integers, times a learned step [8], are interpolated
# multilinearly at the queried coordinate [2], the levels are concatenated, and an MLP with two hidden layers
# of 64 units (plus a linear skip) turns them into the value. Video grids have separate cell lengths in space
# and time [9], 18 combinations such as 1 pixel by 8 frames for still detail and 4 pixels by 1 frame for fast
# motion; images and audio use the 7 lengths of their one kind of axis. The integers are entropy coded with
# LZMA, as Deep Compression [10] codes quantised weights, into a byte buffer that together with the float16
# decoder is the saved 1 MiB. Evaluation rebuilds the grids from that buffer alone.
# Training draws coordinates uniformly and minimises MSE x (1 + price x estimated code length / budget)
# [5, 6] with straight-through rounding [7]. The code length is the sum of token costs under each channel's
# own token frequencies, interpolated linearly between integer magnitudes so that it has a gradient; it
# drives the latents that are not worth their bits to exactly zero. A controller moves the price so that the
# code fills the byte limit, and one common rounding threshold lands on the limit exactly at the end.
# Finally the grids and decoder are frozen and each drawn sample's remaining error is written into a
# residual table as an integer multiple of one step, the finest step whose code still fits (bisection);
# for the audio the latents are first coarsened to make room, as a lossless audio coder trades predictor
# for residual [11]. 1 MiB is 27 bits per audio sample, so the waveform is stored to within 1e-8, below the
# floor of the PSNR formula; the photographs use the same step for a sparse correction, and the video, at
# 0.12 bits per pixel, is too large for a table.
# Tuning runs: about 45.5 dB on parrot and 52.2 dB on crow after 6,000 updates of 131,072 samples (20,000
# updates were no better); 126.0 dB on bach, the ceiling of the PSNR formula (MSE below its floor of
# 1e-12); 32.2 dB on cat after 3,000 updates and still rising with more, which is why the run is long.
# The sine-ReLU starter scores 25.8, 27.7, 31.9 and 21.5 dB. At the same 1 MiB, JPEG 2000 reaches 43.7 and
# 50.4 dB on the photographs and x265 39.2 dB on the video. Worse in my tests: a multiresolution hash grid
# with float16 tables [2] (8 to 12 dB lower on the photographs); a Laplace model of the code length instead
# of the token frequencies (-1 dB); adding the finest level to the output instead of feeding it to the
# decoder (-3 dB); updating only the latents a batch touches (-0.9 dB). Within run-to-run noise: cubic
# interpolation, neighbouring latents as decoder inputs, additive noise instead of straight-through
# rounding, and a wider or deeper decoder.
#
# References / Materials used
# [1] Deep Learning Practical 9: Neural networks as images, video and sound. Durham University.
# [2] T. Mueller, A. Evans, C. Schied and A. Keller. Instant neural graphics primitives with a
#     multiresolution hash encoding. SIGGRAPH 2022. https://arxiv.org/abs/2201.05989
# [3] T. Ladune, P. Philippe, F. Henry, G. Clare and T. Leguay. COOL-CHIC: coordinate-based low complexity
#     hierarchical image codec. ICCV 2023. https://arxiv.org/abs/2212.05458
# [4] H. Kim, M. Bauer, L. Theis, J. R. Schwarz and E. Dupont. C3: high-performance and low-complexity
#     neural compression from a single image or video. CVPR 2024. https://arxiv.org/abs/2312.02753
# [5] J. Balle, V. Laparra and E. P. Simoncelli. End-to-end optimized image compression. ICLR 2017.
#     https://arxiv.org/abs/1611.01704
# [6] S. Girish, A. Shrivastava and K. Gupta. SHACIRA: scalable hash-grid compression for implicit neural
#     representations. ICCV 2023. https://arxiv.org/abs/2309.15848
# [7] Y. Bengio, N. Leonard and A. Courville. Estimating or propagating gradients through stochastic
#     neurons for conditional computation. 2013. https://arxiv.org/abs/1308.3432
# [8] S. K. Esser, J. L. McKinstry, D. Bablani, R. Appuswamy and D. S. Modha. Learned step size
#     quantization. ICLR 2020. https://arxiv.org/abs/1902.08153
# [9] S. Kim, S. Yu, J. Lee and J. Shin. Scalable neural video representations with learnable positional
#     features. NeurIPS 2022. https://arxiv.org/abs/2210.06823
# [10] S. Han, H. Mao and W. J. Dally. Deep compression: compressing deep neural networks with pruning,
#     trained quantization and Huffman coding. ICLR 2016. https://arxiv.org/abs/1510.00149
# [11] T. Robinson. SHORTEN: simple lossless and near-lossless waveform compression. Technical report
#     CUED/F-INFENG/TR.156, Cambridge University, 1994.
# [12] D. P. Kingma and J. Ba. Adam: a method for stochastic optimization. ICLR 2015.
#     https://arxiv.org/abs/1412.6980
# [13] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell and G. Wetzstein. Implicit neural
#     representations with periodic activation functions. NeurIPS 2020. https://arxiv.org/abs/2006.09661
# [14] I. Pavlov. LZMA (the xz format), used through Python's lzma module. https://tukaani.org/xz/
#
# AI acknowledgement
# Claude (Anthropic) was used to plan and run the experiments, write the changes to the template code and
# draft these comments.
#
# Set STUDENT_ID, then run this file or paste it into one notebook cell.
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import lzma
import math
from pathlib import Path
import re
import time
from urllib.request import urlretrieve
import zipfile

import numpy as np
import torch
from torch import nn

STUDENT_ID = "studentID"
DATA_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
DATA_URL = "https://raw.githubusercontent.com/cwkx/research-data/main/datasets/multimodal.npz"
DATA_SHA256 = "d83c908f100d76b458608e75347eaaa33e2cf882d09a06350bdece51e15c4d20"
CONFIG = dict(task="implicit", steps=100000, batch_size=262144,
              learning_rate=0.01, weight_decay=0.0, storage_dtype="float16")
MAX_MODEL_BYTES = 1024**2
NAMES = ("parrot", "crow", "bach", "cat")
SAMPLE_NAMES = ("parrot.png", "crow.png", "bach.wav", "cat.mp4")

# %% Model: a pyramid of integer latent grids, entropy coded, and a small decoder
# Cell lengths (space, time) in samples of the latent grids. An image uses the space lengths, audio the time
# lengths and video both, so a level can be fine in space and coarse in time or the other way round.
LEVELS = ((1, 8), (1, 32), (2, 2), (2, 4), (2, 8), (2, 16), (4, 1), (4, 2), (4, 4), (4, 8), (4, 32),
          (8, 2), (8, 8), (8, 16), (16, 4), (16, 16), (32, 32), (64, 64))
CHANNELS = 3             # latent channels per level
WIDTH, DEPTH = 64, 2     # hidden layers of the decoder
STEP = 0.05              # initial quantisation step of the latents
RESIDUAL_LIMIT = 3e7     # a signal with more values than this gets no residual table
MARGIN = 0.006           # the rate controller aims this far below the byte limit
REFRESH, MEASURE = 50, 1000  # updates between refreshes of the token costs / measurements of the real code
# The latent integers are modelled as tokens: 0 to 15 are themselves, then two buckets per octave whose
# position inside the bucket costs raw bits. TOKEN_START holds the smallest magnitude of each token.
OCTAVES = [(octave, half) for octave in range(4, 28) for half in (0, 1)]
TOKEN_START = torch.tensor(list(range(16)) + [(2+half) << (octave-1) for octave, half in OCTAVES]).float()
TOKEN_WIDTH = torch.tensor([1]*16 + [1 << (octave-1) for octave, half in OCTAVES]).float()
LZMA = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 3, "lp": 0, "pb": 0,
         "dict_size": 1 << 26}]
POOL = ThreadPoolExecutor(16)


def rounded(x):
    """float16 storage seen in the forward pass, with a straight-through gradient."""
    return x + (x.half().float() - x).detach()


def layout(sizes):
    """Cell lengths and vertex counts of the levels of a lattice with these sizes along (x, y, t)."""
    levels = []
    for space, time in LEVELS:
        cells = {1: (time,), 2: (space, space), 3: (space, space, time)}[len(sizes)]
        if cells not in [c for c, _ in levels]:
            levels.append((cells, [-(-(n-1)//c) + 1 for n, c in zip(sizes, cells)]))
    return levels


def interpolate(table, cells, resolution, position, step=None):
    """Multilinear interpolation of a [vertices, channels] table at sample positions [batch, d].
    With a step, the entries are rounded to multiples of it first (straight-through gradient)."""
    index, weight, stride = [0], [1.0], 1
    for axis, (cell, size) in enumerate(zip(cells, resolution)):
        u = position[:, axis]/cell
        low = u.floor()
        high, low = u - low, low.long()*stride
        if cell == 1:  # the scorer asks for sample positions, which sit on the vertices of this axis
            index = [i + low for i in index]
        else:
            above = (low + stride).clamp(max=(size-1)*stride)
            index = [i + low for i in index] + [i + above for i in index]
            weight = [w*(1 - high) for w in weight] + [w*high for w in weight]
        stride *= size
    index = torch.stack(torch.broadcast_tensors(*index), dim=1)
    value = table[index]
    if step is not None:
        scaled = value/step
        value = (scaled + (scaled.round() - scaled).detach())*step
    if len(weight) == 1:
        return value[:, 0]
    return (value*torch.stack(weight, dim=1)[:, :, None]).sum(1)


def pack_plane(q):
    """Entropy-code one integer array: narrowest integer type, split into byte planes, raw LZMA2."""
    kind = next(k for k, t in enumerate((np.int8, np.int16, np.int32))
                if np.iinfo(t).min <= q.min(initial=0) and q.max(initial=0) <= np.iinfo(t).max)
    raw = np.ascontiguousarray(q.astype((np.int8, np.int16, np.int32)[kind]))
    raw = np.ascontiguousarray(raw.view(np.uint8).reshape(len(q), -1).T)
    return bytes([kind]) + lzma.compress(raw.tobytes(), format=lzma.FORMAT_RAW, filters=LZMA)


def unpack_plane(data, count):
    kind = (np.int8, np.int16, np.int32)[data[0]]
    raw = np.frombuffer(lzma.decompress(data[1:], format=lzma.FORMAT_RAW, filters=LZMA), np.uint8)
    return np.ascontiguousarray(raw.reshape(-1, count).T).view(kind).ravel()


def pack(tables):
    """Integer tables [entries, channels] -> one byte string: every channel is a stream with its length."""
    tables = [table.cpu().numpy() for table in tables]
    planes = [np.ascontiguousarray(q[:, channel]) for q in tables for channel in range(q.shape[1])]
    streams = list(POOL.map(pack_plane, planes))
    return b"".join(len(s).to_bytes(4, "little") + s for s in streams)


def unpack(data, shapes):
    streams, at = [], 0
    for entries, channels in shapes:
        for _ in range(channels):
            size = int.from_bytes(data[at:at+4], "little")
            streams.append((data[at+4:at+4+size], entries))
            at += 4 + size
    planes = iter(POOL.map(lambda s: unpack_plane(*s), streams))
    return [np.stack([next(planes) for _ in range(channels)], axis=1) for entries, channels in shapes]


class Model(nn.Module):
    def __init__(self, inputs=2, outputs=3):
        super().__init__()
        # Inputs: 1 time coordinate for audio, 2 for images, 3 for video.
        features = CHANNELS*len(layout([2]*inputs))
        layers, last = [], features
        for _ in range(DEPTH):
            layers += [nn.Linear(last, WIDTH), nn.ReLU()]
            last = WIDTH
        # Outputs: 1 amplitude for mono audio, or 3 RGB values for images/video.
        self.mlp = nn.Sequential(*layers, nn.Linear(last, outputs))
        self.skip = nn.Linear(features, outputs, bias=False)
        # Logarithm of the quantisation step of every latent channel; the last one is the residual table's.
        self.log_step = nn.Parameter(torch.full((features+1,), math.log(STEP)))
        # Samples along (x, y, t), then the number of code bytes and whether a residual table follows.
        self.register_buffer("lattice", torch.zeros(inputs+2, dtype=torch.long))
        self.register_buffer("code", torch.zeros(MAX_MODEL_BYTES, dtype=torch.uint8))
        self.code = self.code[:2*MAX_MODEL_BYTES - self.saved_bytes()].clone()  # the file is exactly 1 MiB
        self.tables, self.residual, self.outputs = None, None, outputs

    def saved_bytes(self):
        """Size of the file that save_model writes for this model."""
        arrays = {name: value.detach().cpu().numpy().astype(
            CONFIG["storage_dtype"] if value.is_floating_point() else value.cpu().numpy().dtype)
            for name, value in list(self.named_parameters()) + list(self.named_buffers())}
        with io.BytesIO() as file:
            np.savez(file, **arrays)
            return file.tell()

    def sizes(self):
        return [int(n) for n in self.lattice[:-2]]

    def steps(self):
        """Quantisation step of every level's channels, then of the residual table."""
        return rounded(self.log_step).exp().split(CHANNELS)

    @torch.no_grad()
    def write(self, tables, residual=None):
        """Entropy-code integer tables (and the residual table) into the saved state. Returns the bytes used."""
        data = pack(list(tables) + ([residual] if residual is not None else []))
        if len(data) <= len(self.code):
            self.code.zero_()
            self.code[:len(data)] = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(self.code.device)
            self.lattice[-2], self.lattice[-1] = len(data), int(residual is not None)
        return len(data)

    @torch.no_grad()
    def read(self):
        """Rebuild the latent tables and the residual table from the saved state alone."""
        sizes = self.sizes()
        if not self.lattice[-2]:
            return  # nothing has been written yet
        shapes = [(math.prod(resolution), CHANNELS) for _, resolution in layout(sizes)]
        if self.lattice[-1]:
            shapes.append((math.prod(sizes), self.outputs))
        data = self.code[:int(self.lattice[-2])].cpu().numpy().tobytes()
        tables = [torch.as_tensor(q.astype(np.int32), device=self.code.device) for q in unpack(data, shapes)]
        self.residual = tables.pop() if self.lattice[-1] else None
        self.tables = [q.float()*step for q, step in zip(tables, self.steps())]

    def train(self, mode=True):
        super().train(mode)
        if not mode:  # evaluation always starts from the saved bytes
            self.read()
        return self

    def forward(self, x, residual=True):
        sizes = self.sizes()
        # Sample index along each axis; the scorer queries exact sample positions, so snap to them.
        position = (x + 1)*x.new_tensor(sizes)/2 - 0.5
        nearest = position.round()
        position = torch.where((position - nearest).abs() < 0.05, nearest, position)
        steps = self.steps() if self.training else [None]*len(self.tables)
        features = torch.cat([interpolate(table, cells, resolution, position, step) for table, step,
                              (cells, resolution) in zip(self.tables, steps, layout(sizes))], dim=1)
        hidden = features
        for layer in self.mlp:
            if isinstance(layer, nn.Linear) and self.training:
                hidden = nn.functional.linear(hidden, rounded(layer.weight), rounded(layer.bias))
            else:
                hidden = layer(hidden)
        out = hidden + nn.functional.linear(features, rounded(self.skip.weight))
        if self.residual is not None and residual:
            index, stride = 0, 1
            for axis, size in enumerate(sizes):
                index = index + nearest[:, axis].long().clamp(0, size-1)*stride
                stride *= size
            out = out + self.residual[index].float()*self.steps()[-1]
        return out


def build_model(inputs=2, outputs=3):
    return Model(inputs, outputs)


def coordinates(index, shape):
    """Convert flat array indices to (x, y), (t,) or (x, y, t)."""
    axes = []
    for size in reversed(shape):
        axes.append(2*((index % size).float()+0.5)/size-1)
        index = index // size
    return torch.stack(axes, dim=1)


# %% Training: rate-distortion fitting of the latents, then an exact residual table
def token_costs(integers):
    """Bits to code each token of every channel of an integer table under the channel's own token
    frequencies (sign and position bits included), and the bits the whole table takes that way."""
    size = integers.abs().float()
    token = torch.bucketize(size, TOKEN_START.to(size.device), right=True) - 1
    costs, total = [], 0.0
    for channel in range(size.shape[1]):
        count = torch.bincount(token[:, channel], minlength=len(TOKEN_START)).float()
        unseen = 1e-3*count.sum()*0.5**torch.arange(1, len(count)+1, device=size.device)  # for new tokens
        cost = -torch.log2((count + unseen)/(count + unseen).sum()) + TOKEN_WIDTH.to(size.device).log2()
        cost[1:] += 1
        costs.append(cost)
        total += (count*cost).sum().item()
    return torch.stack(costs), total


@torch.no_grad()
def add_rate(latent, step, costs, step_grad, weight, chunk=1 << 22):
    """Estimated bits to code a latent table: the token costs, interpolated linearly between the magnitudes
    where tokens start so that they have a slope. Adds weight * gradient to latent.grad and step_grad."""
    start, width = TOKEN_START.to(latent.device), TOKEN_WIDTH.to(latent.device)
    channel = torch.arange(latent.shape[1], device=latent.device)
    bits = 0.0
    for first in range(0, len(latent), chunk):
        value = latent[first:first+chunk]
        size = (value/step).abs()
        token = size.long()                       # tokens 0..15 are the integers themselves
        if size.max() >= 16:
            token = torch.bucketize(size, start, right=True).sub_(1).clamp_(max=len(start)-2)
        low = costs[channel, token]
        slope = (costs[channel, token+1] - low)/width[token]
        bits += (low + (size - start[token])*slope).sum().item()
        latent.grad[first:first+chunk] += weight*torch.sign(value)*slope/step
        step_grad -= weight*(size*slope).sum(0)
    return bits


@torch.no_grad()
def land(model, latents, capacity):
    """Integer latents whose code fits the capacity exactly, by one common rounding threshold: entries with
    |value/step| below it are stored as 0, the others as round(|value/step|) but at least 1. (0.5 is plain
    rounding; lower promotes the zeros nearest to +-1, higher drops the smallest non-zero entries.)"""
    sizes = [(value/step).abs() for value, step in zip(latents, model.steps())]
    tables = lambda threshold: [torch.sign(value)*torch.where(size < threshold, 0, size.round().clamp(min=1))
                                for value, size in zip(latents, sizes)]
    low, high = 0.4, 4.0  # too many bytes, fits
    for _ in range(26):  # many entries sit very close to 0.5, so the threshold has to be resolved finely
        middle = (low+high)/2
        fits = len(pack([q.int() for q in tables(middle)])) <= capacity
        low, high = (low, middle) if fits else (middle, high)
    return tables(high)


@torch.no_grad()
def residual_of(model, values, shape, index):
    """Target minus the prediction of the latents and decoder alone, in the scorer's float32 arithmetic.
    Zero where the scorer's clamp to [-1, 1] already removes the error."""
    target = values[index].float()
    if values.dtype == torch.uint8:
        target = target/127.5-1
    error = target - model(coordinates(index, shape), residual=False).float()
    return torch.where(((target >= 1) & (error < 0)) | ((target <= -1) & (error > 0)), 0, error)


@torch.no_grad()
def plan_endgame(model, values, shape, integers, capacity):
    """Split the bytes between the trained latents and a table that stores the remaining error of every
    sample as an integer multiple of one step. Candidates: all latent integers divided by 1, 2, 4, ...
    (0: latents dropped); for each, the finest residual step that fits; the lowest exact MSE wins."""
    base = model.log_step.detach().clone()
    best, misses = None, 0
    for coarsen in (1, 2, 4, 8, 16, 32, 64, 256, 0):
        tables = [(q/coarsen).round() if coarsen else torch.zeros_like(q) for q in integers]
        model.log_step.copy_((base + math.log(max(coarsen, 1))).half().float())
        model.tables = [q*step for q, step in zip(tables, model.steps())]
        room = capacity - len(pack([q.int() for q in tables]))
        chunks = torch.arange(len(values), device=values.device).split(2**19)
        error = torch.cat([residual_of(model, values, shape, index) for index in chunks])
        if best is None:
            best = (error.double().square().mean().item(), 1, None)  # no residual table at all
        peak = error.abs().max().item()
        size = lambda log: len(pack([(error/math.exp(log)).round().int()]))
        low, high = math.log(max(peak/2**22, 1e-12)), math.log(max(2*peak, 1e-9))  # too fine, surely fits
        if size(high) > room:
            continue
        for _ in range(14):
            middle = (low+high)/2
            low, high = (low, middle) if size(middle) <= room else (middle, high)
        log = float(np.float16(high))  # the step is stored as a float16 logarithm
        while size(log) > room:
            log = float(np.nextafter(np.float16(log), np.float16(np.inf)))
        mse = (error - (error/math.exp(log)).round()*math.exp(log)).double().square().mean().item()
        if mse < 0.98*best[0]:
            best, misses = (mse, coarsen, log), 0
        else:
            misses += 1
            if misses == 2:
                break
    mse, coarsen, log = best
    model.log_step.copy_((base + math.log(max(coarsen, 1))).half().float())
    if log is not None:
        model.log_step[-1] = log
        model.residual = torch.zeros(len(values), values.shape[1], dtype=torch.int32, device=values.device)
    return [(q/coarsen).round() if coarsen else torch.zeros_like(q) for q in integers]


def train_signal(model, values, shape, seed):
    model.train()
    generator = torch.Generator(device=values.device).manual_seed(seed+1)
    steps, batch = CONFIG["steps"], CONFIG["batch_size"]
    model.lattice[:len(shape)] = torch.tensor(shape[::-1])
    model.residual = None
    # Fresh latents: one dense grid per level over the sample lattice, all zero.
    latents = [torch.zeros(math.prod(resolution), CHANNELS, device=values.device, requires_grad=True)
               for _, resolution in layout(model.sizes())]
    decoder = list(model.parameters())
    optimisers = [torch.optim.Adam(latents, lr=CONFIG["learning_rate"], betas=(0.9, 0.99), eps=1e-15,
                                   fused=values.is_cuda),
                  torch.optim.Adam(decoder, lr=CONFIG["learning_rate"]/2, betas=(0.9, 0.99), eps=1e-15,
                                   weight_decay=CONFIG["weight_decay"])]
    rates = [optimiser.param_groups[0]["lr"] for optimiser in optimisers]
    integers = lambda: [(value/step).round().int() for value, step in zip(latents, model.steps())]
    capacity = len(model.code) - 2048                    # bytes for the code, with a small reserve
    budget = 8*(1-MARGIN)*capacity                       # bits the controller aims for
    price, integral, calibration, packing, estimate = 1.0, 1.0, 1.0, 1.0, None
    # A signal that is small enough gets a residual table, written by the last updates: one pass over it.
    table = values.numel() <= RESIDUAL_LIMIT
    joint = steps - (math.ceil(len(values)/batch) if table else 0)
    history = []
    for step in range(1, steps+1):
        if step <= joint:
            index = torch.randint(len(values), (batch,), device=values.device, generator=generator)
            # Joint stage: minimise MSE * (1 + price * coded bits / budget) with rounded latents.
            scale = min(1, step/100)*(0.01 + 0.99*0.5*(1 + math.cos(math.pi*(step-1)/max(1, joint-1))))
            for optimiser, rate in zip(optimisers, rates):
                optimiser.param_groups[0]["lr"] = rate*scale
            if step % REFRESH == 1:
                # Token costs of the current integers; calibrate the estimate against them and, more
                # rarely, against the size of the real code.
                with torch.no_grad():
                    costs, exact = zip(*[token_costs(q) for q in integers()])
                    if step % MEASURE == 1 and step > 1:
                        packing = 8*len(pack(integers()))/sum(exact)
                    calibration = packing*sum(exact)/estimate if estimate else 1.0
            target = values[index].float()
            if values.dtype == torch.uint8:
                target = target/127.5-1
            model.tables = latents
            error = model(coordinates(index, shape)) - target
            # Predictions are clamped to [-1, 1] when scored, so overshoot beyond a saturated target is free.
            error = torch.where(((target >= 1) & (error > 0)) | ((target <= -1) & (error < 0)), 0, error)
            loss = error.square().mean()
            for optimiser in optimisers:
                optimiser.zero_grad()
            loss.backward()
            weight = price*loss.item()*calibration/budget
            bits = sum(add_rate(value, step_, cost, model.log_step.grad[k*CHANNELS:(k+1)*CHANNELS], weight)
                       for k, (value, step_, cost) in enumerate(zip(latents, model.steps(), costs)))
            for optimiser in optimisers:
                optimiser.step()
            # A controller moves the price of bits so that the coded latents just fill the budget.
            estimate = bits if estimate is None else 0.9*estimate + 0.1*bits
            excess = min(max(calibration*estimate/budget - 1, -0.5), 0.5)
            integral = min(max(integral*math.exp((0.03 if excess > 0 else 0.01)*excess), 0.05), 300.0)
            price = min(integral*math.exp(3*max(excess, -0.2)), 300.0)
            if step == joint:
                # Fix everything at its stored value, land on the byte limit and plan the residual table.
                with torch.no_grad():
                    for parameter in decoder:
                        parameter.copy_(parameter.half().float())
                    final = land(model, latents, capacity)
                    if table:
                        final = plan_endgame(model, values, shape, final, capacity)
                    model.tables = [q*step_ for q, step_ in zip(final, model.steps())]
                    if model.residual is not None:
                        order = torch.randperm(len(values), device=values.device, generator=generator)
        elif model.residual is not None:
            # Endgame: every sample is drawn once, in random order, and its residual is written into the
            # table. This is an exact Newton step, because the table enters the output linearly.
            with torch.no_grad():
                index = order[(step-joint-1)*batch:(step-joint)*batch]
                error = residual_of(model, values, shape, index)
                unit = model.steps()[-1]
                model.residual[index] = (error/unit).round().int()
                loss = (error - model.residual[index]*unit).square().mean()
        if step == 1 or step % max(1, steps//40) == 0 or step == steps:
            history.append([step, loss.item()])
        if step == 1 or step % max(1, steps//10) == 0 or step == steps:
            print(f"{step} / {steps}: MSE {loss.item():.6g}", flush=True)
    final, threshold = [q.int() for q in final], 0
    while model.write(final, model.residual) > len(model.code):  # rare: drop the smallest residuals
        threshold += 1
        model.residual[model.residual.abs() <= threshold] = 0
    return history


# %% Save model state and reconstruct the complete signals
@torch.no_grad()
def save_model(model, path, precision):
    arrays = {}
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        value = value.detach().cpu()
        if value.is_floating_point():
            value = value.to(getattr(torch, precision))
        arrays[name] = value.numpy()
        if not np.isfinite(arrays[name]).all():
            raise ValueError("Non-finite model state: "+name)
    np.savez(path, **arrays)
    size = Path(path).stat().st_size
    if size > MAX_MODEL_BYTES:
        raise ValueError(f"Model uses {size:,} bytes; the limit is {MAX_MODEL_BYTES:,}.")
    return size


@torch.no_grad()
def reconstruct(model, values, shape, path, rate=44100, fps=(25, 1), batch_size=32768):
    model.eval()
    path = Path(path)
    video = path.suffix == ".mp4"
    output = None if video else np.empty((len(values), values.shape[1]), dtype=np.float32)
    writer = None
    if video:
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(str(path), (shape[2], shape[1]),
            fps=fps[0]/fps[1], codec="libx264", pix_fmt_out="yuv420p",
            quality=None, macro_block_size=1,
            output_params=["-crf", "18", "-maxrate", "2500k", "-bufsize", "5000k",
                           "-movflags", "+faststart"])
        writer.send(None)
    error, energy = 0.0, 0.0
    try:
        for first in range(0, len(values), batch_size):
            last = min(first+batch_size, len(values))
            index = torch.arange(first, last, device=values.device)
            target = values[first:last].float()
            if values.dtype == torch.uint8:
                target = target/127.5-1
            prediction = model(coordinates(index, shape)).float().clamp(-1, 1)
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise ValueError("Return one finite prediction per target value.")
            error += (prediction-target).square().double().sum().item()
            energy += target.square().double().sum().item()
            if video:
                # Stream RGB batches; FFmpeg assembles them into complete frames.
                writer.send(((prediction+1)*127.5).round().byte().cpu().numpy().tobytes())
            else:
                output[first:last] = prediction.cpu().numpy()
    finally:
        if writer is not None:
            writer.close()
    if video and path.stat().st_size >= 5_000_000:
        raise ValueError("Keep the compressed video preview below 5 MB.")
    if path.suffix == ".png":
        from PIL import Image
        pixels = np.rint((output.reshape(*shape, -1)+1)*127.5).astype(np.uint8)
        Image.fromarray(pixels).save(path)
    elif path.suffix == ".wav":
        from scipy.io import wavfile
        wavfile.write(path, rate, output[:, 0] if output.shape[1] == 1 else output)
    mse = error/values.numel()
    return dict(mse=mse, psnr_db=10*math.log10(4/max(mse, 1e-12)),
                snr_db=10*math.log10(max(energy, 1e-12)/max(error, 1e-12)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def program_source():
    if "__file__" in globals():
        source = Path(__file__).read_text()
    else:
        source = get_ipython().history_manager.input_hist_raw[-1].rstrip()+"\n"
    names = {getattr(node, "name", "") for node in ast.parse(source).body}
    if not {"build_model", "run"} <= names:
        raise ValueError("Run the complete program in one notebook cell.")
    return source


# %% Run four independent fits and write the submission ZIP
def run(out=None, device=None, seed=3801, split="reconstruction", show=True):
    source = program_source()
    if split != "reconstruction" or CONFIG["task"] != "implicit":
        raise ValueError("Use the full-signal reconstruction task.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", STUDENT_ID):
        raise ValueError("Use letters, digits, underscore or hyphen for STUDENT_ID.")
    if CONFIG["storage_dtype"] not in ("float16", "float32"):
        raise ValueError("Store floating-point state as float16 or float32.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    path = Path(DATA_DIR)/"multimodal.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, path)
    if digest(path) != DATA_SHA256:
        raise ValueError("Use multimodal.npz from the research-data repository.")
    folder = Path(out or f"lab-09-{STUDENT_ID}")
    for subfolder in ("samples", "models"):
        (folder/subfolder).mkdir(parents=True, exist_ok=True)
    losses, metrics, representations = {}, {}, {}
    with np.load(path, allow_pickle=False) as data:
        for name, sample in zip(NAMES, SAMPLE_NAMES):
            array = data[name]
            shape, channels = array.shape[:-1], array.shape[-1]
            values = torch.as_tensor(array.reshape(-1, channels), device=device)
            torch.manual_seed(seed)
            model = build_model(len(shape), channels).to(device)
            model_path = folder/"models"/(name+".npz")
            save_model(model, model_path, CONFIG["storage_dtype"])
            print(f"Fitting {name}: {shape}, {channels} channel(s)", flush=True)
            start = time.perf_counter()
            losses[name] = train_signal(model, values, shape, seed)
            size = save_model(model, model_path, CONFIG["storage_dtype"])
            with np.load(model_path, allow_pickle=False) as saved, torch.no_grad():
                for key, value in list(model.named_parameters()) + list(model.named_buffers()):
                    value.copy_(torch.as_tensor(saved[key], device=device, dtype=value.dtype))
            score = reconstruct(model, values, shape, folder/"samples"/sample,
                                int(data["bach_sample_rate"]), tuple(data["cat_fps"]))
            metrics[name+"_psnr_db"] = score["psnr_db"]
            if name == "bach":
                metrics["bach_snr_db"] = score["snr_db"]
            representations[name] = dict(shape=list(shape), channels=channels,
                parameters=sum(p.numel() for p in model.parameters()),
                buffer_elements=sum(b.numel() for b in model.buffers()),
                model_bytes=size, model_sha256=digest(model_path), mse=score["mse"],
                seconds=time.perf_counter()-start)
            print(f"{name}: {score['psnr_db']:.2f} dB, {size:,} bytes", flush=True)
            del model, values
    metrics["mean_psnr_db"] = sum(metrics[n+"_psnr_db"] for n in NAMES)/4
    code, record = folder/"lab-09-code.py", folder/"lab-09-results.json"
    code.write_text(source)
    result = dict(format=7, lab=9, seed=seed, config=CONFIG, loss=losses, metrics=metrics,
        protocol="dl-multimodal-v1-reconstruction", code_sha256=digest(code), data_sha256=DATA_SHA256,
        parameters=max(r["parameters"] for r in representations.values()),
        buffer_elements=max(r["buffer_elements"] for r in representations.values()),
        updates={n: CONFIG["steps"] for n in NAMES}, representations=representations,
        samples_sha256={n: digest(folder/"samples"/n) for n in SAMPLE_NAMES})
    record.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    paths = [code, record] + [folder/"samples"/n for n in SAMPLE_NAMES]
    paths += [folder/"models"/(n+".npz") for n in NAMES]
    archive = folder.with_name(folder.name+".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as pack:
        for path in paths:
            pack.write(path, path.relative_to(folder))
    print(json.dumps(metrics, indent=2), "\nSubmission:", archive.resolve())
    return result


if __name__ == "__main__":
    run()
