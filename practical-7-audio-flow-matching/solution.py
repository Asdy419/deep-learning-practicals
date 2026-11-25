# Abstract
# Flow matching [2, 3] for 128x64 spectrograms of two voices saying the digits one to nine [12]. The sampler
# (64 Euler steps, guidance 1.5 [4]), the audio conversion and the scoring are the template's. I changed the
# network, the training loop, the distribution of training times and two preprocessing steps.
# Network: a U-Net over patches of four frequency bins and one frame (a 32x64 grid) with 48- and 64-channel
# residual blocks, whose normalisation is scaled and shifted by the time, digit and voice embeddings, and
# self-attention at 16x32 [7]; 900676 parameters. Its input is scaled towards unit variance and its time input
# is the log noise level [5]. The patches are one frame wide on purpose: the same network over two-frame
# patches scored about 86 instead of 66, while patches eight bins high and one frame wide scored 68. Averaging
# the two-frame network over the two positions of its patch grid in time (cycle spinning [13]) in the second
# half of sampling recovered more than half of the difference, which points to the two-frame grid itself.
# Training: 10000 updates of batch 64, 500 warm-up updates then cosine decay, and an exponential moving
# average of the weights for sampling [5]. Times are logit-normal [6] with a quarter drawn uniformly: with
# logit-normal times alone the first sampler steps, which decide how long the word is, are hardly trained and
# some words came out up to 8 frames too long. There is no dropout: with only 540 recordings a closer fit
# scored better (about 97 against 113 in an earlier configuration).
# Preprocessing: the fixed Griffin-Lim conversion [8, 9] handles two things in the stored spectrograms badly,
# and the speech encoder [10] behind KAD [11] is very sensitive to both. (1) Magnitudes are clamped at 0.001,
# so the zero padding around each word is a constant -0.76, which becomes faint noise (about three 16-bit
# steps) where the recordings are exactly silent. The real training spectrograms, converted the same way,
# score about 210, and about 65 once their clamped values are lowered by 2. (2) Row 0 is the DC bin. It holds
# slow components below about 30 Hz and no speech: taking it out of the recordings themselves hardly changes
# their score (4 against 2), but what the conversion rebuilds from it costs about 17 points, since the real
# spectrograms score about 46 with row 0 silent as well. I therefore lower the clamped values of the training
# images by 2 and set their row 0 to silence.
# In tuning runs (six sets of starting noise, two seeds each) KAD x 1000 was about 102 with the silence
# lowered and two-frame patches, about 86 with row 0 silent as well, and about 66 with one-frame patches; the
# two-frame network trained on the unmodified images scored about 257, and the template baseline scores 695.
#
# References / Materials used
# [1] C. G. Willcocks and J. Newbould. Deep Learning, Lecture 5 (diffusion and flow matching), Lecture 7
#     (attention) and Practical 7 (Flow matching: from points to speech). Durham University.
# [2] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel and M. Le. Flow matching for generative modeling.
#     ICLR 2023. https://arxiv.org/abs/2210.02747
# [3] X. Liu, C. Gong and Q. Liu. Flow straight and fast: learning to generate and transfer data with
#     rectified flow. ICLR 2023. https://arxiv.org/abs/2209.03003
# [4] J. Ho and T. Salimans. Classifier-free diffusion guidance. 2022. https://arxiv.org/abs/2207.12598
# [5] T. Karras, M. Aittala, T. Aila and S. Laine. Elucidating the design space of diffusion-based
#     generative models. NeurIPS 2022. https://arxiv.org/abs/2206.00364
# [6] P. Esser et al. Scaling rectified flow transformers for high-resolution image synthesis. ICML 2024.
#     https://arxiv.org/abs/2403.03206
# [7] P. Dhariwal and A. Nichol. Diffusion models beat GANs on image synthesis. NeurIPS 2021.
#     https://arxiv.org/abs/2105.05233
# [8] D. Griffin and J. Lim. Signal estimation from modified short-time Fourier transform. IEEE Transactions
#     on Acoustics, Speech, and Signal Processing 32(2), 1984.
# [9] N. Perraudin, P. Balazs and P. L. Sondergaard. A fast Griffin-Lim algorithm. IEEE WASPAA 2013.
# [10] A. Baevski, H. Zhou, A. Mohamed and M. Auli. wav2vec 2.0: a framework for self-supervised learning of
#     speech representations. NeurIPS 2020. https://arxiv.org/abs/2006.11477
# [11] Y. Chung, P. Eu, J. Lee, K. Choi, J. Nam and B. S. Chon. KAD: no more FAD! An effective and efficient
#     evaluation metric for audio generation. ICML 2025. https://arxiv.org/abs/2502.15602
# [12] S. Becker, J. Vielhaben, M. Ackermann, K.-R. Muller, S. Lapuschkin and W. Samek. AudioMNIST: exploring
#     explainable artificial intelligence for audio analysis on a simple benchmark. Journal of the Franklin
#     Institute 361(1), 2024. https://arxiv.org/abs/1807.03418
# [13] R. R. Coifman and D. L. Donoho. Translation-invariant de-noising. In Wavelets and Statistics, Lecture
#     Notes in Statistics 103, Springer, 1995.
#
# AI acknowledgement
# Claude (Anthropic) was used to plan and run the experiments, write the changes to the template code and
# draft these comments.
#
# Set STUDENT_ID, then run this file or paste it into one notebook cell.
import wave
import ast
import copy
import hashlib
import json
import math
import random
import re
import zipfile
from urllib.request import urlretrieve
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STUDENT_ID = "studentID"
DATA_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")

# %% Model and training settings

CONFIG = dict(task="audio_flow", steps=10000, batch_size=64,
              learning_rate=0.002, weight_decay=0.0)

WIDTHS, EMBED, HEADS = (48, 64), 128, 4  # channels at the two resolutions, conditioning size, attention heads
PATCH = (4, 1)  # frequency bins and frames that form one position of the U-Net
WARMUP, EMA_HALFLIFE = 500, 50_000  # warm-up updates; EMA half-life in training images [5]
SILENCE = 2.0  # how far the clamped silence of the training images is lowered

def groups(channels):
    return math.gcd(channels, 32)


def to_patches(x):
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // PATCH[0], PATCH[0], w // PATCH[1], PATCH[1]).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * PATCH[0] * PATCH[1], h // PATCH[0], w // PATCH[1])


def from_patches(x):
    b, c, h, w = x.shape
    x = x.reshape(b, -1, PATCH[0], PATCH[1], h, w).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, -1, h * PATCH[0], w * PATCH[1])


class Attention(nn.Module):
    """Self-attention over all positions of a feature map."""
    def __init__(self, width):
        super().__init__()
        self.norm = nn.GroupNorm(groups(width), width)
        self.qkv = nn.Conv2d(width, 3 * width, 1)
        self.proj = nn.Conv2d(width, width, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, HEADS, c // HEADS, h * w)
        q, k, v = qkv.transpose(-1, -2).unbind(1)
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class Block(nn.Module):
    """Residual block; the conditioning scales and shifts its second normalisation [7]."""
    def __init__(self, cin, cout, attention=False):
        super().__init__()
        self.norm1, self.norm2 = nn.GroupNorm(groups(cin), cin), nn.GroupNorm(groups(cout), cout)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)
        self.condition = nn.Linear(EMBED, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.attention = Attention(cout) if attention else nn.Identity()

    def forward(self, x, e):
        r = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(e)[:, :, None, None].chunk(2, 1)
        r = F.silu(self.norm2(r) * (1 + scale) + shift)
        return self.attention((self.skip(x) + self.conv2(r)) * 0.5 ** 0.5)


class Model(nn.Module):
    """U-Net over 4x1 patches of the spectrogram; predicts the flow velocity."""
    def __init__(self):
        super().__init__()
        fine, coarse = WIDTHS
        self.time = nn.Sequential(nn.Linear(17, EMBED), nn.SiLU(), nn.Linear(EMBED, EMBED))
        self.label = nn.Embedding(10, EMBED)
        self.voice = nn.Embedding(3, EMBED)
        self.input = nn.Conv2d(3 * PATCH[0] * PATCH[1], fine, 3, padding=1)
        self.down = nn.ModuleList([Block(fine, fine), nn.Conv2d(fine, fine, 3, 2, 1),
                                   Block(fine, coarse, attention=True)])
        self.middle = nn.ModuleList([Block(coarse, coarse, attention=True), Block(coarse, coarse)])
        self.up = nn.ModuleList([
            Block(2 * coarse, coarse, attention=True), Block(coarse + fine, coarse, attention=True),
            nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(coarse, coarse, 3, padding=1)),
            Block(coarse + fine, fine), Block(2 * fine, fine)])
        self.output = nn.Sequential(nn.GroupNorm(groups(fine), fine), nn.SiLU(),
                                    nn.Conv2d(fine, PATCH[0] * PATCH[1], 3, padding=1))
        nn.init.zeros_(self.output[-1].weight); nn.init.zeros_(self.output[-1].bias)

    def voice_embedding(self, voice):
        amount = voice[:, None].to(self.voice.weight.dtype).clamp(0, 1)
        blend = torch.lerp(self.voice.weight[0], self.voice.weight[1], amount)
        return torch.where((voice == 2)[:, None], self.voice.weight[2], blend)

    def forward(self, x, t, y, voice):
        # x = (1 - t) noise + t image has noise level (1 - t) / t; its logarithm is the time input [5].
        t = t.clamp(1e-5, 1 - 1e-5)
        c = torch.log((1 - t) / t) / 4
        angles = c[:, None] * 2 ** torch.arange(8, device=t.device)[None] * math.pi
        e = self.time(torch.cat([c[:, None], angles.sin(), angles.cos()], -1))
        e = F.silu(e + self.label(y) + self.voice_embedding(voice))
        x = x * ((1 - t) ** 2 + t ** 2).rsqrt()[:, None, None, None]  # keeps the input variance near one [5]
        f = torch.linspace(-1, 1, x.size(2), device=x.device)
        frequency = f[None, None, :, None].expand_as(x)
        s = torch.linspace(-1, 1, x.size(3), device=x.device)
        position = s[None, None, None, :].expand_as(x)
        h = self.input(to_patches(torch.cat((x, frequency, position), 1)))
        skips = [h]
        for layer in self.down:
            h = layer(h, e) if isinstance(layer, Block) else layer(h)
            skips.append(h)
        for layer in self.middle:
            h = layer(h, e)
        for layer in self.up:
            h = layer(torch.cat([h, skips.pop()], 1), e) if isinstance(layer, Block) else layer(h)
        return from_patches(self.output(h))

def build_model():
    return Model()

MAX_PARAMETERS, MAX_STEPS, MAX_BATCH = 1000000, 10000, 64

# %% Data, evaluation and submission
def program_source():
    if "__file__" in globals():
        source = Path(__file__).read_text()
    else:
        source = get_ipython().history_manager.input_hist_raw[-1].rstrip() + "\n"
    tree = ast.parse(source)
    names = {getattr(node, "name", "") for node in tree.body}
    if not {"build_model", "run", "save_submission"} <= names:
        raise ValueError("Paste the complete program into one notebook cell and run that cell.")
    return source


def model_size(model):
    parameters = sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
    buffers = sum(b.numel()*(2 if b.is_complex() else 1) for b in model.buffers())
    if parameters + buffers > MAX_PARAMETERS:
        raise ValueError(f"Model has {parameters + buffers} values; limit is {MAX_PARAMETERS}.")
    return parameters, buffers


def save_submission(folder, source, history, metrics, size, seed, split):
    stem = "lab-07"
    code = folder / (stem + "-code.py")
    code.write_text(source)
    names = [f"sample-{digit}.wav" for digit in range(1, 10)]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    updates = {"model": CONFIG["steps"]}
    result = dict(format=6, lab=7, seed=seed, config=dict(CONFIG),
                  parameters=size[0], buffer_elements=size[1], updates=updates,
                  protocol="comp3801-audiomnist-voices-kad-v1-" + split, code_sha256=digest(code),
                  metrics=metrics, loss=history,
                  samples_sha256={name: digest(folder/"samples"/name) for name in names})
    record = folder / (stem + "-results.json")
    record.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    archive = folder.with_name(folder.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as pack:
        for path in [code, record] + [folder/"samples"/name for name in names]:
            pack.write(path, path.relative_to(folder))
    print(json.dumps(metrics, indent=2))
    print("KAD x 1000:", round(metrics["kad"]*1000, 3))
    print("Submission:", archive.resolve())
    return result


DATA_SHA256 = '0a5b11b61cf95d0b581b70c5c1bfbdef396912bfbbb80c39584f922dbb2e6bf5'

DATA_URL = "https://raw.githubusercontent.com/cwkx/research-data/main/datasets/audio.npz"

def audio_data(split="train"):
    path = DATA_DIR / "audio.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Use audio.npz from https://github.com/cwkx/research-data/tree/main/datasets")
    data = np.load(path)
    keep = data["split"] == {"train": 0, "validation": 1, "test": 2}[split]
    x = torch.from_numpy(data["spectrograms"][keep].astype(np.float32))[:, None]
    y = torch.from_numpy(data["labels"][keep].astype(np.int64))
    voices = torch.from_numpy(data["voices"][keep].astype(np.int64))
    waves = torch.from_numpy(data["waveforms"][keep].astype(np.float32))/32767
    return x, y, voices, waves


def flow_loss(model, clean, labels, voices):
    labels, voices = labels.clone(), voices.clone()
    hidden = torch.rand(len(labels), device=clean.device) < .1
    labels[hidden], voices[hidden] = 9, 2
    noise = torch.randn_like(clean)
    t = torch.sigmoid(torch.randn(len(clean), device=clean.device))  # logit-normal times [6] ...
    t = torch.where(torch.rand_like(t) < 0.25, torch.rand_like(t), t)  # ... a quarter of them uniform
    time = t[:, None, None, None]
    state = (1-time)*noise + time*clean
    return F.mse_loss(model(state, t, labels, voices), clean-noise)


@torch.no_grad()
def sample(model, labels, voices, start):
    model.eval()
    x = start.clone()
    for k in range(64):
        t = torch.full((len(x),), k/64, device=x.device)
        conditional = model(x, t, labels, voices)
        unconditional = model(x, t, torch.full_like(labels, 9), torch.full_like(voices, 2))
        x = x + (unconditional + 1.5*(conditional-unconditional))/64
    return x[:, 0]


@torch.no_grad()
def to_audio(spectrograms, generator):
    # Reconstruct 8 kHz audio from log magnitudes with Griffin-Lim.
    magnitude = (spectrograms*2.5-5).exp().clamp(0, 100)
    magnitude = F.pad(magnitude, (0, 0, 0, 1))
    window = torch.hann_window(256, device=spectrograms.device)
    angles = 2*torch.pi*torch.rand(magnitude.shape, device=magnitude.device, generator=generator)
    z = magnitude*torch.exp(1j*angles)
    previous = 0
    for _ in range(64):
        waves = torch.istft(z, 256, 128, window=window, length=8064)
        rebuilt = torch.stft(waves, 256, 128, window=window, return_complex=True)
        z = rebuilt - (.99/1.99)*previous
        previous = rebuilt
        z = magnitude*z/(z.abs()+1e-8)
    return torch.istft(z, 256, 128, window=window, length=8064)


def save_audio(path, waves):
    # Each word is followed by 0.2 seconds of silence in the saved WAV.
    pcm = (waves.cpu().clamp(-1, 1)*32767).round().to(torch.int16)
    joined = F.pad(pcm, (0, 1600)).flatten().numpy().astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(8000)
        output.writeframes(joined.tobytes())
    return pcm.float()/32767


@torch.no_grad()
def audio_features(encoder, waves, device):
    import torchaudio
    features = []
    for batch in waves.split(8):
        batch = torchaudio.functional.resample(batch.to(device), 8000, 16000)
        hidden = encoder.extract_features(batch)[0][-1]
        features.append(hidden.mean(1).cpu())
    return torch.cat(features)


def kernel_distance(real, generated):
    # Unbiased squared MMD in fixed speech features; small negative estimates are valid.
    real, generated = real.double(), generated.double()
    bandwidth = torch.pdist(real).median().square().clamp_min(1e-12)
    kernel = lambda a, b: torch.exp(-torch.cdist(a, b).square()/(2*bandwidth))
    rr, gg, rg = kernel(real, real), kernel(generated, generated), kernel(real, generated)
    off_diagonal = lambda k: (k.sum()-k.diag().sum())/(len(k)*(len(k)-1))
    return float(off_diagonal(rr) + off_diagonal(gg) - 2*rg.mean())


@torch.no_grad()
def audio_score(model, folder, device, split):
    import torchaudio
    _, digits, voices, real = audio_data(split)
    generator = torch.Generator(device=device).manual_seed(921)
    endpoints = []
    for digit in range(9):
        start = torch.randn(20, 1, 128, 64, device=device, generator=generator)
        labels = torch.full((40,), digit, device=device, dtype=torch.long)
        voice = torch.tensor([0., 1.], device=device).repeat_interleave(20)
        generated = sample(model, labels, voice, start.repeat(2, 1, 1, 1))
        waves = to_audio(generated, generator).cpu()
        pcm = (waves.clamp(-1, 1)*32767).round()/32767
        endpoints.append(pcm.reshape(2, 20, 8064))
        # One female-to-male transition in each submitted preview.
        labels = torch.full((5,), digit, device=device, dtype=torch.long)
        voice = torch.linspace(0, 1, 5, device=device)
        preview = sample(model, labels, voice, start[:1].repeat(5, 1, 1, 1))
        save_audio(folder/f"sample-{digit+1}.wav", to_audio(preview, generator))
    # Frozen evaluator weights are downloaded once and cached by PyTorch.
    encoder = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model().to(device).eval()
    real_features = audio_features(encoder, real, device)
    scores = [[], []]
    for digit, pair in enumerate(endpoints):
        for voice in (0, 1):
            reference = real_features[(digits == digit) & (voices == voice)]
            features = audio_features(encoder, pair[voice], device)
            scores[voice].append(kernel_distance(reference, features))
    female, male = map(np.mean, scores)
    return dict(kad=float((female+male)/2), kad_female=float(female),
                kad_male=float(male), samples=360)


# %% Training
def train(model, device):
    images, labels, voices, _ = audio_data()
    images, labels, voices = images.to(device), labels.to(device), voices.to(device)
    # The stored images clamp silence at magnitude 0.001; lower those values so the padding stays silent.
    images = torch.where(images <= images.min(), images - SILENCE, images)
    # Row 0 is the DC bin (slow components, no speech); the conversion rebuilds it badly, so silence it.
    images[:, :, 0] = images.min()
    optimiser = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                  weight_decay=CONFIG["weight_decay"])
    total, batch = CONFIG["steps"], CONFIG["batch_size"]

    def factor(step):  # linear warm-up, then cosine decay to zero
        if step < WARMUP:
            return (step + 1) / WARMUP
        return 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / (total - WARMUP)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, factor)
    ema = copy.deepcopy(model).requires_grad_(False)  # exponential moving average of the weights
    history = []
    model.train()
    for step in range(1, total + 1):
        ids = torch.randint(len(images), (batch,), device=device)
        loss = flow_loss(model, images[ids], labels[ids], voices[ids])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimiser.step()
        scheduler.step()
        halflife = min(EMA_HALFLIFE, 0.05 * step * batch)  # shorter while training starts [5]
        with torch.no_grad():
            for average, current in zip(ema.parameters(), model.parameters()):
                average.lerp_(current, 1 - 0.5 ** (batch / halflife))
        if step == 1 or step % max(1, CONFIG["steps"]//40) == 0 or step == CONFIG["steps"]:
            history.append([step, round(loss.item(), 6)])
        if step == 1 or step % max(1, CONFIG["steps"]//10) == 0 or step == CONFIG["steps"]:
            print(f"{step} / {CONFIG['steps']}: loss {loss.item():.4f}", flush=True)
    model.load_state_dict(ema.state_dict())
    return history


def show_results(history, folder):
    import matplotlib.pyplot as plt
    values = np.array(history)
    plt.plot(values[:, 0], values[:, 1])
    plt.xlabel("Update"); plt.ylabel("Flow matching loss"); plt.show()
    if "get_ipython" in globals():
        from IPython.display import Audio, display
        for name in ("sample-1.wav", "sample-7.wav"):
            with wave.open(str(folder/"samples"/name)) as f:
                pcm = np.frombuffer(f.readframes(5*9664), dtype="<i2")
            display(Audio(pcm.astype(np.float32)/32767, rate=8000, normalize=False))


# %% Run the experiment
def run(out=None, device=None, seed=3801, split="validation", show=True):
    source = program_source()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", STUDENT_ID):
        raise ValueError("Use letters, digits, underscore or hyphen for STUDENT_ID.")
    if CONFIG["task"] != "audio_flow":
        raise ValueError("Keep task='audio_flow' in CONFIG.")
    if not 1 <= CONFIG["steps"] <= MAX_STEPS or not 1 <= CONFIG["batch_size"] <= MAX_BATCH:
        raise ValueError("Training steps or batch size exceeds the practical limit.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    folder = Path(out or f"lab-07-{STUDENT_ID}")
    (folder/"samples").mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    size = model_size(model)
    history = train(model, device)
    metrics = audio_score(model, folder/"samples", device, split)
    result = save_submission(folder, source, history, metrics, size, seed, split)
    if show:
        show_results(history, folder)
    return result


if __name__ == "__main__":
    run()
