# Abstract
# Image flow matching [2, 3] for 32x32 CIFAR-10 horses with a U-Net of
# 48-channel residual blocks at 32x32 and 16x16 resolution, with self-attention
# in every 16x16 block [8], 922803 parameters in total. Training times come
# from a logit-normal distribution whose mean is shifted towards low noise
# levels [4, 5]; the input is scaled to unit variance and the network is told
# the log noise level (1 - t) / t [5]. Half the images are mirrored, dropout
# is 0.1, the learning rate warms up over 500 updates and then decays to zero
# on a cosine, and sampling uses an exponential moving average of the weights
# with a half-life of 50000 images [5]. Samples come from the template's Euler
# sampler with 50 steps and autoguidance [6]: the velocity is pushed away from
# a weak version of the same network that skips its two middle residual
# branches [7], so guidance needs no extra parameters. Guidance, with weight
# 1.65, is only applied at noise levels up to 2 [9]. In tuning runs without
# the 16x16 attention, guidance lowered KID x 1000 from about 18 to about 5.
# Adding that attention, which made the samples more varied, and shifting the
# training times lowered the guided score to 2.7 and 3.1 in runs with two
# seeds.
#
# References / Materials used
# [1] C. G. Willcocks. Deep Learning, Lecture 5: Energy-based models (diffusion
#     and flow matching). Durham University.
# [2] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel and M. Le. Flow matching
#     for generative modeling. ICLR 2023. https://arxiv.org/abs/2210.02747
# [3] X. Liu, C. Gong and Q. Liu. Flow straight and fast: learning to generate
#     and transfer data with rectified flow. ICLR 2023.
#     https://arxiv.org/abs/2209.03003
# [4] P. Esser et al. Scaling rectified flow transformers for high-resolution
#     image synthesis. ICML 2024. https://arxiv.org/abs/2403.03206
# [5] T. Karras, M. Aittala, T. Aila and S. Laine. Elucidating the design space
#     of diffusion-based generative models. NeurIPS 2022.
#     https://arxiv.org/abs/2206.00364
# [6] T. Karras, M. Aittala, T. Kynkaanniemi, J. Lehtinen, T. Aila and S. Laine.
#     Guiding a diffusion model with a bad version of itself. NeurIPS 2024.
#     https://arxiv.org/abs/2406.02507
# [7] J. Hyung, K. Kim, S. Hong, M.-J. Kim and J. Choo. Spatiotemporal skip
#     guidance for enhanced video diffusion sampling. CVPR 2025.
#     https://arxiv.org/abs/2411.18664
# [8] P. Dhariwal and A. Nichol. Diffusion models beat GANs on image synthesis.
#     NeurIPS 2021. https://arxiv.org/abs/2105.05233
# [9] T. Kynkaanniemi, M. Aittala, T. Karras, S. Laine, T. Aila and J. Lehtinen.
#     Applying guidance in a limited interval improves sample and distribution
#     quality in diffusion models. NeurIPS 2024. https://arxiv.org/abs/2404.07724
# [10] Deep Learning Practical 6: Diffusion probabilistic modeling. Durham
#     University.
#
# AI acknowledgement
# Claude (Anthropic) was used to plan and run the experiments, write the
# changes to the template code and draft these comments.
#
# Set STUDENT_ID, then run this file or paste it into one notebook cell.
import ast
import copy
import hashlib
import json
import math
import random
import re
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch_fidelity import calculate_metrics  # pip install torch-fidelity
from torch.nn import functional as F

STUDENT_ID = "studentID"
DATA_DIR = Path("data")

# %% Model and training settings

CONFIG = dict(task="flow", steps=20000, batch_size=64,
              learning_rate=0.002, weight_decay=0.0)

WIDTH, LEVELS, BLOCKS, EMBED = 48, 2, 2, 128  # U-Net channels, resolutions, blocks per level, time features
DROPOUT, SIGMA_DATA, SKIP = 0.1, 0.5, 0.5 ** 0.5
WARMUP, EMA_HALFLIFE = 500, 50_000  # warm-up updates; EMA half-life in images [5]
T_MEAN = 0.5  # mean of logit(t); above zero, more training at low noise levels [4, 5]
GUIDANCE, GUIDE_SIGMA = 1.65, 2.0  # guidance weight [6, 7], used at noise levels up to GUIDE_SIGMA [9]
SAMPLE_STEPS = 50


def groups(channels):
    return math.gcd(channels, 32)


class Attention(nn.Module):
    def __init__(self, width, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(groups(width), width)
        self.qkv = nn.Conv2d(width, 3 * width, 1)
        self.proj = nn.Conv2d(width, width, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv.transpose(-1, -2).unbind(1)
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class Block(nn.Module):
    """Residual block; the time embedding scales and shifts its second normalisation [8]."""
    def __init__(self, cin, cout, attention=False):
        super().__init__()
        self.norm1, self.norm2 = nn.GroupNorm(groups(cin), cin), nn.GroupNorm(groups(cout), cout)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)
        self.condition = nn.Linear(EMBED, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.dropout = nn.Dropout(DROPOUT)
        self.attention = Attention(cout) if attention else nn.Identity()

    def forward(self, x, e, weak=False):
        h = self.skip(x)
        if not weak:  # the weak model used for guidance skips this residual branch
            r = self.conv1(F.silu(self.norm1(x)))
            scale, shift = self.condition(e)[:, :, None, None].chunk(2, 1)
            r = F.silu(self.norm2(r) * (1 + scale) + shift)
            h = h + self.conv2(self.dropout(r))
        return self.attention(h * SKIP)


class Model(nn.Module):
    """U-Net of equal-width residual blocks with self-attention at 16x16 resolution [8]."""
    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(17, EMBED), nn.SiLU(), nn.Linear(EMBED, EMBED))
        self.input = nn.Conv2d(3, WIDTH, 3, padding=1)
        self.down = nn.ModuleList()
        for level in range(LEVELS):  # blocks below full resolution also use self-attention
            self.down.extend(Block(WIDTH, WIDTH, attention=level > 0) for _ in range(BLOCKS))
            if level < LEVELS - 1:
                self.down.append(nn.Conv2d(WIDTH, WIDTH, 3, 2, 1))
        self.middle = nn.ModuleList([Block(WIDTH, WIDTH, attention=True), Block(WIDTH, WIDTH)])
        self.up = nn.ModuleList()
        for level in reversed(range(LEVELS)):
            self.up.extend(Block(2 * WIDTH, WIDTH, attention=level > 0) for _ in range(BLOCKS + 1))
            if level > 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(WIDTH, WIDTH, 3, padding=1)))
        self.out = nn.Sequential(nn.GroupNorm(groups(WIDTH), WIDTH), nn.SiLU(), nn.Conv2d(WIDTH, 3, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, y=None, weak=False):
        # x = (1 - t) noise + t data has noise level sigma = (1 - t) / t relative to the data.
        t = t.clamp(1e-5, 1 - 1e-5)
        c = torch.log((1 - t) / t) / 4  # log-sigma time input [5]
        angles = c[:, None] * 2 ** torch.arange(8, device=t.device)[None] * math.pi
        e = F.silu(self.time(torch.cat([c[:, None], angles.sin(), angles.cos()], -1)))
        x = x / ((1 - t) ** 2 + (t * SIGMA_DATA) ** 2).sqrt()[:, None, None, None]  # unit variance [5]
        h = self.input(x)
        hs = [h]
        for layer in self.down:
            h = layer(h, e) if isinstance(layer, Block) else layer(h)
            hs.append(h)
        for layer in self.middle:
            h = layer(h, e, weak)
        for layer in self.up:
            h = layer(torch.cat([h, hs.pop()], 1), e) if isinstance(layer, Block) else layer(h)
        return self.out(h)


def build_model():
    return Model()

MAX_PARAMETERS, MAX_STEPS, MAX_BATCH = (1000000, 20000, 64)

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
    stem = "lab-06"
    code = folder / (stem + "-code.py")
    code.write_text(source)
    names = [f"batch-{i:02d}.png" for i in range(1, 11)]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    updates = {"model": CONFIG["steps"]}
    result = dict(format=6, lab=6, seed=seed, config=dict(CONFIG),
                  parameters=size[0], buffer_elements=size[1], updates=updates,
                  protocol="comp3801-cifar10-horses-kid1000-v1-" + split, code_sha256=digest(code),
                  metrics=metrics, loss=history,
                  samples_sha256={name: digest(folder/"samples"/name) for name in names})
    record = folder / (stem + "-results.json")
    record.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    archive = folder.with_name(folder.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as pack:
        for path in [code, record] + [folder/"samples"/name for name in names]:
            pack.write(path, path.relative_to(folder))
    print(json.dumps(metrics, indent=2))
    print("Submission:", archive.resolve())
    return result


def show_results(history, folder):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 2.5))
    curves = np.array(history)
    plt.plot(curves[:, 0], curves[:, 1])
    plt.xlabel("Update"); plt.ylabel("Loss"); plt.tight_layout(); plt.show()
    from PIL import Image
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(folder/"samples/batch-01.png"))
    plt.axis("off"); plt.tight_layout(); plt.show()


def image_data(split="train"):
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(root=DATA_DIR, train=split != "test", download=True)
    index = torch.where(torch.tensor(dataset.targets) == 7)[0]
    if split != "test":
        order = torch.randperm(len(index), generator=torch.Generator().manual_seed(3802))
        index = index[order[:4000] if split == "train" else order[4000:]]
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2)[index].float()/127.5 - 1
    return images, torch.zeros(len(images), dtype=torch.long)


def image_bytes(images):
    return ((images.clamp(-1, 1) + 1)*127.5 + .5).floor().to(torch.uint8)


class Images(torch.utils.data.Dataset):
    """Give the KID library one uint8 RGB image at a time."""
    def __init__(self, pixels): self.pixels = pixels
    def __len__(self): return len(self.pixels)
    def __getitem__(self, index): return self.pixels[index]


@torch.no_grad()
def image_score(model, folder, device, split):
    from PIL import Image
    model.eval()
    rng = torch.Generator(device=device).manual_seed(921)
    fake = torch.cat([generate(model, 50, device, rng).clamp(-1, 1).cpu() for _ in range(20)])
    real, _ = image_data(split)
    pixels = image_bytes(fake)
    # One full set of 1,000 images, using the same pixels as the saved grids.
    metrics = calculate_metrics(
        input1=Images(image_bytes(real)), input2=Images(pixels),
        cuda=torch.device(device).type == "cuda", batch_size=32,
        kid=True, kid_subset_size=1000, kid_subsets=1,
        feature_extractor="inception-v3-compat", feature_layer_kid="2048",
        rng_seed=921, save_cpu_ram=True, verbose=False)
    score = metrics["kernel_inception_distance_mean"]
    for i, batch in enumerate(pixels.split(100), 1):
        grid = batch.reshape(10, 10, 3, 32, 32).permute(0, 3, 1, 4, 2).reshape(320, 320, 3)
        Image.fromarray(grid.numpy()).save(folder/f"batch-{i:02d}.png")
    print(f"KID x 1000: {1000*score:.3f}")
    return {"kid": score, "samples": 1000}


def diffusion_schedule(steps=100, device='cpu'):
    # The cosine schedule approaches pure noise within 100 diffusion steps.
    grid = torch.linspace(0, 1, steps+1, device=device, dtype=torch.float64)
    cumulative = torch.cos((grid+.008)/1.008 * torch.pi/2).square()
    cumulative = cumulative/cumulative[0]
    betas = (1-cumulative[1:]/cumulative[:-1]).clamp(max=.999).float()
    alphas = 1-betas
    bars = torch.cumprod(alphas, dim=0)
    previous = torch.cat([torch.ones(1, device=device), bars[:-1]])
    variance = betas*(1-previous)/(1-bars)
    return betas, alphas, bars, variance

def add_noise(clean, epsilon, t, bars):
    a = bars[t].reshape(-1, 1, 1, 1)
    return a.sqrt()*clean + (1-a).sqrt()*epsilon

def diffusion_loss(model, clean, bars, labels=None):
    t = torch.randint(len(bars), (len(clean),), device=clean.device)
    epsilon = torch.randn_like(clean)
    noisy = add_noise(clean, epsilon, t, bars)
    # Integer index 0 denotes mathematical diffusion step 1.
    prediction = model(noisy, (t.float()+1)/len(bars), labels) if labels is not None else model(noisy, (t.float()+1)/len(bars))
    return F.mse_loss(prediction, epsilon)

def flow_path(noise, data, t):
    t = t.reshape(-1, 1, 1, 1)
    return (1-t)*noise+t*data, data-noise

def flow_loss(model, data, labels):
    noise = torch.randn_like(data)
    t = torch.sigmoid(T_MEAN + torch.randn(len(data), device=data.device))  # logit-normal times [4]
    x, target = flow_path(noise, data, t)
    return F.mse_loss(model(x, t, labels), target)

@torch.no_grad()
def sample_ddpm(model, count, schedule, device='cpu', generator=None, labels=None, shape=(3, 32, 32)):
    betas, alphas, bars, variance = schedule
    x = torch.randn(count, *shape, device=device, generator=generator)
    for i in reversed(range(len(betas))):
        t = torch.full((count,), (i+1)/len(betas), device=device)
        epsilon = model(x, t, labels) if labels is not None else model(x, t)
        # Predict clean x0, clip to the known data support, then use q(x_{t-1}|x_t,x0).
        clean = ((x-(1-bars[i]).sqrt()*epsilon)/bars[i].sqrt()).clamp(-1, 1)
        prev = bars[i-1] if i else torch.ones((), device=device)
        mean = (prev.sqrt()*betas[i]/(1-bars[i]))*clean
        mean = mean + (alphas[i].sqrt()*(1-prev)/(1-bars[i]))*x
        if i:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = mean + variance[i].sqrt()*noise
        else:
            x = mean
    return x.clamp(-1, 1)

@torch.no_grad()
def sample_flow(model, labels, steps=25, device='cpu', generator=None, shape=(3, 32, 32)):
    if steps < 1:
        raise ValueError('sampling steps must be positive')
    x = torch.randn(len(labels), *shape, device=device, generator=generator)
    dt = 1/steps
    for i in range(steps):
        t = torch.full((len(labels),), i/steps, device=device)
        v = model(x, t, labels)
        if 1 - i/steps <= GUIDE_SIGMA * i/steps:  # noise level (1 - t) / t is within the interval [9]
            # Guide away from the weak model that skips the middle residual branches [6, 7].
            weak = model(x, t, labels, weak=True)
            v = weak + GUIDANCE * (v - weak)
        x = x + dt*v
    return x.clamp(-1, 1)

# %% Training
def train(model, device):
    images, labels = image_data()
    images, labels = images.to(device), labels.to(device)
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
        index = torch.randint(len(images), (batch,), device=device)
        x, y = images[index], labels[index]
        x = torch.where(torch.rand(len(x), 1, 1, 1, device=device) < .5, x.flip(3), x)  # mirror half
        loss = flow_loss(model, x, y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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


@torch.no_grad()
def generate(model, count, device, generator):
    labels = torch.zeros(count, dtype=torch.long, device=device)
    if CONFIG["task"] == "ddpm":
        return sample_ddpm(model, count, diffusion_schedule(device=device), device,
                           generator, labels, shape=(3, 32, 32))
    return sample_flow(model, labels, SAMPLE_STEPS, device, generator, shape=(3, 32, 32))

# %% Run the experiment
def run(out=None, device=None, seed=3801, split="validation", show=True):
    source = program_source()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", STUDENT_ID):
        raise ValueError("Use letters, digits, underscore or hyphen for STUDENT_ID.")
    if CONFIG["task"] not in ("ddpm", "flow"):
        raise ValueError("Set task to 'ddpm' or 'flow'.")
    if not 1 <= CONFIG["steps"] <= MAX_STEPS or not 1 <= CONFIG["batch_size"] <= MAX_BATCH:
        raise ValueError("Training steps or batch size exceeds the practical limit.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    if torch.device(device).type == "cuda":
        torch.backends.cudnn.benchmark = True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    folder = Path(out or f"lab-06-{STUDENT_ID}")
    (folder/"samples").mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    size = model_size(model)
    history = train(model, device)
    metrics = image_score(model, folder/"samples", device, split)
    result = save_submission(folder, source, history, metrics, size, seed, split)
    if show:
        show_results(history, folder)
    return result


if __name__ == "__main__":
    run()
