# Abstract
# Expanding on the file from practical 5, we produce a DCGAN for 32x32
# CIFAR-10 horses that keeps the supplied loss, Adam settings, batch size and
# one update of each network per iteration. The generator normalises its
# projected 8x8 features and utilizes nearest neighbour upsampling with 3x3
# convolutions. Both networks are wider, with 96 and 128 channels, 925734
# parameters and buffers. The learning rate decays linearly to zero over the
# second half of the 10000 iterations. Half the real images are mirrored, and
# every image the discriminator sees whether that be generated or real, is
# shifted by up to 4 pixels. Adaptive discriminator augmentation (ADA) [4, 5]
# adds differentiable geometric and colour transforms with probability p,
# adjusted so the mean sign of the discriminator's outputs on real images
# tracks a target. Shifts keep that signal below the reference paper's target
# of 0.6, so 0.3 is used. ADA then switches on in the second half and ends
# near p=0.1. From my testing, my runs scored KID x 1000 of 29.3 to 39.4
# (mean 35.2), against a mean of 39.0 without ADA and 203.5 for the unmodified
# starter on this machine.
#
# References / Materials used
# [1] C. G. Willcocks. Deep Learning, Lecture 3: Designing architectures.
#     Durham University.
# [2] C. G. Willcocks. Deep Learning, Lecture 4: Adversarial models. Durham
#     University.
# [3] Deep Learning Practical 5: GANs, from digits to horses. Durham University.
# [4] T. Karras, M. Aittala, J. Hellsten, S. Laine, J. Lehtinen and T. Aila.
#     Training generative adversarial networks with limited data. NeurIPS
#     2020. https://arxiv.org/abs/2006.06676
# [5] NVIDIA. StyleGAN2-ADA, official PyTorch implementation of [4].
#     https://github.com/NVlabs/stylegan2-ada-pytorch
#
# AI acknowledgement
# Claude helped with tuning hyperparameters and optimizing KID x 1000 scores, code was
# written by myself largely however, with minor fixes, comments and readability changes generated from claude. 
# 
#
# Set STUDENT_ID, then run this file or paste it into one notebook cell.
import ast
import hashlib
import json
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

CONFIG = dict(task="dcgan", steps=10000, batch_size=64,
              learning_rate=0.001, weight_decay=0.0)


class Generator(nn.Module):
    def __init__(self, latent=16, width=96):
        super().__init__()
        self.latent = latent
        self.project = nn.Linear(latent, width * 2 * 8 * 8)
        self.net = nn.Sequential(
            nn.BatchNorm2d(width * 2), nn.ReLU(),            # normalise projected features [3]
            nn.Upsample(scale_factor=2),                     # 8 -> 16 px, nearest neighbour [3]
            nn.Conv2d(width * 2, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(),
            nn.Upsample(scale_factor=2),                     # 16 -> 32 px
            nn.Conv2d(width, 3, 3, 1, 1), nn.Tanh())

    def forward(self, z):
        return self.net(self.project(z).reshape(z.size(0), -1, 8, 8))


class Discriminator(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, width, 4, 2, 1), nn.LeakyReLU(.2),
                                 nn.Conv2d(width, width * 2, 4, 2, 1), nn.LeakyReLU(.2),
                                 nn.Flatten(), nn.Linear(width * 2 * 8 * 8, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits; no sigmoid before BCE with logits


class Model(nn.Module):
    def __init__(self, generator_width=96, discriminator_width=128, latent=16):
        super().__init__()
        self.generator = Generator(latent=latent, width=generator_width)
        self.discriminator = Discriminator(width=discriminator_width)


def build_model():
    return Model()

MAX_PARAMETERS, MAX_STEPS, MAX_BATCH = (1000000, 10000, 64)

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
    stem = "lab-05"
    code = folder / (stem + "-code.py")
    code.write_text(source)
    names = [f"batch-{i:02d}.png" for i in range(1, 11)]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    updates = {"generator": CONFIG["steps"], "discriminator": CONFIG["steps"]}
    result = dict(format=6, lab=5, seed=seed, config=dict(CONFIG),
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
    plt.plot(curves[:, 0], curves[:, 1], label="Generator")
    plt.plot(curves[:, 0], curves[:, 2], label="Discriminator")
    plt.legend()
    plt.xlabel("GAN iteration"); plt.ylabel("Loss"); plt.tight_layout(); plt.show()
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


def discriminator_loss(real_logits, fake_logits):
    return (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))

def generator_loss(fake_logits):
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

# %% Training
def translate(images, shift=4):
    """Differentiable augmentation [2]: move each image up to `shift` pixels in each
    direction, filling the gap with zeros. Real and generated images are both moved
    before the discriminator sees them, and gradients still reach the generator."""
    n, _, size, _ = images.shape
    i = torch.arange(size, device=images.device)
    dy, dx = torch.randint(2 * shift + 1, (2, n, 1, 1), device=images.device)
    padded = F.pad(images, (shift,) * 4).permute(0, 2, 3, 1)
    moved = padded[torch.arange(n, device=images.device)[:, None, None], i[:, None] + dy, i + dx]
    return moved.permute(0, 3, 1, 2)


def augment(images, p):
    """Adaptive discriminator augmentation [4]: the authors' 'bgc' transforms at their
    default strengths, each applied to each image with probability p. The geometric
    changes share one reflection-padded resample and the colour changes one matrix, so
    every step is differentiable and gradients still reach the generator."""
    if p == 0:
        return images
    n, size, dev = len(images), images.shape[-1], images.device
    one, zero = torch.ones(n, device=dev), torch.zeros(n, device=dev)
    pick = lambda value, identity: torch.where(torch.rand(n, device=dev) < p, value, identity)
    normal = lambda std: torch.randn(n, device=dev) * std
    uniform = lambda: torch.rand(n, device=dev) * 2 - 1

    def affine(a, b, c, d, e, f):  # one 3x3 matrix per image
        return torch.stack([torch.stack([a, b, c], -1), torch.stack([d, e, f], -1),
                            torch.stack([zero, zero, one], -1)], 1)

    scale = lambda sx, sy: affine(sx, zero, zero, zero, sy, zero)
    rotate = lambda t: affine(t.cos(), -t.sin(), zero, t.sin(), t.cos(), zero)
    shift = lambda tx, ty: affine(one, zero, tx, zero, one, ty)
    p_rot = 1 - max(0., 1 - p) ** .5  # two rotations, so either happens with probability p
    turn = lambda: torch.where(torch.rand(n, device=dev) < p_rot, uniform() * np.pi, zero)

    # Output-to-input coordinates, with the image spanning [-1, 1]. Every distribution is
    # symmetric, so composing these gives the inverse of a random transform.
    g = scale(pick(1 - 2 * torch.randint(2, (n,), device=dev).float(), one), one)  # x-flip
    g = g @ rotate(pick(torch.randint(4, (n,), device=dev) * np.pi / 2, zero))      # quarter turns
    t = torch.round(torch.stack([uniform(), uniform()]) * size / 8) * 2 / size      # whole pixels
    g = g @ shift(*pick(t, 0))                                                      # integer shift
    s = pick(torch.exp2(normal(.2)), one)
    g = g @ scale(s, s) @ rotate(turn())                                            # scale, rotate
    s = pick(torch.exp2(normal(.2)), one)
    g = g @ scale(s, 1 / s) @ rotate(turn())                                        # stretch, rotate
    g = g @ shift(*pick(torch.stack([normal(.25), normal(.25)]), 0))                # fractional shift
    grid = F.affine_grid(g[:, :2], list(images.shape), align_corners=False)
    images = F.grid_sample(images, grid, padding_mode="reflection", align_corners=False)

    v = torch.full((3, 1), 3 ** -.5, device=dev)  # luma axis
    vv, eye = v @ v.T, torch.eye(3, device=dev)
    cross = torch.tensor([[0., -1, 1], [1, 0, -1], [-1, 1, 0]], device=dev) * 3 ** -.5
    m = torch.eye(3, 4, device=dev).repeat(n, 1, 1)                                 # [matrix | offset]
    m[:, :, 3] += pick(normal(.2), zero)[:, None]                                   # brightness
    m = m * pick(torch.exp2(normal(.5)), one)[:, None, None]                        # contrast
    flip = pick(torch.randint(2, (n,), device=dev).float(), zero)[:, None, None]
    m = (eye - 2 * vv * flip) @ m                                                   # luma flip
    t = pick(uniform() * np.pi, zero)[:, None, None]
    m = (eye * t.cos() + cross * t.sin() + vv * (1 - t.cos())) @ m                  # hue rotation
    m = (vv + (eye - vv) * pick(torch.exp2(normal(1)), one)[:, None, None]) @ m     # saturation
    return (m[:, :, :3] @ images.flatten(2) + m[:, :, 3:]).reshape(images.shape)


def train(model, device):
    images, _ = image_data()
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=CONFIG["learning_rate"],
                             betas=(.5, .999), weight_decay=CONFIG["weight_decay"])
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=CONFIG["learning_rate"],
                             betas=(.5, .999), weight_decay=CONFIG["weight_decay"])

    def anneal(step):  # full rate for the first half, then linear decay to zero [1]
        return min(1.0, 2 * (1 - step / CONFIG["steps"]))

    schedules = [torch.optim.lr_scheduler.LambdaLR(opt, anneal) for opt in (opt_g, opt_d)]
    p, target, signs = 0.0, 0.3, torch.zeros((), device=device)  # ADA state [4]
    history = []
    model.train()
    for step in range(1, CONFIG["steps"] + 1):
        real = images[torch.randint(len(images), (CONFIG["batch_size"],))].to(device)
        mirror = torch.rand(len(real), 1, 1, 1, device=device) < .5   # flip half the horses [1]
        real = torch.where(mirror, real.flip(3), real)
        noise = torch.randn(len(real), model.generator.latent, device=device)
        fake = model.generator(noise)
        opt_d.zero_grad(set_to_none=True)
        real_logits = model.discriminator(augment(translate(real), p))
        d_loss = discriminator_loss(real_logits,
                                    model.discriminator(augment(translate(fake.detach()), p)))
        d_loss.backward(); opt_d.step()
        signs += real_logits.detach().sign().sum()
        for parameter in model.discriminator.parameters(): parameter.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        g_loss = generator_loss(model.discriminator(augment(translate(fake), p)))
        g_loss.backward(); opt_g.step()
        for parameter in model.discriminator.parameters(): parameter.requires_grad_(True)
        for schedule in schedules:
            schedule.step()
        if step % 4 == 0:  # nudge p so the overfitting signal E[sign(D(real))] tracks the target [4]
            r = signs.item() / (4 * len(real)); signs.zero_()
            p = max(0., p + ((r > target) - (r < target)) * 4 * len(real) / 100_000)
        if step == 1 or step % max(1, CONFIG["steps"]//40) == 0 or step == CONFIG["steps"]:
            history.append([step, round(g_loss.item(), 6), round(d_loss.item(), 6)])
        if step == 1 or step % max(1, CONFIG["steps"]//10) == 0 or step == CONFIG["steps"]:
            print(f"{step} / {CONFIG['steps']}: generator loss {g_loss.item():.4f}, "
                  f"discriminator loss {d_loss.item():.4f}, ADA p {p:.3f}", flush=True)
    return history


@torch.no_grad()
def generate(model, count, device, generator):
    return model.generator(torch.randn(count, model.generator.latent, device=device, generator=generator))

# %% Run the experiment
def run(out=None, device=None, seed=3801, split="validation", show=True):
    source = program_source()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", STUDENT_ID):
        raise ValueError("Use letters, digits, underscore or hyphen for STUDENT_ID.")
    if CONFIG["task"] != "dcgan":
        raise ValueError("Keep task='dcgan' in CONFIG.")
    if not 1 <= CONFIG["steps"] <= MAX_STEPS or not 1 <= CONFIG["batch_size"] <= MAX_BATCH:
        raise ValueError("Training steps or batch size exceeds the practical limit.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    if torch.device(device).type == "cuda":
        torch.backends.cudnn.benchmark = True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    folder = Path(out or f"lab-05-{STUDENT_ID}")
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
