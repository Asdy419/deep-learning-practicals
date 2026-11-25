# Abstract
# A six-block GPT of width 112 [2, 3] (940,726 parameters) predicts the next byte of Shakespeare from a
# 256-byte context. The training text has only 0.9 million bytes and each byte is seen about 180 times, so
# every model I tried memorises it and is over-confident on unseen text. Three changes address that.
# (1) Dropout [5] is never switched off: when the model is not training it returns the average of the
# probabilities predicted under 64 dropout masks (Monte Carlo dropout [8]), an ensemble [1] of thinned
# networks that costs no parameters and needs no fitted temperature.
# (2) Data augmentation [1]. Speaker names never seen in training are 2% of the validation bytes but 5% of
# the loss, and another 6% of the bytes are in words never seen in training: a model that has memorised
# its text does not expect either. So in every training window one in ten of the words of four or more
# letters is replaced by a made-up word (the start of one training word joined to the end of another), and
# in 30% of the windows every speaker gets a made-up name, as does every character that a speech mentions.
# (3) A copy head [11, 12] attends to earlier positions whose final state resembles the current one and
# predicts the bytes that followed them; a learned gate mixes this with the usual prediction, so a new name
# can be copied after its first appearance.
# The rest: rotary positions [9], a gated MLP [10], tied embeddings [4], small initial weights [3], AdamW [6]
# at a constant rate after warm-up and an exponential moving average of the weights [7].
# Validation NLL in tuning runs, each step added to the last: six blocks with dropout 0.3 and cosine decay
# 1.42; constant rate and weight average 1.40; mask averaging 1.365; rotary positions 1.338; gated MLP and
# learning rate 0.003 1.324; renamed speakers and copy head 1.297; names in speeches renamed too 1.291;
# made-up words, which allow dropout 0.15, 1.281 (three seeds). The starter scores 1.783.
#
# References / Materials used
# [1] C. G. Willcocks and J. Newbould. Deep Learning, Lecture 3 (regularisation: data augmentation,
#     dropout, ensembles), Lecture 7 (attention and transformers) and Practical 8. Durham University.
# [2] A. Vaswani et al. Attention is all you need. NeurIPS 2017. https://arxiv.org/abs/1706.03762
# [3] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei and I. Sutskever. Language models are unsupervised
#     multitask learners. OpenAI, 2019; and A. Karpathy, nanoGPT, https://github.com/karpathy/nanoGPT
# [4] O. Press and L. Wolf. Using the output embedding to improve language models. EACL 2017.
#     https://arxiv.org/abs/1608.05859
# [5] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov. Dropout: a simple way to
#     prevent neural networks from overfitting. Journal of Machine Learning Research 15, 2014.
# [6] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. ICLR 2019.
#     https://arxiv.org/abs/1711.05101
# [7] B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM Journal
#     on Control and Optimization 30(4), 1992.
# [8] Y. Gal and Z. Ghahramani. Dropout as a Bayesian approximation: representing model uncertainty in deep
#     learning. ICML 2016. https://arxiv.org/abs/1506.02142
# [9] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen and Y. Liu. RoFormer: enhanced transformer with rotary
#     position embedding. 2021. https://arxiv.org/abs/2104.09864
# [10] N. Shazeer. GLU variants improve transformer. 2020. https://arxiv.org/abs/2002.05202
# [11] E. Grave, A. Joulin and N. Usunier. Improving neural language models with a continuous cache.
#     ICLR 2017. https://arxiv.org/abs/1612.04426
# [12] S. Merity, C. Xiong, J. Bradbury and R. Socher. Pointer sentinel mixture models. ICLR 2017.
#     https://arxiv.org/abs/1609.07843
#
# AI acknowledgement
# Claude (Anthropic) was used to plan and run the experiments, write the changes to the template code and
# draft these comments.
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
from torch.nn import functional as F

STUDENT_ID = "studentID"
DATA_DIR = Path("data")

# %% Model and training settings

CONFIG = dict(task="gpt", steps=10000, batch_size=64,
              learning_rate=0.003, weight_decay=0.1)

WIDTH, HEADS, LAYERS, DROPOUT = 112, 7, 6, 0.15
WARMUP, EMA_HALFLIFE = 200, 500  # warm-up updates; half-life of the weight average, in updates
MASKS = 64  # dropout masks whose predictions are averaged when the model is not training
RENAMED = 0.3  # share of training windows in which the characters get made-up names


def rotate(x):
    """Rotary position encoding [9]: turn each pair of features by an angle proportional to the position."""
    length, width = x.shape[-2:]
    rate = 10000.0 ** (-torch.arange(0, width, 2, device=x.device) / width)
    angle = torch.arange(length, device=x.device)[:, None] * rate
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a*angle.cos() - b*angle.sin(), a*angle.sin() + b*angle.cos()), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3*width, bias=False)
        self.output = nn.Linear(width, width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [a.reshape(batch, length, self.heads, width//self.heads)
                   .transpose(1, 2) for a in (q, k, v)]
        h = F.scaled_dot_product_attention(rotate(q), rotate(k), v, is_causal=True)
        return self.output(h.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        hidden = round(8*width/3)  # a gated MLP [10] with as many weights as the starter's MLP
        self.expand, self.project = nn.Linear(width, 2*hidden), nn.Linear(hidden, width)

    def forward(self, x):
        # Dropout [5] on both residual branches. It is never switched off: see Model.forward.
        x = x + F.dropout(self.attention(self.norm1(x)), DROPOUT)
        gate, value = self.expand(self.norm2(x)).chunk(2, dim=-1)
        return x + F.dropout(self.project(F.silu(gate)*value), DROPOUT)


class Model(nn.Module):
    def __init__(self, width=WIDTH, heads=HEADS, layers=LAYERS):
        super().__init__()
        self.token = nn.Embedding(256, width)
        self.blocks = nn.Sequential(*(Block(width, heads) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 256)
        self.copy = nn.Linear(width, 1)  # how much to trust the copy head at each position
        self.sharpness = nn.Parameter(torch.tensor(math.log(10.0)))
        for module in self.modules():  # small initial weights [3]
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        for block in self.blocks:  # smaller still where a branch re-enters the residual stream
            for last in (block.attention.output, block.project):
                nn.init.normal_(last.weight, std=0.02/math.sqrt(2*layers))
        nn.init.zeros_(self.copy.weight)
        nn.init.constant_(self.copy.bias, -4.0)  # the copy head starts almost switched off
        self.head.weight = self.token.weight  # input and output embeddings share one matrix [4]

    def network(self, tokens):
        """Log-probabilities of the next byte at every position, under one dropout mask."""
        length = tokens.size(1)
        h = self.norm(self.blocks(self.token(tokens)))
        logits = self.head(h)
        if length < 2:
            return logits.float().log_softmax(-1)
        with torch.autocast(tokens.device.type, enabled=False):
            # Copy head [11]: attend to earlier positions whose final state resembles the current one and
            # predict the bytes that followed them; a learned gate mixes this with the usual prediction.
            h = h.float()
            key = F.normalize(h, dim=-1)
            index = torch.arange(length, device=tokens.device)
            earlier = index[None, :] < index[:, None]
            score = (key @ key.transpose(1, 2)) * self.sharpness.exp()
            weight = score.masked_fill(~earlier, float("-inf")).softmax(-1).nan_to_num()
            follow = tokens[:, None, 1:].expand(-1, length, -1)  # the byte after each earlier position
            copied = torch.zeros_like(logits, dtype=torch.float32).scatter_add(2, follow, weight[:, :, :-1])
            gate = torch.sigmoid(self.copy(h)) * (index > 0)[:, None]
            mixed = (1 - gate) * logits.float().softmax(-1) + gate * copied
            return mixed.clamp_min(1e-30).log()

    def forward(self, tokens):
        if self.training:
            return self.network(tokens)
        # Monte Carlo dropout [8]: the log of the probabilities averaged over MASKS dropout masks,
        # computed for at most 512 copies of a window at a time.
        batch, length = tokens.shape
        group = max(1, 512//batch)
        parts = [self.network(tokens.repeat(min(group, MASKS - done), 1)).view(-1, batch, length, 256)
                 for done in range(0, MASKS, group)]
        return torch.cat(parts).logsumexp(0) - math.log(MASKS)


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
    stem = "lab-08"
    code = folder / (stem + "-code.py")
    code.write_text(source)
    names = [f"batch-{i:02d}.txt" for i in range(1, 11)]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    updates = {"model": CONFIG["steps"]}
    result = dict(format=6, lab=8, seed=seed, config=dict(CONFIG),
                  parameters=size[0], buffer_elements=size[1], updates=updates,
                  protocol="comp3801-gpt-context256-v1-" + split, code_sha256=digest(code),
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
    print((folder/"samples/batch-01.txt").read_text(encoding="utf-8", errors="replace"))


def download(url, path, expected):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        from urllib.request import urlretrieve
        temporary = path.with_suffix(".download")
        try:
            urlretrieve(url, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError("Download checksum differs: " + str(path))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Cached file checksum differs: " + str(path))
    return path

def text_data(split="train"):
    path = download("https://raw.githubusercontent.com/cwkx/research-data/main/datasets/shakespeare.txt",
                    DATA_DIR/"tinyshakespeare.txt",
                    "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed")
    data = path.read_bytes()
    a, b = int(.8*len(data)), int(.9*len(data))
    parts = {"train": data[:a], "validation": data[a:b], "test": data[b:]}
    return torch.tensor(list(parts[split]), dtype=torch.long)

def text_batch(tokens, batch, context=256, generator=None):
    if len(tokens) <= context:
        raise ValueError('text split is too short')
    starts = torch.randint(len(tokens)-context, (batch,), device=tokens.device, generator=generator)
    positions = torch.arange(context+1, device=tokens.device)
    windows = tokens[starts[:, None] + positions]
    return windows[:, :-1], windows[:, 1:]

@torch.no_grad()
def language_metrics(model, tokens, device='cpu', batch=32, context=256):
    """Evaluate next-byte loss in non-overlapping windows, counting each target once."""
    model.eval()
    x_windows, y_windows = [], []
    total, count = 0.0, 0
    for start in range(0, len(tokens)-1, context):
        end = min(start+context, len(tokens)-1)
        x_windows.append(tokens[start:end])
        y_windows.append(tokens[start+1:end+1])
        # Group windows by length so the shorter final window is included in the score.
        if len(x_windows) == batch or end == len(tokens)-1:
            groups = {}
            for x, y in zip(x_windows, y_windows):
                groups.setdefault(len(x), []).append((x, y))
            for length, group in groups.items():
                x = torch.stack([p[0] for p in group]).to(device)
                y = torch.stack([p[1] for p in group]).to(device)
                logits = model(x)
                total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum'))
                count += y.numel()
            x_windows, y_windows = [], []
    if not count:
        raise ValueError('at least two tokens required')
    nll = total/count
    return {'nll_nats_per_byte': nll, 'bits_per_byte': nll/math.log(2),
            'perplexity': math.exp(nll), 'target_bytes': count, 'context': context}

# %% Training
SPEAKER = re.compile(rb"\n\n([A-Z][A-Z ]{2,30}):\n")
WORD = re.compile(rb"(?<![A-Za-z'])[A-Za-z][a-z]{3,}(?![A-Za-z'])")


def splice(pool):
    """A made-up string: the start of one entry of the pool joined to the end of another."""
    first, second = random.choice(pool), random.choice(pool)
    return first[:random.randint(1, len(first) - 1)] + second[random.randint(1, len(second) - 1):]


def disguised(text, names, words, mentions, rename):
    """257 bytes of training text in which one in ten of the words of four or more letters is replaced by
    a made-up word. With rename, every speaker (a name in capitals on its own line) also gets a made-up
    name, and so does every character that the speeches mention by name."""
    start = random.randrange(len(text) - 400)
    window = text[start:start + 400]
    if rename:
        new = {}
        for name in set(SPEAKER.findall(window)):
            new[name] = splice(names)
            window = window.replace(b"\n\n" + name + b":\n", b"\n\n" + new[name] + b":\n")
        for word in set(mentions.findall(window)):
            name = new.get(word.upper()) or splice(names)
            if b" " not in name:
                window = re.sub(rb"(?<![A-Za-z'])" + word + rb"(?![A-Za-z])", name.title(), window)
    for word in set(WORD.findall(window)):
        if random.random() < 0.1:
            new = splice(words).capitalize() if word[:1].isupper() else splice(words)
            window = re.sub(rb"(?<![A-Za-z'])" + word + rb"(?![A-Za-z'])", new, window)
    return list(window[:257])


def train(model, device):
    text = bytes(text_data().tolist())
    names, words = sorted(set(SPEAKER.findall(text))), sorted(set(WORD.findall(text.lower())))
    single = sorted((name.title() for name in names if b" " not in name), key=len, reverse=True)
    mentions = re.compile(rb"(?<![A-Za-z'])(" + b"|".join(single) + rb")(?![A-Za-z])")
    matrices = [p for name, p in model.named_parameters() if name.startswith("blocks") and p.dim() > 1]
    others = [p for name, p in model.named_parameters() if not (name.startswith("blocks") and p.dim() > 1)]
    optimiser = torch.optim.AdamW([dict(params=matrices, weight_decay=CONFIG["weight_decay"]),
                                   dict(params=others, weight_decay=0.0)],
                                  lr=CONFIG["learning_rate"], betas=(0.9, 0.99))
    # Linear warm-up, then a constant rate: the weight average below does the smoothing.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: min(1.0, (step + 1) / WARMUP))
    average = copy.deepcopy(model).requires_grad_(False)  # exponential moving average of the weights [7]
    kind = torch.device(device).type  # the forward pass uses 16-bit arithmetic on a GPU
    history = []
    model.train()
    for step in range(1, CONFIG["steps"] + 1):
        # One batch of windows of 256 bytes and their next bytes, cut from disguised training text.
        windows = [disguised(text, names, words, mentions, random.random() < RENAMED)
                   for _ in range(CONFIG["batch_size"])]
        windows = torch.tensor(windows, device=device)
        x, y = windows[:, :-1], windows[:, 1:]
        with torch.autocast(kind, dtype=torch.bfloat16, enabled=kind == "cuda"):
            loss = F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten())
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        halflife = min(EMA_HALFLIFE, 0.05 * step + 1)  # shorter while training starts
        with torch.no_grad():
            for mean, current in zip(average.parameters(), model.parameters()):
                mean.lerp_(current, 1 - 0.5 ** (1 / halflife))
        if step == 1 or step % max(1, CONFIG["steps"]//40) == 0 or step == CONFIG["steps"]:
            history.append([step, round(loss.item(), 6)])
        if step == 1 or step % max(1, CONFIG["steps"]//10) == 0 or step == CONFIG["steps"]:
            print(f"{step} / {CONFIG['steps']}: loss {loss.item():.4f}", flush=True)
    model.load_state_dict(average.state_dict())
    return history

@torch.no_grad()
def generate(model, tokens, length=512, temperature=.8, generator=None):
    model.eval()
    for _ in range(length-tokens.size(1)):
        logits = model(tokens[:, -256:])[:, -1] / temperature
        next_token = torch.multinomial(logits.softmax(-1), 1, generator=generator)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens


@torch.no_grad()
def text_score(model, folder, device, split):
    model.eval()
    scores = language_metrics(model, text_data(split), device, batch=32, context=256)
    metrics = {key: scores[key] for key in ("nll_nats_per_byte", "target_bytes")}
    rng = torch.Generator(device=device).manual_seed(921)
    prefixes = [b"ROMEO:\n ", b"JULIET:\n", b"First Ci", b"KING:\n  "]*2
    for batch in range(10):
        tokens = torch.tensor([list(prefix) for prefix in prefixes], device=device)
        tokens = generate(model, tokens, length=512, temperature=.8, generator=rng)
        samples = [f"=== Sample {i+1:02d} ===\n".encode() + bytes(row.tolist())
                   for i, row in enumerate(tokens)]
        (folder/f"batch-{batch+1:02d}.txt").write_bytes(b"\n\n".join(samples) + b"\n")
    return metrics

# %% Run the experiment
def run(out=None, device=None, seed=3801, split="validation", show=True):
    source = program_source()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", STUDENT_ID):
        raise ValueError("Use letters, digits, underscore or hyphen for STUDENT_ID.")
    if CONFIG["task"] != "gpt":
        raise ValueError("Keep task='gpt' in CONFIG.")
    if not 1 <= CONFIG["steps"] <= MAX_STEPS or not 1 <= CONFIG["batch_size"] <= MAX_BATCH:
        raise ValueError("Training steps or batch size exceeds the practical limit.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    folder = Path(out or f"lab-08-{STUDENT_ID}")
    (folder/"samples").mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    size = model_size(model)
    history = train(model, device)
    metrics = text_score(model, folder/"samples", device, split)
    result = save_submission(folder, source, history, metrics, size, seed, split)
    if show:
        show_results(history, folder)
    return result


if __name__ == "__main__":
    run()
