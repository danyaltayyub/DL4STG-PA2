"""Training of one model / seed (spec section 5)."""
import json
import os
import platform
import time
import torch
from .data import get_split
from .models import ArchConfig, count_params
from .registry import REGISTRY, build_model
from .transforms import apply
from .windows import make_windows


def hardware():
    cpu = platform.processor()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {"gpu": gpu, "cpu": cpu}


def masked_loss(Yh, Y, mask):
    """Squared error over pixels, averaged over the available target frames per video, then videos."""
    se = ((Yh - Y) ** 2).mean(dim=(2, 3, 4))
    return ((se * mask).sum(1) / mask.sum(1)).mean()


def sample_epoch(n, speed_aug, sym_aug, gen):
    """Shuffled order plus one (k, t, g) per video, all from the one CPU generator."""
    perm = torch.randperm(n, generator=gen)
    k = torch.randint(1, 4, (n,), generator=gen) if speed_aug else torch.ones(n, dtype=torch.long)
    tmax = 20 - 6 * k if speed_aug else torch.full((n,), 11)
    t = (torch.rand(n, generator=gen) * tmax).long() + 1
    g = torch.randint(0, 8, (n,), generator=gen) if sym_aug else None
    return perm, k, t, g


def augment(X, Y, g):
    """Apply a per-window group element, batching windows by g."""
    X, Y = X.clone(), Y.clone()
    for gi in range(8):
        sel = (g == gi).nonzero().squeeze(1)
        if len(sel):
            X[sel], Y[sel] = apply(gi, X[sel]), apply(gi, Y[sel])
    return X, Y


def train(name, seed, epochs=30, tag=None, arch=None, root="data", bs=32):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = REGISTRY[name]
    videos = get_split("train", root, dev)
    torch.manual_seed(seed)
    model = build_model(name, arch).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    n = len(videos)
    losses, times = [], []
    for ep in range(epochs):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        perm, k, t, g = sample_epoch(n, spec["speed_aug"], spec["sym_aug"], gen)
        perm, k, t = perm.to(dev), k.to(dev), t.to(dev)
        g = g.to(dev) if g is not None else None
        tot = torch.zeros((), device=dev)
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            X, Y, mask = make_windows(videos[perm[sl]], t[sl], k[sl])
            if g is not None:
                X, Y = augment(X, Y, g[sl])
            loss = masked_loss(model(X), Y, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach() * len(X)
        if dev == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(tot.item() / n)
        print(f"epoch {ep + 1}/{epochs} loss {losses[-1]:.6f} time {times[-1]:.1f}s", flush=True)
    sub = f"{tag}/" if tag else ""   # e.g. smoke/, screen_k5/
    os.makedirs(f"checkpoints/{sub}", exist_ok=True)
    os.makedirs(f"results/{sub}runs", exist_ok=True)
    arch_d = model.arch.to_dict()
    torch.save({"state_dict": model.state_dict(), "arch": arch_d, "name": name, "seed": seed},
               f"checkpoints/{sub}{name}_s{seed}.pt")
    info = {"epoch_loss": losses, "epoch_time_s": times,
            "train_time_per_epoch_s": sum(times) / len(times), "params": count_params(model),
            "hardware": hardware(), "torch_version": torch.__version__, "arch": arch_d}
    with open(f"results/{sub}runs/{name}_s{seed}.json", "w") as fh:
        json.dump(info, fh, indent=1)
    return info
