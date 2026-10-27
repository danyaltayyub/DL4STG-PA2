"""Per-video MSE, consistency and inference-time measurements."""
import time
import torch
from .transforms import mirror, rot
from .windows import make_windows

NAN = float("nan")


def batches(videos, bs=32):
    for i in range(0, len(videos), bs):
        yield videos[i:i + bs]


@torch.no_grad()
def mse_per_video(f, videos, k=1, t=1, bs=32):
    """MSE_h per video, [N,5] (CPU); NaN where the target frame does not exist."""
    out = []
    for vb in batches(videos, bs):
        X, Y, mask = make_windows(vb, t, k)
        mse = ((f(X) - Y) ** 2).mean(dim=(2, 3, 4))
        out.append(mse.masked_fill(~mask, NAN).cpu())
    return torch.cat(out)


def e_over(mse, h):
    """Mean over the first h horizons, then over videos."""
    return mse[:, :h].mean(1).mean().item()


@torch.no_grad()
def consistency(f, videos, bs=32):
    """C_T and E_T per video for T in {M, R}, 1x, t=1, all five horizons."""
    res = {n: [] for n in ("C_M", "E_M", "C_R", "E_R")}
    for vb in batches(videos, bs):
        X, Y, _ = make_windows(vb, 1, 1)
        fX = f(X)
        for n, T in (("M", mirror), ("R", rot)):
            fTX = f(T(X))
            res[f"C_{n}"].append(((fTX - T(fX)) ** 2).mean(dim=(1, 2, 3, 4)).cpu())
            res[f"E_{n}"].append(((fTX - T(Y)) ** 2).mean(dim=(1, 2, 3, 4)).cpu())
    return {n: torch.cat(v) for n, v in res.items()}


@torch.no_grad()
def inference_time(f, videos, bs=32):
    """Seconds to predict all windows (1x, t=1); batches are prepared on device beforehand."""
    xs = [make_windows(vb, 1, 1)[0] for vb in batches(videos, bs)]
    dev = videos.device
    if dev.type == "cuda":
        torch.cuda.synchronize()
    f(xs[0])  # untimed warm-up
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for X in xs:
        f(X)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0
