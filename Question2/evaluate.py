"""CLI: evaluate.py --predictor {A,B,C,D,DB,blank,copylast} [--seeds 0 1] [--symmetrised] ..."""
import argparse
import json
import os
import torch
from src import metrics as M
from src.data import get_split
from src.models import ArchConfig, Symmetrised, count_params
from src.registry import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--predictor", required=True, choices=["A", "B", "C", "D", "DB", "blank", "copylast"])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
ap.add_argument("--symmetrised", action="store_true")
ap.add_argument("--split", default="test", choices=["test", "val"])
ap.add_argument("--tag", default=None)
ap.add_argument("--parts", default="basic,consistency,speed,timing")
ap.add_argument("--unlock-baseline", action="store_true",
                help="allow any part on a non-smoke model A checkpoint (main thread only)")
a = ap.parse_args()
parts = a.parts.split(",")
assert set(parts) <= {"basic", "consistency", "speed", "timing"}
sub = f"{a.tag}/" if a.tag else ""   # e.g. smoke/, screen_k5/
learned = a.predictor not in ("blank", "copylast")
if a.predictor == "A" and not a.tag and not a.unlock_baseline:
    raise SystemExit("pre-registration: refusing non-smoke model A without --unlock-baseline")
dev = "cuda" if torch.cuda.is_available() else "cpu"
videos = get_split(a.split, "data", dev)


def load(seed):
    f = build_model(a.predictor)
    params = 0
    if learned:
        ck = torch.load(f"checkpoints/{sub}{a.predictor}_s{seed}.pt", map_location=dev)
        f = build_model(a.predictor, ArchConfig.from_dict(ck["arch"]))
        f.load_state_dict(ck["state_dict"])
        params = count_params(f)
    f = f.to(dev).eval()
    return (Symmetrised(f) if a.symmetrised else f), params


def merge_json(path, new):
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(new)
    json.dump(old, open(path, "w"), indent=1)


def merge_pt(path, new):
    old = torch.load(path) if os.path.exists(path) else {}
    old.update(new)
    torch.save(old, path)


for seed in (a.seeds if learned else [None]):
    f, params = load(seed)
    stem = a.predictor + ("_sym" if a.symmetrised else "") + (f"_s{seed}" if learned else "")
    base = f"results/{sub}metrics/{stem}_{a.split}"
    os.makedirs(os.path.dirname(base), exist_ok=True)
    js, pv = {}, {}
    if "basic" in parts:
        mse = M.mse_per_video(f, videos, 1)
        pv["mse_1x"] = mse
        js.update({f"MSE_{h + 1}": mse[:, h].mean().item() for h in range(5)})
        js.update(E5=M.e_over(mse, 5), E2=M.e_over(mse, 2), params=params)
    if "consistency" in parts:
        c = M.consistency(f, videos)
        pv.update(c)
        js.update({n: v.mean().item() for n, v in c.items()})
    if "speed" in parts:
        for k in (1, 2, 3):
            mse = M.mse_per_video(f, videos, k)
            pv[f"mse_{k}x"] = mse
            js[f"E2_{k}x"] = M.e_over(mse, 2)
            if k < 3:
                js[f"E5_{k}x"] = M.e_over(mse, 5)
    if "timing" in parts:
        js["inference_time_s"] = M.inference_time(f, videos)
    merge_json(base + ".json", js)
    merge_pt(base + "_pervideo.pt", pv)
    print(stem, a.split, {k: float(f"{v:.4g}") for k, v in js.items()})
