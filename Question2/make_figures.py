"""Report figures and GIFs for Question 2.

Subcommands: fig2 fig3 fig4 fig5 fig6 (all accept --smoke).
Seed-0 models are used throughout. Run from Question2/ with ../.venv/bin/python.
"""
import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch
from PIL import Image, ImageDraw

from src import plotstyle as ps
from src.data import get_split
from src.models import ArchConfig, Symmetrised
from src.registry import build_model
from src.transforms import mirror, rot
from src.windows import make_windows, window_frames

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TFUN = {"mirror": mirror, "rot": rot}
TCONS = {"mirror": "C_M", "rot": "C_R"}


class Paths:
    def __init__(self, smoke):
        base = "results/smoke" if smoke else "results"
        self.ck = "checkpoints/smoke" if smoke else "checkpoints"
        self.met = f"{base}/metrics"
        self.fig = f"{base}/figures"
        self.gif = f"{base}/gifs"


# ------------------------------------------------------------------ helpers
def has_overlap(frame, thr=0.3, min_area=15, margin=0):
    """True if a frame has overlapping digits.

    The frame (2-D float array in [0, 1]) is thresholded at intensity > thr and
    the 8-connected components are counted with cv2.connectedComponentsWithStats.
    If fewer than 2 components have area >= min_area pixels, the two digits are
    touching/merged (or one is missing), and the frame counts as overlapping.
    With margin > 0 the mask is dilated first, so digits closer than about
    2*margin pixels also count as overlapping.
    """
    img = (np.asarray(frame) > thr).astype(np.uint8)
    if margin:
        img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2))
    _, _, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    return int((stats[1:, cv2.CC_STAT_AREA] >= min_area).sum()) < 2


def stem_of(name, sym):
    return f"{name}{'_sym' if sym else ''}_s0"


def load_model(name, sym, P):
    ck = torch.load(f"{P.ck}/{name}_s0.pt", map_location=DEV)
    m = build_model(name, ArchConfig.from_dict(ck["arch"]))
    m.load_state_dict(ck["state_dict"])
    m = Symmetrised(m) if sym else m
    return m.to(DEV).eval()


@torch.no_grad()
def window(videos, i, t, k):
    return make_windows(videos[i:i + 1], t, k)


@torch.no_grad()
def predict(model, X):
    return model(X)


def np2(x):
    """[B?,5,1,64,64] tensor -> list of 5 [64,64] arrays (first batch element)."""
    return [f.numpy() for f in x.reshape(-1, 64, 64).cpu()]


def montage(rows, gap=2, sep_after=None, sep_gap=8):
    """rows: list of lists of 64x64 arrays -> (canvas, row y-centres, col x-centres, separator x)."""
    n, m = len(rows), len(rows[0])
    xs, x = [], 0
    for j in range(m):
        xs.append(x)
        x += 64 + (sep_gap if sep_after is not None and j == sep_after - 1 else gap)
    W, H = x - (sep_gap if sep_after == m else gap), n * 64 + (n - 1) * gap
    canvas = np.ones((H, W))
    ys = [r * (64 + gap) for r in range(n)]
    for r, row in enumerate(rows):
        for j, f in enumerate(row):
            canvas[ys[r]:ys[r] + 64, xs[j]:xs[j] + 64] = f
    sep = xs[sep_after - 1] + 64 + sep_gap / 2 - 0.5 if sep_after else None
    return canvas, [y + 32 for y in ys], [x + 32 for x in xs], sep


def draw(ax, rows, row_labels, col_labels, sep_after=None, fs=6):
    canvas, yc, xc, sep = montage(rows, sep_after=sep_after)
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    for y, lab in zip(yc, row_labels):
        ax.text(-5, y, lab, ha="right", va="center", fontsize=fs, color=ps.REF)
    for x, lab in zip(xc, col_labels):
        ax.text(x, -4, lab, ha="center", va="bottom", fontsize=fs, color=ps.REF)
    if sep is not None:
        ax.axvline(sep, color=ps.REF, lw=0.5)


def sci(v):
    m, e = f"{v:.2e}".split("e")
    return f"{m}e{int(e)}"


def jdump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


# ------------------------------------------------------------------ fig2
def fig2(a):
    P = Paths(a.smoke)
    ps.apply_style()
    videos = get_split("test", "data", DEV)
    fr = videos[:, 5:10, 0].float().div(255).cpu().numpy()          # target frames 6-10
    ov = np.array([any(has_overlap(f) for f in v) for v in fr])
    over = np.flatnonzero(ov)[0]
    clear = np.flatnonzero(~ov)[:2]
    chosen = [int(clear[0]), int(over), int(clear[1])]
    print("fig2 test indices:", chosen, "dataset indices:", [9000 + i for i in chosen],
          "overlapping video:", int(over))
    model = load_model(a.model, a.sym, P)
    fig, axes = plt.subplots(3, 1, figsize=(4.6, 7.2))
    for ax, i in zip(axes, chosen):
        X, Y, _ = window(videos, i, 1, 1)
        Yh = predict(model, X)
        rows = [np2(X), np2(Yh), np2(Y)]
        draw(ax, rows, ["input", "prediction", "truth"], [f"in {j}\nout {j + 5}" for j in range(1, 6)])
        ax.set_title(f"test video {i}" + (" (digits overlap)" if i == over else ""),
                     fontsize=8, color=ps.INK, pad=24)
    fig.tight_layout()
    ps.save(fig, f"{P.fig}/fig2_examples")


# ------------------------------------------------------------------ fig3 / fig6a
def select_videos(pv):
    sel = {}
    for T, key in TCONS.items():
        order = torch.argsort(pv[key], descending=True)[:2]
        sel[T] = [dict(index=int(i), dataset_index=9000 + int(i), C=float(pv[key][i])) for i in order]
    return sel


def gif_for(path, rows, labels, scale=3):
    """rows: per row (last input frame, [5 horizon frames]) -> GIF, one frame per h."""
    S, pad, lab_h, top = 64 * scale, 8, 14, 20
    W = 2 * S + 3 * pad
    H = top + len(rows) * (lab_h + S + 6)
    frames = []
    for h in range(5):
        im = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(im)
        d.text((pad, 4), f"h = {h + 1}", fill=(11, 11, 11))
        for r, ((xin, outs), lab) in enumerate(zip(rows, labels)):
            y0 = top + r * (lab_h + S + 6)
            d.text((pad, y0), lab, fill=(82, 81, 78))
            for c, f in enumerate((xin, outs[h])):
                t = Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
                im.paste(t.convert("RGB"), (pad + c * (S + pad), y0 + lab_h))
        frames.append(im)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=700, loop=0, disposal=2)


def fig3(a, prefix="fig3", name=None):
    P = Paths(a.smoke)
    ps.apply_style()
    name = name or a.model
    stem = stem_of(name, a.sym)
    pv = torch.load(f"{P.met}/{stem}_test_pervideo.pt")
    if a.videos_from:
        sel = json.load(open(a.videos_from))
    else:
        sel = select_videos(pv)
        jdump(f"{P.met}/fig3_selection_{stem}.json", sel)
    videos = get_split("test", "data", DEV)
    model = load_model(name, a.sym, P)
    panels = [(T, e["index"]) for T in ("mirror", "rot") for e in sel[T]]
    fig, axes = plt.subplots(4, 1, figsize=(6.6, 9.4))
    labels = ["(1) X | f(X)", "(2) T(X) | T(Y)", "(3) T(X) | f(T(X))", "(4) T(X) | T(f(X))"]
    for ax, (T, i) in zip(axes, panels):
        f = TFUN[T]
        with torch.no_grad():
            X, Y, _ = window(videos, i, 1, 1)
            fX, fTX = model(X), model(f(X))
            TX, TY, TfX = f(X), f(Y), f(fX)
        rows = [np2(X) + np2(fX), np2(TX) + np2(TY), np2(TX) + np2(fTX), np2(TX) + np2(TfX)]
        draw(ax, rows, labels, [str(j) for j in range(1, 11)], sep_after=5)
        c = float(pv[TCONS[T]][i])
        ax.set_title(f"test video {i}, T = {T}, {TCONS[T]} = {sci(c)}", fontsize=8, color=ps.INK, pad=14)
        # GIF: last input frame next to horizon-h frame
        g = [(np2(X)[4], np2(fX)), (np2(TX)[4], np2(TY)), (np2(TX)[4], np2(fTX)), (np2(TX)[4], np2(TfX))]
        gif_for(f"{P.gif}/{prefix}_{T}_{i}.gif", g, labels)
    fig.tight_layout()
    ps.save(fig, f"{P.fig}/{prefix}_consistency")
    print(prefix, "panels (T, test index):", panels)


# ------------------------------------------------------------------ fig4
def fig4(a):
    P = Paths(a.smoke)
    ps.apply_style()
    stem = stem_of(a.model, a.sym)
    pv = torch.load(f"{P.met}/{stem}_test_pervideo.pt")
    sel = json.load(open(f"{P.met}/fig3_selection_{stem}.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, (T, key, tex) in zip(axes, [("mirror", "C_M", "$C_M$"), ("rot", "C_R", "$C_R$")]):
        v = pv[key].numpy()
        pos = v[v > 0]
        bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 31)
        ax.hist(pos, bins=bins, color=ps.BLUE, edgecolor="white", linewidth=0.8)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ticks = np.geomspace(pos.min(), pos.max(), 4)
        ax.set_xticks(ticks)
        ax.set_xticklabels([sci(t) for t in ticks])
        ax.set_xlabel(f"{tex} per video")
        ax.set_ylabel("number of videos")
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.18)
        for e in sel[T]:
            ax.axvline(e["C"], color=ps.REF, lw=1.2, ls="--")
        # one horizontal label for both marked videos (their lines sit close together)
        cmin = min(e["C"] for e in sel[T])
        ax.text(cmin / 1.08, ymax * 0.6, "Figure 3 videos\n" + ", ".join(str(e["index"]) for e in sel[T]),
                ha="right", va="center", fontsize=7, color=ps.REF)
        med = float(np.median(v))
        ax.axvline(med, color=ps.REF, lw=1.2, ls=(0, (1, 2)))
        ax.text(med, ymax * 1.16, "median", rotation=90, ha="right", va="top", fontsize=7, color=ps.REF)
        ax.set_title(f"{tex}, {len(v)} test videos", fontsize=9)
    fig.tight_layout()
    ps.save(fig, f"{P.fig}/fig4_hist")


# ------------------------------------------------------------------ fig5 / fig6b
SHOWN = {1: (6, 7), 2: (11, 13), 3: (16, 19)}


def choose_fig5_video(videos):
    fr = videos[:, :, 0].float().div(255).cpu().numpy()
    for i in range(len(fr)):
        # faint stroke edges included (thr 0.1) and a 2-px margin, so the digits are clearly apart
        if not any(has_overlap(fr[i, f - 1], thr=0.1, margin=2) for k in SHOWN for f in SHOWN[k]):
            return i
    raise RuntimeError("no qualifying video")


def fig5(a, prefix="fig5", name=None, write_sel=True):
    P = Paths(a.smoke)
    ps.apply_style()
    name = name or a.model
    videos = get_split("test", "data", DEV)
    if a.video is not None:
        i = a.video
    else:
        i = choose_fig5_video(videos)
        if write_sel:
            jdump(f"{P.met}/fig5_selection.json", dict(index=i, dataset_index=9000 + i))
    print(prefix, "test video:", i)
    model = load_model(name, a.sym, P)
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 3.9))
    for ax, k in zip(axes, (1, 2, 3)):
        X, Y, _ = window(videos, i, 1, k)
        Yh = predict(model, X)
        inp, tgt = window_frames(1, k)
        row, cols = np2(X), [str(f) for f in inp]
        for h in range(2):
            row += [np2(Yh)[h], np2(Y)[h]]
            cols += [f"pred {tgt[h]}", f"true {tgt[h]}"]
        draw(ax, [row], [f"{k}$\\times$"], cols, sep_after=5)
        ax.set_title(f"speed {k}$\\times$", fontsize=8, color=ps.INK, pad=14)
    fig.tight_layout()
    ps.save(fig, f"{P.fig}/{prefix}_speeds")


# ------------------------------------------------------------------ fig6
def fig6(a):
    P = Paths(a.smoke)
    tag = f"fig6{a.tag}"
    a.videos_from = f"{P.met}/fig3_selection_{a.ref}_s0.json"
    a.video = json.load(open(f"{P.met}/fig5_selection.json"))["index"]
    fig3(a, prefix=tag)
    fig5(a, prefix=tag, write_sel=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("fig2", "fig3", "fig4", "fig5", "fig6"):
        p = sub.add_parser(c)
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--model", default="A")
        p.add_argument("--sym", action="store_true")
        p.add_argument("--videos-from", dest="videos_from", default=None)
        p.add_argument("--video", type=int, default=None)
        p.add_argument("--tag", default="a", help="fig6 only: a, b, ...")
        p.add_argument("--ref", default="A", help="fig6 only: model whose selections are reused")
    a = ap.parse_args()
    globals()[a.cmd](a)


if __name__ == "__main__":
    main()
