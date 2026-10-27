"""Figure 1: pooled Farneback optical-flow direction and speed statistics on MovingMNIST."""
import json
import math
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from plotstyle import BLUE, REF, apply_style, save  # noqa: E402

N_VIDEOS = 50
N_FRAMES = 20
FIG_BASE = os.path.join(HERE, "results", "figures", "fig1_flow")
JSON_PATH = os.path.join(HERE, "results", "metrics", "flow_stats.json")

# ---------------------------------------------------------------- data
ds = torchvision.datasets.MovingMNIST(root=os.path.join(HERE, "data"), split=None, download=False)
frames = ds.data[:N_VIDEOS].numpy()[:, :, 0]          # [50, 20, 64, 64] uint8
assert frames.dtype == np.uint8 and frames.shape == (N_VIDEOS, N_FRAMES, 64, 64), frames.shape

# ---------------------------------------------------------------- flow
dx_all, dy_all = [], []
n_pairs = 0
for v in range(N_VIDEOS):
    for s in range(N_FRAMES - 1):
        prev = np.ascontiguousarray(frames[v, s])
        nxt = np.ascontiguousarray(frames[v, s + 1])
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx = flow[..., 0]
        dy = flow[..., 1]
        mag = np.sqrt(dx ** 2 + dy ** 2)
        keep = (prev > 51) & (nxt > 51) & (mag > 0.5)
        dx_all.append(dx[keep])
        dy_all.append(dy[keep])
        n_pairs += 1

assert n_pairs == 950, n_pairs
dx = np.concatenate(dx_all).astype(np.float64)
dy = np.concatenate(dy_all).astype(np.float64)
speed = np.sqrt(dx ** 2 + dy ** 2)
n_kept = speed.size

theta = np.degrees(np.arctan2(-dy, dx))
theta = np.mod(theta, 360.0)
theta[theta >= 360.0] -= 360.0                         # guard against float round-up
theta[theta < 0.0] += 360.0

# ---------------------------------------------------------------- direction
dir_edges = np.arange(0, 361, 20.0)
dir_counts, _ = np.histogram(theta, bins=dir_edges)
dir_frac = dir_counts / n_kept
assert abs(dir_frac.sum() - 1.0) <= 1e-9, dir_frac.sum()
assert dir_frac.size == 18

# ---------------------------------------------------------------- speed statistics (all kept pixels)
median_speed = float(np.median(speed))
mean_speed = float(np.mean(speed))
p10, p25, p75, p90, p995 = (float(np.percentile(speed, q)) for q in (10, 25, 75, 90, 99.5))
upper = math.ceil(p995 / 0.25) * 0.25
speed_edges = np.arange(0.5, upper + 1e-9, 0.25)
speed_counts, _ = np.histogram(speed, bins=speed_edges)
speed_frac = speed_counts / n_kept                     # denominator = all kept pixels
plotted_speed_sum = float(speed_frac.sum())
excluded_frac = float(np.mean(speed > speed_edges[-1]))
assert abs(plotted_speed_sum + excluded_frac - 1.0) <= 1e-9, (plotted_speed_sum, excluded_frac)
print(f"speed plotted sum={plotted_speed_sum:.6f}, excluded above {speed_edges[-1]:.2f}px={excluded_frac:.6f}")

# ---------------------------------------------------------------- figure
apply_style()
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.6))

# (a) direction
centers = dir_edges[:-1] + 10.0
ax_a.bar(centers, dir_frac, width=20.0, color=BLUE, edgecolor="white", linewidth=0.8)
ax_a.axhline(1 / 18, ls="--", color=REF, lw=0.9)
ax_a.text(122, 1 / 18 + 0.001, "uniform\n(1/18)", color=REF, ha="left", va="bottom", fontsize=8)
ax_a.set_xlim(0, 360)
ax_a.set_xticks(np.arange(0, 361, 60))
ax_a.set_xlabel("flow direction θ (degrees)")
ax_a.set_ylabel("fraction of kept pixels")
ax_a.set_title("(a) direction", loc="left")

# (b) speed
ax_b.bar(speed_edges[:-1], speed_frac, width=0.25, align="edge", color=BLUE,
         edgecolor="white", linewidth=0.8)
ax_b.axvline(median_speed, ls="--", color=REF, lw=0.9)
ax_b.text(median_speed + 0.05, 0.97, f"median = {median_speed:.2f} px/frame",
          color=REF, ha="left", va="top", fontsize=8, transform=ax_b.get_xaxis_transform())
ax_b.set_xlim(0.5, upper)
ax_b.set_xlabel("flow speed (pixels per frame)")
ax_b.set_ylabel("fraction of kept pixels")
ax_b.set_title("(b) speed", loc="left")

fig.tight_layout()
save(fig, FIG_BASE)

# ---------------------------------------------------------------- metrics
summary = {
    "n_videos": N_VIDEOS,
    "n_pairs": n_pairs,
    "n_pixels_kept": int(n_kept),
    "median_speed": median_speed,
    "mean_speed": mean_speed,
    "speed_p10": p10,
    "speed_p25": p25,
    "speed_p75": p75,
    "speed_p90": p90,
    "speed_2x_median": 2 * median_speed,
    "speed_3x_median": 3 * median_speed,
    "direction_fractions": [float(x) for x in dir_frac],
    "direction_min_fraction": float(dir_frac.min()),
    "direction_max_fraction": float(dir_frac.max()),
}
os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
with open(JSON_PATH, "w") as f:
    json.dump(summary, f, indent=2)

short = {k: summary[k] for k in ("n_pairs", "n_pixels_kept", "median_speed", "mean_speed",
                                  "direction_min_fraction", "direction_max_fraction")}
print(json.dumps(short, indent=2))
print("wrote", FIG_BASE + ".png", "and", JSON_PATH)
