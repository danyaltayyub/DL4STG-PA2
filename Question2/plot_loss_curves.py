"""Supplementary Figure S1: training loss per epoch for every learned model and seed."""
import json

import matplotlib.pyplot as plt

from src import plotstyle as ps

RUNS = {"A": "A", "B": "B", "C": "C", "D": "D", "D+B": "DB"}

ps.apply_style()
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, seed in zip(axes, (0, 1)):
    for label, name in RUNS.items():
        loss = json.load(open(f"results/runs/{name}_s{seed}.json"))["epoch_loss"]
        patch8 = name in ("D", "DB")
        ax.plot(range(1, len(loss) + 1), loss, lw=1.6 if patch8 else 1.0,
                color=ps.ORANGE if patch8 else ps.BLUE, ls="--" if name in ("B", "DB") else "-")
        ax.text(len(loss) + 0.4, loss[-1], label, va="center", fontsize=7, color=ps.REF)
    ax.set_title(f"seed {seed}")
    ax.set_xlabel("epoch")
    ax.set_xlim(1, 33)
axes[0].set_ylabel("training loss (masked MSE)")
fig.tight_layout()
ps.save(fig, "results/figures/figS1_loss")
