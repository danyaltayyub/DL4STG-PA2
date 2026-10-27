"""Shared matplotlib style for report figures."""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
REF = "#52514e"    # reference lines and secondary text
INK = "#0b0b0b"    # titles
GRID = "#e1e0d9"


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 9,
        "axes.titlecolor": INK,
        "axes.labelcolor": REF,
        "xtick.color": REF,
        "ytick.color": REF,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.6,
        "axes.axisbelow": True,
    })


def save(fig, path_without_ext):
    d = os.path.dirname(path_without_ext)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path_without_ext + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(path_without_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)
