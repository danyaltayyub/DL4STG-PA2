"""Moving MNIST loading and the fixed video-level split."""
import torch
from torchvision.datasets import MovingMNIST

SPLITS = {"train": (0, 8000), "val": (8000, 9000), "test": (9000, 10000)}


def load_videos(root="data"):
    """All 10,000 videos as uint8 [10000, 20, 1, 64, 64] (never use split='train'/'test')."""
    return MovingMNIST(root=root, split=None, download=True).data


def get_split(name, root="data", device=None):
    """uint8 videos of one split, optionally moved to a device."""
    a, b = SPLITS[name]
    v = load_videos(root)[a:b].contiguous()
    return v.to(device) if device is not None else v
