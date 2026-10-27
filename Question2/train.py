"""CLI: train.py --model A --seed 0 [--epochs 30] [--tag NAME] [--arch JSON]"""
import argparse
import json
from src.models import ArchConfig
from src.train import train

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["A", "B", "C", "D", "DB"])
ap.add_argument("--seed", type=int, required=True, choices=[0, 1])
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--tag", default=None)
ap.add_argument("--arch", default=None, help="ArchConfig as JSON")
a = ap.parse_args()
arch = ArchConfig.from_dict(json.loads(a.arch)) if a.arch else None
info = train(a.model, a.seed, a.epochs, a.tag, arch)
print(f"done: params {info['params']}, {info['train_time_per_epoch_s']:.1f} s/epoch")
