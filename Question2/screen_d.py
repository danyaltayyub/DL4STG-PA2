"""Choose model D's architecture on the validation split (never the test split).

Each candidate has a one-step reach of 16 px (model A: 8 px) and at most 721,176 parameters.
It is trained once (seed 0) under model D's protocol and scored on the validation videos.
Rule, fixed before any candidate was trained: lowest mean validation E2 over 1x, 2x and 3x.
Writes results/metrics/d_screening.json; the winner's ArchConfig goes into src/registry.py.
"""
import json
import subprocess
import sys

PY = sys.executable
CANDIDATES = {
    "dil": {"patch": 4, "hidden": 64, "kernels": [3, 3], "dilations": [1, 3]},
    "k5":  {"patch": 4, "hidden": 46, "kernels": [5, 5], "dilations": [1, 1]},
    "p8":  {"patch": 8, "hidden": 64, "kernels": [3, 3], "dilations": [1, 1]},
}


def run(*args):
    subprocess.run([PY, *args], check=True)


results = {}
for name, arch in CANDIDATES.items():
    tag = f"screen_{name}"
    run("train.py", "--model", "D", "--seed", "0", "--tag", tag, "--arch", json.dumps(arch))
    run("evaluate.py", "--predictor", "D", "--seeds", "0", "--split", "val", "--tag", tag,
        "--parts", "basic,speed")
    m = json.load(open(f"results/{tag}/metrics/D_s0_val.json"))
    run_info = json.load(open(f"results/{tag}/runs/D_s0.json"))
    results[name] = {"arch": arch, "params": run_info["params"],
                     "train_time_per_epoch_s": run_info["train_time_per_epoch_s"],
                     **{k: m[k] for k in ("E5_1x", "E2_1x", "E2_2x", "E2_3x")},
                     "mean_E2": (m["E2_1x"] + m["E2_2x"] + m["E2_3x"]) / 3}

winner = min(results, key=lambda n: results[n]["mean_E2"])
out = {"rule": "lowest mean validation E2 over 1x, 2x, 3x (seed 0)", "candidates": results, "winner": winner}
json.dump(out, open("results/metrics/d_screening.json", "w"), indent=1)
for n, r in results.items():
    print(f"{n:4s} params {r['params']:,}  E2 1x/2x/3x {r['E2_1x']:.4f} {r['E2_2x']:.4f} {r['E2_3x']:.4f}"
          f"  mean {r['mean_E2']:.4f}  E5_1x {r['E5_1x']:.4f}")
print("winner:", winner, json.dumps(CANDIDATES[winner]))
