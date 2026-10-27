"""Table 1 and extra sanity checks (no dataset needed). Exits non-zero on failure."""
import json
import os
import sys
import torch
from src.models import ConvLSTMForecaster, Symmetrised, count_params
from src.registry import build_model
from src.transforms import apply, inverse, mirror, rot
from src.windows import make_windows, window_frames

torch.manual_seed(0)
x = torch.rand(20, 1, 64, 64)
p = torch.zeros(20, 1, 64, 64)
p[7, 0, 10, 20] = 1


def bright(v):
    return tuple(int(i) for i in (v == 1).nonzero()[0][[0, 2, 3]])


def frames(k):
    v = torch.arange(1, 21).float().view(1, 20, 1, 1, 1).expand(1, 20, 1, 64, 64).contiguous()
    X, Y, m = make_windows(v, 1, k)
    return ([int(a) for a in X[0, :, 0, 0, 0]], [int(a) for a in Y[0, m[0], 0, 0, 0]])


rows, ok = [], True


def row(name, check, value, expected, passed):
    global ok
    ok &= bool(passed)
    rows.append({"row": name, "check": check, "value": value, "expected": expected, "pass": bool(passed)})


d = (mirror(mirror(x)) - x).abs().max().item()
row("a", "max|M(M(x))-x|", d, "0, equal", d == 0 and torch.equal(mirror(mirror(x)), x))
d = (rot(x, 4) - x).abs().max().item()
row("b", "max|R^4(x)-x|", d, "0, equal", d == 0 and torch.equal(rot(x, 4), x))
lhs = mirror(rot(mirror(x)))
d = (lhs - rot(x, -1)).abs().max().item()
row("c", "max|M R M(x)-R^-1(x)|", d, "0, equal", d == 0 and torch.equal(lhs, rot(x, -1)))
row("d", "bright pixel of M(p)", bright(mirror(p)), (7, 10, 43), bright(mirror(p)) == (7, 10, 43))
row("e", "bright pixel of R(p)", bright(rot(p)), (7, 43, 10), bright(rot(p)) == (7, 43, 10))
for r, k, exp in (("f", 1, ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])),
                  ("g", 2, ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19])),
                  ("h", 3, ([1, 4, 7, 10, 13], [16, 19]))):
    got = frames(k)
    row(r, f"{k}x input / target frames", got, exp, got == exp and tuple(got) == tuple(map(list, window_frames(1, k))))

print(f"{'row':<4}{'check':<28}{'value':<48}{'expected':<40}pass")
for r in rows:
    print(f"({r['row']}) {r['check']:<28}{str(r['value']):<48}{str(r['expected']):<40}{r['pass']}")

extra = {}
inv_ok = all(torch.equal(inverse(g, apply(g, x)), x) for g in range(8))
distinct = all(not torch.equal(apply(i, x), apply(j, x)) for i in range(8) for j in range(i))
extra["inverse_all_g"], extra["8_distinct"] = inv_ok, distinct
n = count_params(build_model("A"))
extra["params_A"] = n
extra["params_A_ok"] = n == 480_784
dev = "cuda" if torch.cuda.is_available() else "cpu"
X = torch.rand(8, 5, 1, 64, 64, device=dev)
for nm in ("blank", "copylast"):
    f = build_model(nm).to(dev)
    extra[f"{nm}_C_M"] = ((f(mirror(X)) - mirror(f(X))) ** 2).max().item()
    extra[f"{nm}_C_R"] = ((f(rot(X)) - rot(f(X))) ** 2).max().item()
torch.manual_seed(0)
fs = Symmetrised(build_model("A").to(dev).eval())
with torch.no_grad():
    cm = ((fs(mirror(X)) - mirror(fs(X))) ** 2).mean(dim=(1, 2, 3, 4)).max().item()
    cr = ((fs(rot(X)) - rot(fs(X))) ** 2).mean(dim=(1, 2, 3, 4)).max().item()
extra["sym_A_C_M"], extra["sym_A_C_R"] = cm, cr
ok &= inv_ok and distinct and extra["params_A_ok"]
ok &= all(extra[f"{nm}_C_{t}"] == 0 for nm in ("blank", "copylast") for t in "MR")
ok &= cm <= 1e-10 and cr <= 1e-10
print("\nextra checks:")
for k, v in extra.items():
    print(f"  {k}: {v}")
os.makedirs("results/metrics", exist_ok=True)
json.dump({"table1": rows, "extra": extra, "all_pass": bool(ok)}, open("results/metrics/table1.json", "w"), indent=1)
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
