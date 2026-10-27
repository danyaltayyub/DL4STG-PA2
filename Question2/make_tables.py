"""Write the LaTeX tables of Question 2 into ../report/generated/.

Missing input files never crash the script: the affected table is skipped and
the missing files are printed.
Usage: make_tables.py [--final STEM]   (STEM e.g. DB_sym; default none)
"""
import argparse
import json
import math
import os

OUT = "../report/generated"
MET = "results/metrics"
RUNS = "results/runs"
SEEDS = (0, 1)


class Missing(Exception):
    pass


def load(path):
    if not os.path.exists(path):
        raise Missing(path)
    with open(path) as f:
        return json.load(f)


def metric(stem, key, split="test"):
    d = load(f"{MET}/{stem}_{split}.json")
    if key not in d:
        raise Missing(f"{MET}/{stem}_{split}.json[{key}]")
    return d[key]


# ---------------------------------------------------------------- formatting
def fmt(v, sci=False):
    """3 s.f.; |v|<1e-3 (or sci=True) in scientific notation; exact 0 as '0'."""
    if v == 0:
        return "0"
    if sci or abs(v) < 1e-3:
        m, e = f"{v:.2e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    if abs(v) >= 1000:
        return f"{round(v):,}".replace(",", "{,}")
    s = f"{v:#.3g}"
    return s.rstrip(".")


def fmt_int(v):
    return f"{int(v):,}".replace(",", "{,}")


def learned(stems, getter, f=fmt):
    """mean (seed 0, seed 1) for a learned model; single value if identical."""
    vals = [getter(s) for s in stems]
    if f is fmt_int:
        return fmt_int(vals[0]) if len(set(vals)) == 1 else f"{fmt_int(sum(vals)/len(vals))} ({', '.join(map(fmt_int, vals))})"
    return f"{f(sum(vals) / len(vals))} ({', '.join(f(v) for v in vals)})"


def seeds_of(base):
    return [f"{base}_s{s}" for s in SEEDS]


def mget(key):
    return lambda stem: metric(stem, key)


def run_get(key, base):
    return lambda stem: load(f"{RUNS}/{base}_s{stem.rsplit('_s', 1)[1]}.json")[key]


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.tex", "w") as f:
        f.write(text)
    print("wrote", f"{OUT}/{name}.tex")


def table(n, caption, colspec, header, rows, midrules=(), resize=False):
    body = [header + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        body.append(" & ".join(r) + r" \\")
        if i in midrules:
            body.append(r"\midrule")
    tab = "\\begin{tabular}{" + colspec + "}\n\\toprule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}"
    if resize:
        tab = "\\resizebox{\\textwidth}{!}{%\n" + tab + "}"
    return ("\\begin{table}[htbp]\n\\centering\n\\small\n"
            f"\\caption{{{caption}}}\n\\label{{tab:{n}}}\n{tab}\n\\end{{table}}\n")


# ---------------------------------------------------------------- tables
def flowstats():
    d = load(f"{MET}/flow_stats.json")
    write("flowstats", "\\newcommand{\\FLOWNKEPT}{%s}\n\\newcommand{\\FLOWMEDIAN}{%s}\n"
          % (fmt_int(d["n_pixels_kept"]), f"{d['median_speed']:.2f}"))


def table1():
    d = {r["row"]: r for r in load(f"{MET}/table1.json")["table1"]}
    chk = {
        "a": r"$\max|M(M(x))-x|$", "b": r"$\max|R^4(x)-x|$", "c": r"$\max|MRM(x)-R^{-1}(x)|$",
        "d": r"bright pixel of $M(p)$ (frame, row, col.)", "e": r"bright pixel of $R(p)$ (frame, row, col.)",
        "f": r"$1\times$ input / target frames", "g": r"$2\times$ input / target frames",
        "h": r"$3\times$ input / target frames",
    }
    lst = lambda l: ",".join(map(str, l))
    rng = lambda l: f"{l[0]}--{l[-1]}"

    def cell(r, key):
        v, row = r[key], r["row"]
        if row in "abc":
            return "0, \\texttt{torch.equal}: True" if key == "expected" or r["pass"] else "FAIL"
        if row in "de":
            return "(" + ", ".join(map(str, v)) + ")"
        if row == "f":
            return f"{rng(v[0])} / {rng(v[1])}"
        return f"{lst(v[0])} / {lst(v[1])}"

    rows = []
    for k in "abcdefgh":
        r = d[k]
        yours = cell(r, "value") if k not in "abc" else (
            "0, \\texttt{torch.equal}: True" if r["pass"] and r["value"] == 0 else f"{r['value']}")
        rows.append([f"({k})", chk[k], cell(r, "expected"), yours])
    write("table1", table(1, "Sanity checks of the transformations and window construction "
                          "(the ``Yours'' column is computed by \\texttt{checks.py}).",
                          "cp{5.5cm}ll", "Row & Check & Expected & Yours", rows, midrules=(4,)))


MSEK = [f"MSE_{h}" for h in range(1, 6)]
HDR_MSE = " & ".join(rf"$\mathrm{{MSE}}_{h}$" for h in range(1, 6))


def table2():
    A = seeds_of("A")
    row = lambda lab, f: [lab] + [f(k) for k in MSEK + ["E5"]]
    rows = [
        ["Model A"] + [learned(A, mget(k)) for k in MSEK + ["E5"]] + [learned(A, mget("params"), fmt_int)],
        ["Blank"] + [fmt(metric("blank", k)) for k in MSEK + ["E5"]] + ["0"],
        ["Copy-last"] + [fmt(metric("copylast", k)) for k in MSEK + ["E5"]] + ["0"],
    ]
    write("table2", table(2, "Test MSE per horizon, $E^{(5)}$ and trainable parameters at $1\\times$, $t=1$; "
                          "lower is better, entries are mean (seed 0, seed 1).",
                          "lccccccc", f"Predictor & {HDR_MSE} & $E^{{(5)}}$ & Parameters", rows, resize=True))


def table3():
    A = seeds_of("A")
    keys = ["E5", "C_M", "E_M", "C_R", "E_R"]
    rows = [["Model A"] + [learned(A, mget(k)) for k in keys],
            ["Blank"] + [fmt(metric("blank", k)) for k in keys],
            ["Copy-last"] + [fmt(metric("copylast", k)) for k in keys]]
    write("table3", table(3, "Test error and consistency under mirror ($M$) and rotation ($R$) at $1\\times$; "
                          "lower is better, entries are mean (seed 0, seed 1).",
                          "lccccc", "Predictor & $E^{(5)}$ & $C_M$ & $E_M$ & $C_R$ & $E_R$", rows, resize=True))


def table4():
    A = seeds_of("A")
    keys = ["E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x"]
    rows = [["Model A"] + [learned(A, mget(k)) for k in keys],
            ["Copy-last"] + [fmt(metric("copylast", k)) for k in keys]]
    write("table4", table(4, "Test error of model A at speeds $1\\times,2\\times,3\\times$ (start $t=1$); "
                          "lower is better, entries are mean (seed 0, seed 1).",
                          "lccccc",
                          r"Predictor & $E^{(2)}$ $1\times$ & $E^{(2)}$ $2\times$ & $E^{(2)}$ $3\times$ & $E^{(5)}$ $1\times$ & $E^{(5)}$ $2\times$",
                          rows, resize=True))


def comparison_rows(cols, sci_c):
    """cols: list of (stem_base, train_base) -> rows of (label, [cells])."""
    spec = [
        (r"$E^{(5)}$ $1\times$", "E5_1x", False), (r"$C_M$", "C_M", sci_c), (r"$E_M$", "E_M", False),
        (r"$C_R$", "C_R", sci_c), (r"$E_R$", "E_R", False),
        (r"$E^{(2)}$ $1\times$", "E2_1x", False), (r"$E^{(2)}$ $2\times$", "E2_2x", False),
        (r"$E^{(2)}$ $3\times$", "E2_3x", False),
        ("Trainable parameters", "params", None), ("Training time per epoch (s)", "train", False),
        ("Inference time (s)", "inference_time_s", False),
    ]
    rows = []
    for lab, key, sci in spec:
        r = [lab]
        for stem, tb in cols:
            S = seeds_of(stem)
            if key == "params":
                r.append(learned(S, mget("params"), fmt_int))
            elif key == "train":
                r.append(learned(S, run_get("train_time_per_epoch_s", tb)))
            else:
                r.append(learned(S, mget(key), lambda v, s=sci: fmt(v, sci=bool(s))))
        rows.append(r)
    return rows


def table5():
    names = ["A", "B", "C", "D"]
    rows = comparison_rows([(n, n) for n in names], False)
    write("table5", table(5, "Test results for models A--D (rows: quantities; columns: models); lower is better "
                          "for all rows, entries are mean (seed 0, seed 1). $E^{(2)}$ is used across speeds, with $t=1$.",
                          "l" + "c" * 4, "Quantity & A & B & C & D", rows, midrules=(0, 4, 7, 8), resize=True))


def table6(final):
    cols, heads = [("B_sym", "B"), ("D_sym", "D")], ["Sym.\\ B", "Sym.\\ D"]
    if final:
        base = final[:-4] if final.endswith("_sym") else final
        cols.append((final, base))
        heads.append(f"Final ({final.replace('_', ' ')})")
    rows = comparison_rows(cols, True)
    write("table6", table(6, "Test results for symmetrised models (rows: quantities; columns: models); lower is "
                          "better, entries are mean (seed 0, seed 1). Training time is that of the underlying learned model.",
                          "l" + "c" * len(cols), "Quantity & " + " & ".join(heads), rows, midrules=(0, 4, 7, 8), resize=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=None, help="stem of the final model, e.g. DB_sym")
    a = ap.parse_args()
    jobs = [("flowstats", flowstats), ("table1", table1), ("table2", table2), ("table3", table3),
            ("table4", table4), ("table5", table5), ("table6", lambda: table6(a.final))]
    for name, fn in jobs:
        try:
            fn()
        except Missing as e:
            print(f"SKIPPED {name}: missing {e}")


if __name__ == "__main__":
    main()
