# DL4STG-PA2

AI651 — Deep Learning for Space, Time and Graphs, LUMS, Fall 2026. Assignment 2.

| Path | Contents |
| --- | --- |
| `DL4STG_PA2.pdf` | The handout |
| `Question1/` | The supplied companion notebook (optional, ungraded; not used) |
| `Question2/` | ConvLSTM on Moving MNIST: code, checkpoints, results, figures and GIFs |
| `Question2/Question2.ipynb` | Executed notebook: Table 1 checks, a training run of model A, re-evaluation of every checkpoint against the saved metrics, tables and figures |
| `report/main.pdf` | The submitted report |
| `report/` | LaTeX source of the report (`main.tex`), Question 1 diagrams, generated tables, pre-run predictions, AI-usage log |

## Question 1 in brief

A design proposal (no code): a physics-shaped spatio-temporal graph network that forecasts the one-hour level change at S1–S4 as quantiles. It routes upstream information along the river graph with stage-dependent travel times and treats announced reservoir releases as known future inputs. The diagram source is `report/figures/q1_architecture.dot` (render with Graphviz `dot`).

## Question 2 results at a glance

Test split (videos 9,000–9,999), entries are means over seeds 0 and 1. Lower is better.

| Model | What changes | E⁽⁵⁾ 1× | C_M | C_R | E⁽²⁾ 2× | E⁽²⁾ 3× |
| --- | --- | --- | --- | --- | --- | --- |
| A | ConvLSTM baseline, 4×4 folding (reach 8 px) | 0.0220 | 1.86e-3 | 3.12e-3 | 0.0461 | 0.0540 |
| B | A + symmetry augmentation | 0.0223 | 1.16e-3 | 1.43e-3 | 0.0439 | 0.0491 |
| C | A + speed augmentation | 0.0278 | 1.02e-3 | 1.97e-3 | 0.0294 | 0.0337 |
| D | 8×8 folding (reach 16 px) + speed augmentation | 0.0296 | 1.22e-3 | 1.79e-3 | 0.0295 | 0.0319 |
| **Symmetrised D (final)** | D averaged over the 8 symmetries | 0.0294 | ~1e-17 | ~1e-17 | 0.0292 | 0.0317 |

- Mirror and rotation failures come from **appearance**: B removes the rotation error gap at no 1× cost.
- The 2× and most of the 3× failure come from **unfamiliar motion**: C fixes them with model A's reach. D's larger reach adds a little at 3× and makes the error almost speed-independent.
- The symmetrised forecaster is **exactly** consistent under mirror and rotation (gaps ~1e-17, float rounding).
- Model D's architecture was chosen on the **validation** split only (`screen_d.py`, `results/metrics/d_screening.json`).
- D+B was considered as the final model but dropped: its seed-0 run never left an early loss plateau within the fixed 30 epochs (Figure S1).

Pre-registered predictions (`report/prerun_predictions.md`) were committed before the experiments they predict:

| Prediction | Commit | Before |
| --- | --- | --- |
| Q2.5 | `33d4a9f` (2026-10-27 15:23 PKT) | mirror, rotation and speed tests of model A |
| Q2.11 | `eeee372` (2026-10-27 15:34 PKT) | any training of B, C, D or the D candidates |

GIFs: `Question2/results/gifs/fig3_*.gif` (model A) and `fig6a_*.gif` (final model). Each steps through the five forecast horizons with the four rows of Figure 3.

## Environment

Python 3.12, PyTorch 2.9.0 and torchvision 0.24.0 (CUDA 12.8), on an RTX 4060 Laptop GPU.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Question 2

`Question2/SPEC.md` restates the handout's protocol in implementation terms. Run everything from `Question2/`; the dataset downloads on first use.

`Question2/Question2.ipynb` runs the pipeline in one place, with all cells executed (about 5 minutes on the RTX 4060). It retrains model A, seed 0, as a demonstration, into `checkpoints/notebook/`. It then re-scores all 18 test-set predictors from the committed checkpoints, and every metric matches the saved JSON exactly. The scripts below produce everything the report uses:

```bash
python checks.py                                   # Table 1 and extra checks (exits non-zero on failure)
python flow_stats.py                               # Figure 1
python train.py --model A --seed 0                 # likewise A/B/C/D/DB, seeds 0 and 1
python evaluate.py --predictor A --seeds 0 1 --parts basic --unlock-baseline
python evaluate.py --predictor A --seeds 0 1 --parts consistency,speed,timing --unlock-baseline
python evaluate.py --predictor D --seeds 0 1 --symmetrised --parts basic,consistency,speed,timing
python screen_d.py                                 # model D design, chosen on the validation split
python make_tables.py                              # LaTeX tables into ../report/generated/
python make_figures.py fig2                        # fig3, fig4, fig5; fig6 --model D --sym --ref A
python plot_loss_curves.py                         # supplementary Figure S1
```

| File | Purpose |
| --- | --- |
| `src/data.py` | Moving MNIST loading and the video-level split (0–7,999 / 8,000–8,999 / 9,000–9,999) |
| `src/transforms.py` | Mirror, rotation and the 8-element group G |
| `src/windows.py` | Speed-k windows with 1-based frame numbers and target masks |
| `src/models.py` | ConvLSTM forecaster (model A and variants), Blank, Copy-last, symmetrised wrapper |
| `src/registry.py` | Models A, B, C, D, D+B: architecture, training speeds, augmentation |
| `src/train.py`, `src/metrics.py` | Training protocol; per-video MSE, consistency, inference time |
| `results/metrics/` | All metrics (JSON) and per-video tensors; `d_screening.json` records the model D choice |
| `results/figures/`, `results/gifs/` | Figures 1–6, S1 and the GIFs for Figures 3 and 6(a) |
| `checkpoints/` | Trained weights for every model and seed |

`evaluate.py` refuses to run the Testing-the-Baseline experiments on model A without `--unlock-baseline`.
This enforced the pre-registration rule: the Q2.5 and Q2.11 predictions in `report/prerun_predictions.md` were committed before the corresponding experiments ran.

## Report

```bash
cd report && python build_ai_appendix.py && pdflatex main.tex && pdflatex main.tex
```
