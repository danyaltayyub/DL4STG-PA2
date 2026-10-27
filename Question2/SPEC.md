# Question 2 implementation spec

The contract every Question 2 script follows. It restates the handout (`DL4STG_PA2.pdf`, pp. 4–13) in implementation terms.
If anything here conflicts with the handout, the handout wins: stop and report the conflict instead of guessing.

Run everything from `Question2/` with `../.venv/bin/python`. Data root: `Question2/data`.

## 0. Conventions

- Device: CUDA when available. float32 with PyTorch's default precision settings (no AMP, no deterministic-algorithm flags).
- Frame numbering: the handout's frames 1–20 are tensor indices 0–19. Every argument called `t` or "frame" is **1-based**, as in the handout.
- Shapes: a video is `[..., 20, 1, 64, 64]`. A window is `X, Y` of shape `[B, 5, 1, 64, 64]` plus a bool `mask [B, 5]` (True = that target frame exists).
- Pixel `x[..., i, j]`: `i` is the row (0 at the top), `j` the column.

## 1. Data — `src/data.py`

- `load_videos(root="data")` returns a uint8 tensor `[10000, 20, 1, 64, 64]` from `torchvision.datasets.MovingMNIST(root=root, split=None, download=True).data`. Never use `split="train"` or `split="test"`: they split frames, not videos.
- Split by video index: train 0–7,999, validation 8,000–8,999, test 9,000–9,999.
- Convert with `x.float() / 255`. Keep uint8 on the GPU and convert per batch.

## 2. Transformations — `src/transforms.py` (act on the last two dims of any tensor)

- `mirror(x)`: `(Mx)[i, j] = x[i, 63 - j]`, i.e. `torch.flip(x, dims=(-1,))`.
- `rot(x, a=1)`: `torch.rot90(x, k=a, dims=(-2, -1))`, i.e. 90° anticlockwise per step.
- Group G has 8 elements, index `g = 4*m + a` with `m ∈ {0,1}`, `a ∈ {0,1,2,3}`:
  - `apply(g, x)` = `rot(x, a)` if `m = 0`, else `mirror(rot(x, a))` (rotate first, then mirror).
  - `inverse(g, x)` = `rot(x, -a)` if `m = 0`; if `m = 1`, the element is its own inverse, so `apply(g, x)`.
  - `g = 0` is the identity.
- Transforming a window applies the same transformation to every frame of both `X` and `Y`. The mask is unchanged.

## 3. Windows — `src/windows.py`

- For speed `k ∈ {1,2,3}` and start `t` (1-based): input frames `t, t+k, …, t+4k`; target frames `t+5k, …, t+9k`. A target exists only if its frame number is ≤ 20.
- `window_frames(t, k)` returns `(input_frames, kept_target_frames)` as 1-based frame numbers.
- `make_windows(videos, t, k)` returns `X, Y, mask`. `t` and `k` may be ints or LongTensors of shape `[B]` (one per video). Missing target frames are zero in `Y` and False in `mask`. All five input frames must exist (assert).
- Allowed starts:
  - Normal-speed training (models A, B): `k = 1`, `t ~ U{1, …, 11}`, so all five targets exist.
  - Speed-augmented training (models C, D, DB): `k ~ U{1,2,3}`, then `t ~ U{t : t + 6k ≤ 20}`, i.e. `1..14` at 1×, `1..8` at 2×, `1..2` at 3×. All inputs and at least two targets exist.
- Evaluation always uses `t = 1` unless stated otherwise. At 3× with `t = 1` only targets 16 and 19 exist (h = 1, 2).

## 4. Models — `src/models.py`, `src/registry.py`

### ConvLSTM forecaster (model A and every variant)

`ArchConfig(patch=4, hidden=64, kernels=(3, 3), dilations=(1, 1))`. The number of ConvLSTM layers is `len(kernels)`.

1. Fold: `u = pixel_unshuffle(frame, patch)`. For model A, `[B,1,64,64]` becomes `[B,16,16,16]`.
2. ConvLSTM layer ℓ uses one `Conv2d(in = C_in(ℓ) + hidden, out = 4*hidden, kernel_size = k_ℓ, dilation = d_ℓ, padding = d_ℓ*(k_ℓ - 1)//2, bias=True)` on `cat([u_ℓ, h_ℓ(τ-1)], dim=1)`. Split the output with `chunk(4, dim=1)` into `i, f, o, g`, in that order:
   - `c = sigmoid(f) * c_prev + sigmoid(i) * tanh(g)`
   - `h = sigmoid(o) * tanh(c)`

   There are no peephole terms. Layer 1's input is the folded frame; layer ℓ+1's input is layer ℓ's new `h` from the same step. Each layer has its own weights. `h` and `c` start at zero for every video.
3. Head: `Conv2d(hidden, patch**2, kernel_size=1, bias=True)` on the top layer's `h`, then `pixel_shuffle(·, patch)`, then `sigmoid`, giving `[B,1,64,64]`.

- Use PyTorch's default initialisation. No normalisation, dropout, residual connections or extra layers.
- **Assert: model A (default `ArchConfig`) has exactly 480,784 trainable parameters.**
- Rollout, `forward(X) -> Y_hat` (`[B,5,1,64,64]` in and out). It runs 9 steps, τ = 1..9:
  - The input `u_τ` is `X[:, τ-1]` for τ ≤ 5. For τ ≥ 6 it is the previous step's prediction, with no detach and no teacher forcing.
  - Step τ outputs `x̂_{τ+1}`. The outputs of τ = 5..9 form `Y_hat` (`x̂_6 … x̂_10`). Outputs of τ = 1..4 are discarded, so the head may be skipped there.
  - The rollout is identical in training and evaluation, at every speed; the model never sees `k`.

### Reference predictors

- `Blank`: `Y_hat = 0`.
- `CopyLast`: `Y_hat[:, h] = X[:, 4]` for every h.
- Neither has parameters.

### Symmetrised wrapper

`Symmetrised(f)(X) = (1/8) * Σ_{g ∈ G} inverse(g, f(apply(g, X)))`. The eight transformed copies may be stacked into one batch for a single forward pass.

### Registry

| name | architecture | training speeds | symmetry augmentation |
|---|---|---|---|
| A | `ArchConfig()` | normal (1× only) | none |
| B | `ArchConfig()` | normal | one `g ~ U(G)` per window |
| C | `ArchConfig()` | `k ~ U{1,2,3}` | none |
| D | `ARCH_D` (chosen later on the validation split; any `ArchConfig` must work) | `k ~ U{1,2,3}` | none |
| DB | `ARCH_D` | `k ~ U{1,2,3}` | one `g ~ U(G)` per window |

Model D must have at most 721,176 trainable parameters.

## 5. Training — `src/train.py`, CLI `train.py --model A --seed 0 [--epochs 30] [--tag NAME] [--arch JSON]`

- Adam, learning rate 1e-3, no schedule, batch 32, 30 epochs. Keep the final-epoch weights: no early stopping, no checkpoint selection.
- An epoch is one pass over the 8,000 training videos in shuffled order. Each video contributes one window, sampled afresh every epoch under its model's rule (§3). With symmetry augmentation, draw one `g` per window and apply it to both `X` and `Y`.
- Loss: squared error averaged over pixels and **available** target frames of each video, then over the videos in the batch:
  - `se = ((Y_hat - Y)**2).mean(dim=(2,3,4))`, shape `[B,5]`
  - `loss_v = (se * mask).sum(1) / mask.sum(1)`
  - `loss = loss_v.mean()`
- Seeds: exactly 0 and 1. The seed controls:
  - initialisation: `torch.manual_seed(seed)` immediately before building the model
  - shuffling, window sampling and augmentation sampling: one CPU `torch.Generator().manual_seed(seed)` used for all of them
- Timing: wall-clock per epoch, calling `torch.cuda.synchronize()` before each clock read. Report the mean over all 30 epochs, excluding data loading and download.
- Outputs:
  - `checkpoints/{name}_s{seed}.pt`: `{state_dict, arch, name, seed}`
  - `results/runs/{name}_s{seed}.json`: `{epoch_loss: [...], epoch_time_s: [...], train_time_per_epoch_s, params, hardware (GPU and CPU names), torch_version, arch}`
- A `--tag` other than the default writes under `results/smoke/` and `checkpoints/smoke/`. Use it for test runs that are not results.
- Performance: keep the split tensors on the GPU and gather windows with index tensors. No DataLoader is needed.

## 6. Metrics and evaluation — `src/metrics.py`, CLI `evaluate.py`

For a predictor `f` on split S (default test: 1,000 videos), batch 32, `model.eval()` and `torch.no_grad()`:

- `MSE_h`: per video and horizon, the mean over the 64×64 pixels of horizon h, shape `[N,5]`; NaN where the target is missing.
- `E5`: mean of `MSE_h` over h = 1..5, then over videos.
- `E2`: mean over h = 1, 2, then over videos. Use E2 whenever comparing across speeds.
- Consistency at 1×, t = 1, all five horizons, for `T ∈ {M = mirror, R = rot(a=1)}`:
  - `C_T = mean over h and pixels of (f(T(X)) - T(f(X)))**2`, per video
  - `E_T = mean over h and pixels of (f(T(X)) - T(Y))**2`, per video
  - Report the mean over videos and keep the per-video arrays.
- Inference time: all 1,000 test windows at 1×, t = 1, batch 32.
  - Move every batch to the device first, then synchronise and run one untimed warm-up batch.
  - Time the loop over all batches, with a synchronise at the end.
  - For symmetrised models the timed region includes the 8 transforms, forward passes, inverse transforms and the averaging.
- CLI: `evaluate.py --predictor {A,B,C,D,DB,blank,copylast} [--seeds 0 1] [--symmetrised] [--split test|val] [--tag NAME] --parts basic,consistency,speed,timing`
  - `basic`: 1×, t = 1: MSE_1..MSE_5, E5, E2, parameter count.
  - `consistency`: C_M, E_M, C_R, E_R.
  - `speed`: t = 1 at 1×, 2×, 3×: E2 at all three speeds, E5 at 1× and 2×.
  - `timing`: inference time.
  - Outputs:
    - `results/metrics/{name}[_sym]_s{seed}_{split}.json`: scalar summaries, merged across separate calls
    - `results/metrics/{name}[_sym]_s{seed}_{split}_pervideo.pt`: per-video tensors, e.g. `mse_1x [N,5]`, `mse_2x`, `mse_3x`, `C_M [N]`, `E_M`, `C_R`, `E_R`
    - `blank` and `copylast` have no seed, so drop `_s{seed}`
  - Each part runs on its own. **`consistency` and `speed` are the "Testing the Baseline" experiments: they must not run on a real model A checkpoint until the main thread says so** (pre-registration rule). Smoke checkpoints are fine.

## 7. Checks — `checks.py` (produces Table 1)

Synthetic tensors:
- random video: `torch.manual_seed(0); x = torch.rand(20, 1, 64, 64)`
- marker video: `p` all zeros except `p[7, 0, 10, 20] = 1`

Table 1:

| row | check | expected |
|---|---|---|
| (a) | max \|M(M(x)) − x\| | 0, and `torch.equal` is True |
| (b) | max \|R⁴(x) − x\| | 0, and `torch.equal` is True |
| (c) | max \|M R M(x) − R⁻¹(x)\| | 0, and `torch.equal` is True |
| (d) | bright pixel of M(p) (frame, row, column) | (7, 10, 43) |
| (e) | bright pixel of R(p) | (7, 43, 10) |
| (f) | 1× input / target frames | 1–5 / 6–10 |
| (g) | 2× input / target frames | 1,3,5,7,9 / 11,13,15,17,19 |
| (h) | 3× input / target frames | 1,4,7,10,13 / 16,19 |

For (f)–(h), call `make_windows` with t = 1 on a video whose frame f (1-based) is filled with the value f, and read the frame numbers off the values.

Extra checks, also printed:
- `inverse(g, apply(g, x)) == x` for all 8 g, and the 8 elements are distinct on `x`
- model A has 480,784 parameters
- Blank and CopyLast have C_M = C_R = 0 exactly
- `Symmetrised` of a randomly initialised model A has C_M and C_R ≤ 1e-10

Write `results/metrics/table1.json` and print the table. Exit non-zero on any failure.

## 8. Figure style — `src/plotstyle.py` (matplotlib; save PNG at 200 dpi and PDF)

- White background. A single-series chart uses one hue, blue `#2a78d6`. If a second series is ever needed, it uses orange `#eb6834`.
- Reference lines (uniform level, median, marked videos): `#52514e`, 1.2 pt, dashed. Label them directly with text in `#52514e`. A single-series chart gets no legend.
- Text: titles `#0b0b0b`; axis labels and ticks `#52514e`. Hairline y-grid `#e1e0d9` drawn behind the bars. No top or right spines.
- Histogram bars get a thin white edge (linewidth 0.8) so adjacent bars stay separate.
- Frame grids: `cmap="gray"`, `vmin=0`, `vmax=1`, no ticks; row and column labels in `#52514e`.

## 9. Report artefacts (built later, listed so the outputs above cover them)

- Table 1: checks. Figure 1: optical-flow direction and speed histograms (first 50 training videos).
- Table 2: MSE_1..MSE_5, E5 and parameter count for A, Blank, Copy-last. Figure 2: input, prediction and truth for three test videos (one with overlapping digits).
- Table 3: E5, C_M, E_M, C_R, E_R for A, Blank, Copy-last.
- Figure 3: for the 2 test videos with the largest per-video C_M and the 2 with the largest C_R (seed-0 A), 4 rows × 10 columns: (1) X | f(X), (2) T(X) | T(Y), (3) T(X) | f(T(X)), (4) T(X) | T(f(X)). Plus one GIF per panel stepping through h = 1..5 with the same four rows.
- Figure 4: histograms of per-video C_M and C_R (seed-0 A), with the Figure 3 videos marked.
- Table 4: A's E2 at 1×/2×/3× and E5 at 1×/2×, with Copy-last for comparison. Figure 5: one test video with no digit overlap in the displayed true frames: five inputs plus the first two predicted and true frames at 1×, 2×, 3×.
- Table 5: for A–D, E5 (1×); C_M, E_M, C_R, E_R; E2 at 1×/2×/3×; params; training time per epoch; inference time.
- Table 6: the Table 5 columns for symmetrised B, symmetrised D and the final model, if different.
- Figure 6: Figures 3 and 5 repeated with the final model on the same videos, plus GIFs.
- Formatting: every learned-model entry is `mean (seed 0, seed 1)`; 3 significant figures; scientific notation below 1e-3. The seed-0 model is used for figures, GIFs and per-video histograms.
