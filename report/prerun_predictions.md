# Pre-run predictions (Question 2)

The handout requires Q2.5 and Q2.11 to be written before the corresponding experiments and committed to the repository before those experiments run.
The report quotes these answers unchanged, even where the measurements later contradict them.

## Q2.5: model A under mirror, rotation and speed-up

Recorded before any experiment in "Testing the Baseline" (Tables 3–4, Figures 3–5).
Inputs used: Figure 1 (training-motion statistics) and the reach estimate of Q2.3.
Model A's test numbers for mirror, rotation and speed-up had not been computed when this was written.

Status: **Approved by the student without changes**. Recorded: 2026-10-27 15:20 PKT, before any experiment in Testing the Baseline.

**Mirror test: should pass, approximately.**
Figure 1(a) shows training motion in every direction with no empty sector, so a mirrored trajectory (θ → 180° − θ) is familiar motion. Only the digits look unfamiliar.
Model A is convolutional and mostly has to carry local strokes along, so mirrored digits should cost little. Expected: E_M within about 10% of E^(5), and C_M well below E_M.

**Rotation test: should also pass, but less cleanly than mirror.**
A 90° rotation (θ → θ + 90°) again gives familiar motion. But it turns vertical strokes into horizontal ones, and the learned 3×3 filters are not rotation-equivariant, so the appearance shift is larger.
Expected: E_R > E_M and C_R > C_M, with E_R up to about 20–30% above E^(5). Degraded, not broken.

**Speed-up: should break down at 3×.**
Model A's reach is 8 px per time step along each axis (Q2.3; up to 8√2 ≈ 11.3 px diagonally). The median training speed is 3.3 px/frame (Figure 1(b)), so 2× ≈ 6.6 px/frame and 3× ≈ 10 px/frame.
- At 2×, motion stays within reach but is faster than anything in training (90th percentile 4.1 px/frame). Predictions should degrade, with digits lagging behind their true positions, but stay roughly on track.
- At 3×, the displacement exceeds the reach for most directions (any motion within about 37° of an axis). The network cannot link a digit's old position to its new one in one update. Digits should lag, smear or fade, and E^(2) should come close to, or exceed, Copy-last's.

## Q2.11: which of models B–D fixes which failure (at most three sentences)

Recorded before models B, C and D are trained.

Status: _not yet drafted_.
