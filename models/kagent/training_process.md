# kagent

Submission produced autonomously by a coding agent ("alphonse") competing
against seven other agents on the public warped-ifw dataset. Best val/l2_error
during autonomous training: **0.8707**.

## Architecture

`VoxelResidualModel` — residual-from-last-frame predictor with a voxel-UNet
spatial mixer and a wall-distance feature.

Per-point input features (22 dims):
- 5 past velocity frames flattened (15)
- 3D position (3)
- Airfoil surface mask (1)
- Signed distance to the nearest airfoil point, raw and `log1p` (2)

Backbone:
- Linear projection to hidden=256
- 2 × ResMLP pre-blocks
- `VoxelSpatial` — scatter-mean into a 64³ voxel grid → 3D UNet (c_mid=64,
  three levels with avg-pool/trilinear upsample) → grid-sample back to points,
  added as a residual
- 4 × ResMLP post-blocks
- Final linear head predicts the **residual** from the last input frame in
  normalised velocity space; prediction is `last_frame + delta * vel_std`

Post-processing:
- Zero velocity at `idcs_airfoil` to respect the no-slip boundary condition

`vel_mean` and `vel_std` are stored as registered buffers and live in the
state dict; they were fit on the training split.

## Training

- Dataset: `warped-ifw` official splits (146 train sims, 16 val sims)
- Optimizer: AdamW, lr=5e-4, weight_decay=1e-4
- Schedule: cosine anneal over 60 epochs
- Batch size: 1 (100k points per sample)
- Loss: MSE on velocity predictions
- Training ran under a 30-minute per-iteration wall-clock budget in
  a single-GPU pod (96GB VRAM available, ~20GB used).

## Signature mismatch with the training codebase

Training used `model(velocity_in, pos, t, idcs_airfoil, sdf)` with the SDF
precomputed once per sample. The competition signature
`model(t, pos, idcs_airfoil, velocity_in)` has no SDF input, so the
submission's `Model` computes it on the fly inside `forward` using a chunked
`cdist`.

## Notes

Training was fully autonomous — the agent read its own leaderboard, wrote its
own code, and iterated six versions before converging on this architecture.
Earlier attempts (pure ResMLP, EMA, mirror-flip augmentation, larger UNet)
all underperformed.
