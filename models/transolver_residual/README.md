# TransolverResidual — GRaM 2026 Submission

**Task:** Predict 5 future velocity field timesteps from 5 input timesteps on irregular
3D point clouds (~100k points) around F1-style airfoil geometries.

**Approach:** Physics-Attention Transolver on top of a polynomial extrapolation baseline.
The network learns only the turbulent *correction*; the low-frequency dynamics are
handled analytically for free.

---

## Architecture

```
velocity_in (B, 5, N, 3)
      │
      ├─ poly_extrapolate() ──────────────────────────────────────────► poly_baseline (B, 5, N, 3)
      │     quadratic fit at input times, evaluated at output times
      │
      ├─ compute_features()
      │     pos_normalized    ( 3)  bounding-box normalised position
      │     velocity_in       (15)  5 input snapshots flattened
      │     poly_residual     (15)  v_in − poly_fit(v_in) — turbulence proxy
      │     temporal_mean     ( 3)  mean velocity over input window
      │     temporal_std      ( 3)  std velocity over input window
      │     is_airfoil        ( 1)  binary surface flag
      │     dist_to_airfoil   ( 1)  distance to nearest surface point
      │     upstream_dist     ( 1)  signed x-offset to nearest surface point
      │     t_values          (10)  all 10 time values, broadcast to every point
      │     local_nbr_mean    (15)  mean velocity of 8 nearest neighbours (k-NN)
      │     temporal_deltas   (12)  Δv_t = v_t − v_{t−1} for t = 1..4
      │     ─────────────────────
      │     total             (79)
      │
      ├─ MLP encoder: 79 → 512 → 256 (GELU)
      │
      ├─ 8 × TransolverBlock
      │     PhysicsAttention(dim=256, heads=8, slices=32)
      │     FFN(dim=256, ratio=1)
      │     LayerNorm + residual connections
      │
      ├─ LayerNorm + Linear: 256 → 15
      │
      ├─ reshape: (B, N, 15) → (B, 5, N, 3)  [correction]
      │
      └─ output = poly_baseline + correction
                  zeroed at idcs_airfoil  (hard no-slip enforcement)
```

**Parameters:** 2.85M

---

## Design Decisions

### Residual Learning on a Polynomial Baseline

The polynomial extrapolation (degree 2) handles low-frequency laminar dynamics.  
The Transolver learns only the turbulent correction — the part the polynomial gets wrong.
Benefits:
- Zero-initialised decoder → model starts as a pure polynomial baseline; gradient descent
  learns corrections from a stable initialisation.
- Graceful degradation: if the correction is poor, the polynomial still provides a
  reasonable prediction.
- Learning target is near zero-mean, which is easier to optimise.

### Physics-Attention (Transolver)

Instead of attending over all N=100k points (O(N²)), each point is soft-assigned to one
of M=32 physics slices via a learned linear projection + softmax.  Standard self-attention
runs only among the M slice tokens, then broadcasts back.  
Complexity: O(N·M·C + M²·C) — linear in N.

The slices learn to separate physical regimes (freestream, boundary layer, wake) without
explicit supervision.

### Feature Engineering

- **`poly_residual`:** deviation of the input velocity from the polynomial fit — a direct,
  explicit proxy for where turbulence is already present in the input window.
- **`dist_to_airfoil` + `upstream_dist`:** compact geometry encoding that differentiates
  upstream (clean flow) from downstream (wake / interference) relative to the surface.
- **`local_nbr_mean`:** mean velocity of 8 nearest spatial neighbours.  Gives each point
  awareness of its local flow context, compensating for the lack of explicit local
  aggregation in the Transolver.
- **`temporal_deltas`:** velocity differences Δv_t encode local acceleration and the
  arrow of time.

### No-Slip Boundary Condition

Velocity at `idcs_airfoil` points is hard-zeroed after the final addition.  
Exact, free, and guaranteed — the network never needs to learn to predict zeros at the
surface.

---

## Training

### Data

- 905 samples (181 simulations × 5 time windows), all used for training.
- 90 / 10 random train / val split.
- Each sample: `velocity_in (5, 100k, 3)`, `pos (100k, 3)`, `idcs_airfoil`, `velocity_out (5, 100k, 3)`.

### Preprocessing (offline)

Distance features (`dist_to_airfoil`, `upstream_dist`, `is_airfoil`) and k-NN indices
are precomputed per simulation and stored as `.distcache.npz` / `.knncache.npz` sidecar
files to avoid recomputing them every epoch.

### Loss

Relative L² loss on the velocity correction (predicted output vs. ground truth):

```
loss = ‖ velocity_out_pred − velocity_out_gt ‖₂ / ‖ velocity_out_gt ‖₂
```

### Optimiser

Adam (lr=1e-3) with cosine annealing over 400 epochs.  
Gradient accumulation over 4 steps (effective batch size 4 with batch_size=1).

### Augmentation

Y-axis flip: the velocity y-component and position y-coordinate are negated.
This is the dominant augmentation — without it, the model cannot break y-symmetry
because the training geometries are y-biased (consistent pitch orientation).
Layer-0 slice entropy dropped from ~100% to ~10.6% after adding y-flip augmentation,
indicating the slices transitioned from uninformative to physically meaningful routing.

### Hardware

Single NVIDIA A100 (40 GB).  Training time: ~6 hours for 400 epochs.

---

## Validation Results

| Metric | Value |
|--------|-------|
| Val rel-L2 (mean over t5–t9) | 0.066 |
| Val rel-L2 at t5 (first future step) | 0.069 |
| Val rel-L2 at t9 (last future step) | 0.084 |
| Polynomial baseline at t9 | 1.277 |

The model achieves ~15× error reduction over the polynomial baseline at the longest
horizon (t9), and a modest +22% error growth from t5 to t9 (vs. +514% for the polynomial).

---

## Inference Notes

The model caches per-geometry distance features and k-NN indices by geometry fingerprint.
The first call on a novel geometry (not seen during training) incurs a one-time precompute
cost (~5–10 seconds for 100k points on CPU).  Subsequent calls on the same geometry
(e.g. different time windows of the same simulation) are instant.

The model is fully self-contained: no external config files are required.
`TransolverResidual()` instantiates with trained weights loaded automatically.

---

## Reproduction

```bash
# 1. Precompute distance and k-NN caches
python precompute_caches.py --data_dir ./data/gram/

# 2. Train
python train.py \
    --n_layers 8 --hidden_dim 256 --epochs 400 \
    --lr 1e-3 --accum_steps 4 --num_workers 16 \
    --train_fraction 1.0 --augment \
    --use_local_feats --use_temporal_deltas \
    --run_name run_final

# 3. Best checkpoint is saved to models/transolver_residual/weights.pt
```
