# LeversTailV2Submission — architecture & design notes

This document describes the **submitted model** `LeversTailV2Submission` located in:

- `models/levers_tail_submission/model.py`

and the committed weights file:

- `models/levers_tail_submission/state_dict.pt`

The implementation is **self-contained** (no imports from other `models/*` packages) to
keep the submission easy to audit and portable.

---

## Problem setting (what the model solves)

Given a **fixed point cloud** in \(\mathbb{R}^3\) and a short input velocity time series,
predict the continuation of the velocity field.

### Inputs

- **`t`**: \((B, 10)\) — time stamps for the full window (input + output)
- **`pos`**: \((B, N, 3)\) — spatial points
- **`idcs_airfoil`**: list of length \(B\); each is a 1D tensor of indices into `pos`
  marking **surface points** (airfoil boundary)
- **`velocity_in`**: \((B, 5, N, 3)\) — 5 input timesteps of 3D velocity

### Output

- **`velocity_out`**: \((B, 5, N, 3)\) — 5 predicted timesteps of 3D velocity

---

## High-level architecture

`LeversTailV2Submission` consists of:

1. **Feature construction** per point (geometry + temporal context)
2. **kNN graph** on the point cloud
3. **Neighbor attention pooling** (learned local aggregation)
4. **Two residual message passing blocks** (richer local mixing)
5. **Pointwise MLP trunk**
6. **Per-output-timestep heads** to produce \((5 \times 3)\) velocity components

The backbone module is named `StrongMLPKnnMPv2` inside `model.py`.

---

## Detailed components & justification

### 1) Per-point input features

The model builds a per-point feature vector that concatenates:

- **Position** `pos` (\(3\) dims)
  - *Why*: geometry conditioning; defines the spatial domain.
- **Flattened input velocities** (flatten \((T_{in}=5, 3)\) into \(15\) dims)
  - *Why*: provides local temporal state at each point.
- **Broadcast time stamps** `t` (\(10\) dims) duplicated at each point
  - *Why*: allows a time-conditioned operator mapping even when sampling times differ.
- **Surface indicator** (1 dim) derived from `idcs_airfoil`
  - *Why (CFD)*: near-wall physics differs strongly from bulk flow (no-slip, boundary layers).
- **Neighbor attention pooled vector** (learned summary of neighborhood)
  - *Why*: injects locality beyond a pure pointwise MLP.
- **Per-input-timestep neighbor mean velocity** (5×3 dims)
  - *Why*: a simple, stable statistic that captures neighborhood dynamics across input time.
- **Distance-to-surface features**: \([d, \log(1+d)]\) (2 dims)
  - *Why (CFD)*: wall influence decays with distance; `log1p` improves dynamic range.

### 2) kNN graph over `pos`

The model constructs a **k-nearest-neighbor** graph using Euclidean distance.

- *Why*: the dataset is a point cloud, not a mesh; kNN approximates a local stencil.
- *Practical*: the implementation uses a row-chunked brute-force routine for robustness.

### 3) Neighbor attention pooling

For each point \(i\), neighbors \(j\) are scored via a dot-product attention:

- Query from \([x_i, \bar{v}_i, s_i]\) where \(s_i\) is the surface flag
- Keys/values from neighbor \([\bar{v}_j, (x_j - x_i)]\)

This yields:

- a pooled neighbor vector (value aggregation)
- attention weights (used again for message passing aggregation)

*Why*: flow dependence is local but anisotropic (wake directionality, shear layers). Attention
lets the model **learn** which neighbors matter.

### 4) Rich edge features

Each edge carries:

- **\(\Delta x\)** (relative position)
- **\(\Delta \bar{v}\)** (relative mean velocity)
- **\(1/(||\Delta x||+\epsilon)\)** (inverse distance)

*Why*: local gradients and geometry strongly drive fluid evolution; relative signals are a
natural inductive bias for PDE-like fields.

### 5) Two residual MP blocks

Each MP block:

- builds edge messages with an MLP
- aggregates messages with **attention weights**
- updates node state with an MLP
- applies **residual connection + LayerNorm**

Two blocks improve expressive power while keeping complexity local.

### 6) Merge + MLP trunk

The trunk sees **raw features** concatenated with **both MP states**. This guards against
oversmoothing (raw signals are still available) while enabling deeper nonlinear mixing.

### 7) Per-timestep output heads

The final latent per point is mapped through 5 separate linear heads to output the 3D
velocity at each future timestep.

*Why*: each horizon step can have different error structure; separate heads provide
timestep-specific capacity without changing the contract.

---

## “Levers tail” (training recipe vs inference architecture)

The submitted checkpoint corresponds to the **v2 backbone** trained with a recipe that
increases optimization pressure on later prediction horizons (“tail”) while retaining
stability levers (e.g., EMA / weighted loss in training runs).

Important: **the submission-time model is purely the learned neural mapping**; loss
weighting is only a training-time choice.

---

## How to validate locally

From repo root:

```bash
export PYTHONPATH=.
python scripts/verify_submission_contract.py --class-name LeversTailV2Submission --num-pos 100000 --batch-size 1
pytest -q -m "not hf"
```

