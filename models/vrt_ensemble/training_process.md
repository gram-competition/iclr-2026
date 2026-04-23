# Volumetric Routing Transformer (VRT): full technical description

The name **Volumetric Routing Transformer (VRT)** reflects the three core ideas:

- **Volumetric**: point-cloud features are projected to a structured 3D lattice
  (volume) and processed with multi-scale 3D convolutional operators.
- **Routing**: information is routed point -> lattice -> point, allowing global
  spatial context exchange while keeping pointwise representation compatibility.
- **Transformer**: pointwise token refinements use transformer-style residual
  blocks with normalization and gated feed-forward mixing (SwiGLU-like blocks),
  and temporal information is encoded in a token-centric manner.

In short, VRT combines unstructured point processing with structured volumetric
reasoning and transformer-style token updates.

## Block diagram (architecture overview)

```text
Inputs
  t (B,10), pos (B,N,3), idcs_airfoil, velocity_in (B,5,N,3)
        |
        v
[Feature Builder]
  - position Fourier features
  - velocity history + spectral features
  - boundary distance + direction features
  - temporal velocity encoder
  - boundary mask / flow stats
        |
        v
[Linear Projection to Hidden Dim]
        |
        v
[Pre Pointwise Blocks x3  <-- Transformer-style token blocks]
  PointwiseSwiGLU residual refinement (LN + gated FFN + residual)
        |
        v
[Volumetric Routing Core]
  scatter points -> 3D lattice (64x32x32)
        -> 3D ConvNeXt-V2 UNet (multi-scale)
        -> sample lattice back to points
        |
        v
[Post Pointwise Blocks x6]
  pointwise residual refinement (same transformer-style token blocks)
        |
        v
[LayerNorm + Delta Head]
  predict 5x3 residual velocity channels per point
        |
        v
[Residual Add from Last Input Frame]
        |
        v
[No-slip Boundary Projection]
  enforce zero velocity on airfoil indices
        |
        v
Output
  velocity_out (B,5,N,3)
```

This document describes the exact model family, training protocol, loss
formulation, and final submission assembly used for the `submitted_vrt` entry.
It is written as an implementation-level technical note suitable to reproduce
the submission behavior end to end.

---

## 1) Problem setting and tensor interface

The model solves a spatiotemporal velocity forecasting task on 3D point clouds:

- Input time grid: `T_in = 5` historical frames
- Output horizon: `T_out = 5` future frames
- Spatial points: `N = 100000` (full resolution)
- Velocity channels: `C = 3`

Expected callable signature (competition format):

```python
model(
    t: Tensor[B, 10],
    pos: Tensor[B, N, 3],
    idcs_airfoil: list[Tensor[Mi]],
    velocity_in: Tensor[B, 5, N, 3],
) -> Tensor[B, 5, N, 3]
```

No-slip boundary condition is explicitly enforced on indices `idcs_airfoil`.

---

## 2) Code locations

This submission contains:

- a single-member VRT architecture implementation,
- an ensemble wrapper that combines four trained VRT members,
- training-compatible loss definitions and normalization logic.

The architecture and wrapper are self-contained in the submission package.

---

## 3) Architecture (single VRT member)

### 3.1 High-level decomposition

Each forward pass is structured as:

1. Build geometry- and dynamics-aware pointwise features
2. Apply lightweight pointwise pre-routing blocks
3. Route features through a structured volumetric lattice solver
4. Apply post-routing pointwise refinement
5. Decode residual future velocity and enforce physical boundary constraints

This combines unstructured point-cloud inputs with structured 3D-grid operators.

### 3.2 Input feature construction

For each point, the feature stack includes:

- **Position Fourier features** (`num_pos_freqs=10`)
- **Flattened velocity history** (`5 x 3 = 15`)
- **Last-frame velocity spectral features** (`num_vel_freqs=4`)
- **Temporal encoder features** from a 1D conv stack over time
- **Boundary geometry features**:
  - nearest-surface distance
  - `log1p(distance)`
  - per-sample normalized distance
  - inverse distance
  - Fourier encoding of `log1p(distance)` (`num_dist_freqs=8`)
  - unit direction vector to nearest boundary point
- **Flow statistics features**:
  - per-point mean speed over input frames
  - per-point std speed over input frames
- **Boundary mask**

Nearest-boundary geometry is computed by chunked `torch.cdist` over surface points
for memory safety.

### 3.3 Pointwise blocks (pre/post routing)

Pre- and post-routing stages use `PointwiseSwiGLUBlock`, which is a
parameter-efficient residual MLP block with multiplicative gating.

For an input point feature $x \in \mathbb{R}^{D}$, the block computes:

```math
\tilde{x} = \mathrm{LN}(x)
```
```math
g = \mathrm{SiLU}(W_g \tilde{x}), \quad u = W_u \tilde{x}
```
```math
y = x + W_o (g \odot u)
```

where:

- $W_g, W_u \in \mathbb{R}^{D \times H}$
- $W_o \in \mathbb{R}^{H \times D}$
- $H = \left\lceil \frac{8}{3}D \right\rceil$ rounded to even
- $\odot$ is elementwise multiplication

Key properties:

- **Token-local transformation**: no explicit pairwise attention in this block
  itself, keeping per-point cost low.
- **Gated nonlinearity**: `SiLU(gate) * up` improves expressivity vs plain MLP.
- **Residual form**: stabilizes depth and preserves identity signal paths.
- **Bias-free linear maps** in gate/up/down projections reduce parameter
  overhead and often improve scaling behavior with LayerNorm.

Role in the full architecture:

- **Pre-routing blocks** (`3` blocks) refine local point descriptors before
  lattice projection.
- **Post-routing blocks** (`6` blocks) re-mix routed features after lattice
  globalization, acting as high-capacity pointwise correctors.

Configured in training runs as:

- pre blocks: `3`
- post blocks: `6`

### 3.4 Volumetric routing core

The core operator (`SOTAConvNeXtV2Lattice`) is a structured 3D backbone that
maps unordered point tokens to a Cartesian latent volume, processes that volume
with ConvNeXt-V2-style blocks, then maps back to points.

It performs:

1. **Scatter** point features to a fixed Cartesian lattice
2. **3D ConvNeXt-V2-style UNet** over lattice tensor
3. **Sample back** lattice features to points via trilinear `grid_sample`

Grid and channels used in final training:

- grid resolution: `64 x 32 x 32`
- base channels: `64`
- multi-scale channel pyramid: `64, 128, 256, 512`

Skip connections are additive (not concat), reducing memory pressure.

#### 3.4.1 Scatter to lattice

Let normalized coordinates be $c_i = (x_i, y_i, z_i)\in [0,1)^3$.
Each point is assigned to a voxel index:

```math
v_i = \lfloor x_i G_x \rfloor (G_y G_z) + \lfloor y_i G_y \rfloor G_z + \lfloor z_i G_z \rfloor
```

For feature $f_i \in \mathbb{R}^{D}$, lattice aggregation uses `scatter_add`
for both summed features and counts, then computes voxel averages:

```math
F_v = \frac{\sum_{i: v_i=v} f_i}{\max(1, n_v)}
```

This gives a dense tensor $F \in \mathbb{R}^{B \times D \times G_x \times G_y \times G_z}$.

#### 3.4.2 ConvNeXt-V2 3D block details

Each 3D block uses:

1. depthwise `Conv3d(kernel=7, groups=channels)` (spatial mixing at low FLOPs),
2. channel-last LayerNorm,
3. pointwise expansion (`Linear(C, 4C)`),
4. GELU,
5. GRN (Global Response Normalization),
6. pointwise contraction (`Linear(4C, C)`),
7. residual connection.

GRN normalizes channel responses using global spatial norms:

```math
g = \|x\|_{2,\text{spatial}}, \quad n = \frac{g}{\mathrm{mean}_c(g)+\epsilon}
```
```math
\mathrm{GRN}(x)=\gamma(x\odot n)+\beta+x
```

This encourages competition/cooperation between channels and improves stability
for deep ConvNeXt-style stacks.

#### 3.4.3 Multi-scale UNet path

The lattice backbone uses four scales:

- Encoder:
  - level1 (`C1=64`): 2 ConvNeXt-V2 blocks
  - downsample to `C2=128`, 2 blocks
  - downsample to `C3=256`, 2 blocks
  - downsample to `C4=512`
- Bottleneck:
  - 3 ConvNeXt-V2 blocks at `C4`
- Decoder:
  - transpose-conv upsample + **additive skip**
  - 2 blocks at `C3`
  - upsample + additive skip, 2 blocks at `C2`
  - upsample + additive skip, 2 blocks at `C1`

Additive skip design avoids channel doubling from concat, reducing memory and
bandwidth cost while preserving hierarchical information flow.

#### 3.4.4 Sample back to points

The final lattice tensor is sampled at point locations using trilinear
interpolation (`grid_sample`, border padding), producing routed point features
aligned to original point ordering.

### 3.5 Decoder and physical projection

After routing/refinement:

- `LayerNorm`
- linear head predicts `3 * T_out` residual channels per point
- reshape to `[B, T_out, N, 3]`
- add residual to last observed input frame
- enforce no-slip by zeroing predicted velocity on `idcs_airfoil`

---

## 4) Normalization and statistics handling

Each member stores external stats in `vrt_flow_stats.pt`:

- `flow_channel_mean`
- `flow_channel_scale`
- `spatial_bounds_lo`
- `spatial_bounds_hi`

During training, these statistics are fit from the training split and saved.
During inference, each member loads its own stats file before prediction.

---

## 5) Loss formulation (training objective)

VRT uses a weighted composite objective:

```math
\mathcal{L}
= \alpha \cdot \mathcal{L}_{mse}
+ \beta \cdot \mathcal{L}_{l1}
+ \gamma \cdot \mathcal{L}_{temp}
+ \delta \cdot \mathcal{L}_{airfoil} \cdot w_{airfoil}
+ \lambda_{grad} \cdot \mathcal{L}_{gmse}
```

Where:

- `mse`: global reconstruction MSE
- `l1`: global reconstruction L1
- `temp`: consistency of temporal deltas between prediction and target
- `airfoil`: mean squared velocity on airfoil points (no-slip pressure term)
- `gmse`: MSE between spatial finite differences of prediction and target

### 5.1 Explicit per-term definitions

Let prediction and target be $ \hat{u}, u \in \mathbb{R}^{B \times T \times N \times 3} $.

#### (a) Reconstruction MSE
```math
\mathcal{L}_{mse}=\frac{1}{BTN3}\sum_{b,t,n,c}(\hat{u}_{btnc}-u_{btnc})^2
```

#### (b) Reconstruction L1
```math
\mathcal{L}_{l1}=\frac{1}{BTN3}\sum_{b,t,n,c}\left|\hat{u}_{btnc}-u_{btnc}\right|
```

L1 adds robustness to outliers and sharp local deviations that MSE alone may
over-smooth.

#### (c) Temporal consistency term

Define frame-to-frame deltas:
```math
\Delta \hat{u}_{b,t}=\hat{u}_{b,t+1}-\hat{u}_{b,t}, \quad
\Delta u_{b,t}=u_{b,t+1}-u_{b,t}
```
```math
\mathcal{L}_{temp}=\mathrm{MSE}(\Delta \hat{u}, \Delta u)
```

This directly regularizes rollout dynamics, not just absolute frame values.

#### (d) Airfoil no-slip penalty

For boundary index set $\mathcal{A}_b$:
```math
\mathcal{L}_{airfoil}=
\frac{1}{B}\sum_b \mathrm{mean}_{t,n\in\mathcal{A}_b,c}\left(\hat{u}_{btnc}^2\right)
```

This pushes boundary velocity toward zero even when reconstruction terms alone
could tolerate small slip.

#### (e) Gradient MSE (GMSE)

Using finite differences along point dimension:
```math
\nabla_n \hat{u}_{b,t,n,c}=\hat{u}_{b,t,n+1,c}-\hat{u}_{b,t,n,c}
```
```math
\nabla_n u_{b,t,n,c}=u_{b,t,n+1,c}-u_{b,t,n,c}
```
```math
\mathcal{L}_{gmse}=\mathrm{MSE}(\nabla_n \hat{u}, \nabla_n u)
```

GMSE constrains local variation patterns and combats over-smoothing in high
gradient regions.

### 5.2 Why this composite loss was used

- `mse` anchors global accuracy and stable optimization scale.
- `l1` improves robustness and sharper residual fitting.
- `temp` aligns temporal evolution statistics.
- `airfoil` encodes a hard physical prior (no-slip) as a soft training term.
- `gmse` preserves fine-scale spatial structure.

The combined objective balances global correctness, temporal fidelity, boundary
physics, and local sharpness.

### 5.3 Effective weighting in this submission

With configured coefficients:

```math
\mathcal{L}
=1.0\,\mathcal{L}_{mse}
+0.1\,\mathcal{L}_{l1}
+0.5\,\mathcal{L}_{temp}
+(0.2\times 5.0)\,\mathcal{L}_{airfoil}
+0.5\,\mathcal{L}_{gmse}
```

So the effective airfoil multiplier is `1.0` relative to raw
$\mathcal{L}_{airfoil}$.

Training weights (from run logs):

- `alpha=1.0`
- `beta=0.1`
- `gamma=0.5`
- `delta=0.2`
- `airfoil_weight=5.0`
- `lambda_grad=0.5`

---

## 6) Optimization and schedule

Training protocol used for the final members:

- Optimizer: `AdamW`
- Initial LR: `2e-4`
- Weight decay: `1e-4`
- Epochs: `300`
- Batch size: `1`
- Grad clip: `1.0`
- AMP: enabled
- Scheduler: warmup + cosine annealing
- Warmup epochs: `5`

All reported VRT members have:

- model params: `18,249,359` total/trainable

---

## 7) Data split and augmentation protocol

The VRT training path in `train.py` uses:

- simulation-aware split mode
- fixed number of validation simulations (via trainer defaults)
- y-reflection augmentation applied to training data only
- validation kept on original orientation (no reflection duplication)

For listed runs, each member uses its own `(seed, split_seed)` pair, improving
ensemble diversity while preserving architecture and objective.

---

## 8) Final trained members used for submission

The final submission ensemble consists of 4 independently trained members with
distinct random seeds and split seeds for diversity.

Each member is stored with:

- `checkpoints/memberX/best.pt`
- `checkpoints/memberX/vrt_flow_stats.pt`

---

## 9) Inference-time ensembling strategy

`VRTEnsemble` performs:

1. Forward each of 4 members on original input
2. Reflect input across y-axis and forward each member again
3. Unreflect reflected predictions back to original coordinates
4. Average all 8 predictions

Additionally, a hard fallback can be applied:

- If input has very high norm but low temporal change, return persistence
  (repeat last input frame), then apply no-slip boundary projection.

Current wrapper defaults:

- `enable_reflection_tta=True`
- `enable_hard_fallback=True`

---

## 10) Run outcomes

For the four final members, the best validation values reached were:

- `seed7_split3`: best val L2 `0.6007`, best val relL2 `0.0427`
- `seed24_split31`: best val L2 `0.7058`, best val relL2 `0.0471`
- `seed20260421_split41820260`: best val L2 `0.7402`, best val relL2 `0.0496`
- `seed2026_split4182`: best val L2 `0.6654`, best val relL2 `0.0439`

These values are aggregated from the training records of the four selected members.

---

## 11) Evaluation metrics and final benchmark results

Let $\hat{u}, u \in \mathbb{R}^{B\times T\times N\times 3}$ be prediction and
ground truth, and $u^{persist}$ be persistence baseline (repeat last input
frame). For sample index $b$, timestep $t$, point $n$:

### 11.1 Primary evaluation metrics

**Relative L2 error**
```math
\mathrm{RelL2}
= \frac{1}{B}\sum_b
\frac{\|\hat{u}_b-u_b\|_2}{\|u_b\|_2+\epsilon}
```
where norms are over all flattened $(T,N,3)$ elements.

**Temporal rollout error**
```math
\mathrm{Rollout}
= \frac{1}{B}\sum_b \frac{1}{T}\sum_t
\frac{\|\hat{u}_{b,t}-u_{b,t}\|_2}{\|u_{b,t}\|_2+\epsilon}
```

**High-frequency residual L2**
```math
\bar{u}_b=\frac{1}{T}\sum_t u_{b,t},\quad
\bar{\hat{u}}_b=\frac{1}{T}\sum_t \hat{u}_{b,t}
```
```math
\mathrm{HFRes}
= \frac{1}{B}\sum_b
\frac{\|(\hat{u}_b-\bar{\hat{u}}_b)-(u_b-\bar{u}_b)\|_2}
{\|u_b-\bar{u}_b\|_2+\epsilon}
```

**Boundary condition error**
```math
\mathrm{BCE}
= \frac{1}{B}\sum_b
\sqrt{
\frac{1}{T|\mathcal{A}_b|}
\sum_{t}\sum_{n\in\mathcal{A}_b}\|\hat{u}_{b,t,n}\|_2^2 + \epsilon
}
```
where $\mathcal{A}_b$ is the airfoil/boundary index set.

### 11.2 Training-log style auxiliary metrics

**L2 (competition metric)**
```math
\mathrm{L2}
= \frac{1}{B}\sum_b \frac{1}{TN}\sum_{t,n}\|\hat{u}_{b,t,n}-u_{b,t,n}\|_2
```

**L1**
```math
\mathrm{L1}
= \frac{1}{BTN3}\sum_{b,t,n,c}|\hat{u}_{btnc}-u_{btnc}|
```

**MSE**
```math
\mathrm{MSE}
= \frac{1}{BTN3}\sum_{b,t,n,c}(\hat{u}_{btnc}-u_{btnc})^2
```

**GMSE**
```math
\mathrm{GMSE}
= \mathrm{MSE}\!\left(
(\hat{u}_{:,:,1:,:}-\hat{u}_{:,:,:-1,:}),
(u_{:,:,1:,:}-u_{:,:,:-1,:})
\right)
```

**Weighted L2 and vs\_persist**

Define point weights from persistence innovation:
```math
\Delta_{b,n}=\frac{1}{T}\sum_t\|u_{b,t,n}-u^{persist}_{b,t,n}\|_2,\quad
w_{b,n}=\frac{\Delta_{b,n}+\epsilon}{\sum_n(\Delta_{b,n}+\epsilon)}
```

Then:
```math
\mathrm{wL2}_b = \frac{1}{T}\sum_t\sum_n w_{b,n}\|\hat{u}_{b,t,n}-u_{b,t,n}\|_2
```
```math
\mathrm{pL2}_b = \frac{1}{T}\sum_t\sum_n w_{b,n}\|u^{persist}_{b,t,n}-u_{b,t,n}\|_2
```
```math
\mathrm{weighted\_l2}=\frac{1}{B}\sum_b \mathrm{wL2}_b,\quad
\mathrm{vs\_persist}=\frac{1}{B}\sum_b\frac{\mathrm{wL2}_b}{\mathrm{pL2}_b+\epsilon}
```

Final reported mean results for this submission (mean over the evaluation set):

- `relative_l2_error`: `0.016577152168689664`
- `temporal_rollout_error`: `0.016529230637325656`
- `high_frequency_residual_l2`: `0.20932431787620356`
- `boundary_condition_error`: `9.999999974752427e-07`
- `vs_persist`: `0.13580659090736766`
- `l2`: `0.30194369584710384`
- `weighted_l2`: `1.4445130011693426`
- `l1`: `0.14753634393320994`
- `mse`: `0.18546677442167814`
- `gmse`: `0.2883375277318957`

---

## 12) Final submitted model 

The submission is a **composite inference system**, not a single network:

- 4x independently trained VRT members
- per-member external normalization/stats files
- reflection TTA (2x per member)
- arithmetic mean aggregation across 8 outputs
- optional hard persistence fallback
- explicit no-slip boundary projection in model forward

The final predictor used in evaluation is the ensemble wrapper provided in this
submission package.
