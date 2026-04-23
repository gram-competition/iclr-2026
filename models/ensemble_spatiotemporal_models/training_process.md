# Ensemble SpatioTemporal GNN: 


The submitted model is named **Ensemble SpatioTemporal GNN** because it combines:

- **SpatioTemporal** modeling: spatial interaction on the full point cloud plus
  temporal forecasting of future states.
- **GNN** backbone: message passing over a k-nearest-neighbor graph built from
  3D point positions.
- **Ensemble** inference: arithmetic mean of two complementary predictors:
  a baseline spatiotemporal GNN and a physics-feature-augmented variant.

Design objective:

- keep native full-resolution point-cloud fidelity (`N=100000`),
- inject geometric structure explicitly (airfoil mask, and in one branch
  distance/direction-to-surface),
- preserve stable residual forecasting behavior with strict no-slip boundary
  projection.

---

## Block diagram (end-to-end)

```text
Inputs
  t (B,10), pos (B,N,3), idcs_airfoil, velocity_in (B,5,N,3)
        |
        v
[Input sanitization]
  - clamp boundary indices into [0, N-1]
  - device alignment for all tensors
        |
        v
[Branch A: SpatioTemporalGNN] --------------------\
  - optional Fourier position encoding             |
  - node feature build: [pos_feat | vel_flat | mask]
  - full-resolution kNN graph (k=16)              |
  - spatial backbone (GAT / MeshGraphNet / GT)    |
  - temporal attention head                        |
  - decoder + residual add + no-slip              |
                                                   +--> [MeanOutputEnsemble] --> prediction
[Branch B: SpatioTemporalGNNPhysFeat] -----------/
  - same as Branch A, plus:
  - distance-to-airfoil + direction-to-airfoil
  - augmented node feature build

Optional post-ensemble safeguard:
  [Hard persistence fallback]
    if high-norm + low-dynamics condition is met:
      output = repeated last frame (with no-slip)
```

---

## 1) Problem setting and tensor interface

The model is a forecaster on 3D point clouds with:

- input horizon `T_in = 5`,
- output horizon `T_out = 5`,
- point count `N = 100000`,
- vector channels `C = 3`.

Expected callable signature:

```python
model(
    t: Tensor[B, 10],
    pos: Tensor[B, N, 3],
    idcs_airfoil: list[Tensor[Mi]],
    velocity_in: Tensor[B, 5, N, 3],
) -> Tensor[B, 5, N, 3]
```

The forward path enforces no-slip by zeroing velocity on `idcs_airfoil`.

---

## 2) Top-level model composition

The final predictor is a two-member mean ensemble:

1. **Base branch**: `SpatioTemporalGNN`
2. **PhysFeat branch**: `SpatioTemporalGNNPhysFeat`

Both members are loaded from pretrained checkpoints and kept in eval mode.
Their outputs are combined as:

```text
u_hat = 0.5 * (u_hat_base + u_hat_physfeat)
```

An optional hard fallback (enabled by default in submission wrapper) can replace
the ensemble prediction with persistence for pathological high-magnitude,
low-dynamics inputs.

---

## 3) Shared computational pipeline (both branches)

### 3.1 Full-resolution operation

The model does not subsample points at inference. All operators run on the full
100k-point cloud, preserving geometric detail and boundary localization.

### 3.2 Airfoil mask construction

For each sample, a binary boundary mask is built:

```text
mask[n] = 1 if n in idcs_airfoil else 0
```

This mask is concatenated to node features and used again for hard projection.

### 3.3 kNN graph construction

A spatial graph is constructed from positions using `k=16` nearest neighbors.
The graph returns:

- neighbor indices,
- relative neighbor positions,
- neighbor distances.

These are consumed by the selected spatial backbone.

### 3.4 Node feature encoding

Base branch node feature:

```text
x_node = [pos_encoded, flatten(velocity_in over time), airfoil_mask]
```

PhysFeat branch node feature:

```text
x_node = [pos_encoded, flatten(velocity_in), airfoil_mask, dist_to_surface, dir_to_surface]
```

Both are projected by an MLP + LayerNorm to hidden dimension.

### 3.5 Spatial backbone pass

A configurable graph backbone transforms node states using neighborhood message
passing. Backbones supported by the package:

- GAT-style local attention,
- MeshGraphNet-style edge/node updates,
- Graph Transformer with edge-conditioned value modulation.

### 3.6 Temporal forecasting head

After spatial encoding, each node feature is projected to `T_out` temporal
tokens and refined by stacked multi-head self-attention over the output-time
dimension (per node). This yields node-wise future latent trajectories.

### 3.7 Decoder and residual projection

A pointwise decoder maps each future latent token to a 3D velocity delta.
Predicted deltas are added to the last observed velocity frame (residual
forecasting).

### 3.8 No-slip projection

Final output is projected to satisfy boundary condition:

```text
u_hat[:, :, idcs_airfoil, :] = 0
```

---

## 4) Branch-specific details

### 4.1 Base branch: SpatioTemporalGNN

This branch is the general dynamics model and serves as the geometric-temporal
backbone of the ensemble.

#### Base branch diagram

```text
pos (N,3) ---------------------------> [FourierPosEnc or identity] ----\
                                                                         \
velocity_in (T_in,N,3) -> [permute+flatten over time] -------------------+--> [concat] --> [Node Encoder MLP+LN] --> x0 (N,D)
                                                                          /
airfoil_idx -----------------------> [binary boundary mask (N,1)] -------/

pos (N,3) -----------------------> [kNN graph, k=16] --> neighbors, rel_pos, dists
                                                         |
                                                         v
x0 (N,D) ------------------------------------------> [Spatial Backbone] --> xs (N,D)
                                                         (GAT / MGN / GraphTransformer)
                                                                               |
                                                                               v
xs (N,D) ------------------------------------------> [TemporalAttentionHead] --> z (N,T_out,D)
                                                                               |
                                                                               v
z -----------------------------------------------> [Decoder MLP] --> delta_u (N,T_out,3)
                                                                               |
last input frame u_{T_in-1} (N,3) --------------------------------------------+
                                                                               v
residual add: u_hat_local = delta_u + u_{T_in-1}
                                                                               |
                                                                               v
permute -> (T_out,N,3) -> no-slip projection on airfoil points -> output branch A
```

#### Base branch formulation

For node `n`, let:

- `p_n in R^3` be position,
- `v_n in R^{3*T_in}` be flattened input history,
- `m_n in {0,1}` be boundary mask.

If Fourier encoding is enabled:

```math
\phi(p_n)=\left[p_n,\ \sin(2^0p_n),\cos(2^0p_n),\ldots,\sin(2^{F-1}p_n),\cos(2^{F-1}p_n)\right]
```

Node feature:

```math
x_n^{(0)}=\mathrm{MLP}_{enc}\Big([\phi(p_n),\ v_n,\ m_n]\Big)\in\mathbb{R}^{D}
```

Spatial processing over kNN graph `G=(V,E)`:

```math
x^{(s)}=\mathcal{G}_{spatial}\!\left(x^{(0)},\ \text{neighbors},\ \Delta p,\ d\right)
```

Temporal head:

```math
z_n=\mathcal{T}\!\left(x_n^{(s)}\right)\in\mathbb{R}^{T_{out}\times D}
```

Decoder + residual:

```math
\Delta u_n=\mathrm{MLP}_{dec}(z_n),\qquad
\hat{u}_n(t)=\Delta u_n(t)+u_n(T_{in}-1)
```

No-slip projection:

```math
\hat{u}_n(t)=0\ \ \text{if}\ \ n\in\mathcal{I}_{airfoil}
```

### 4.2 PhysFeat branch: SpatioTemporalGNNPhysFeat

This branch shares the same macro-architecture as the base branch, but augments
node features with continuous wall-proximity geometry.

#### PhysFeat branch diagram

```text
pos (N,3) -----------------------------------------------> [FourierPosEnc or identity] --\
                                                                                            \
velocity_in (T_in,N,3) -> [permute+flatten] ------------------------------------------------+--> [concat] --> [Node Encoder] --> x0' (N,D)
                                                                                             /
airfoil_idx -----------------> [mask (N,1)] -------------------------------------------------/
                                                                                             \
                                                                                              +--> [Surface geometry module]
                                                                                                   - nearest airfoil point
                                                                                                   - distance d_surf (N,1)
                                                                                                   - unit direction r_hat (N,3)
                                                                                                   - chunked cdist for memory safety

pos -> [kNN graph] -> neighbors, rel_pos, dists
x0' + graph -------------------------------------> [Spatial Backbone] -> [TemporalAttentionHead]
                                               -> [Decoder + residual + no-slip] -> output branch B
```

#### PhysFeat geometry construction

For each point `p_n`, nearest boundary point:

```math
q_n^\*=\arg\min_{q\in\mathcal{S}_{airfoil}}\|p_n-q\|_2
```

Distance and direction features:

```math
d_n=\|p_n-q_n^\*\|_2,\qquad
r_n=\frac{q_n^\*-p_n}{\max(d_n,\epsilon)}
```

The module outputs `g_n=[d_n,\ r_n] in R^4`.

PhysFeat node embedding:

```math
x_n^{(0,phys)}=\mathrm{MLP}_{enc}\Big([\phi(p_n),\ v_n,\ m_n,\ g_n]\Big)
```

All downstream stages (spatial backbone, temporal head, decoder, residual add,
no-slip projection) mirror the base branch.

#### Why this branch differs operationally

- The base branch infers boundary influence implicitly from connectivity and
  mask.
- The physfeat branch introduces an explicit continuous geometric signal that
  varies smoothly away from the wall.
- Averaging both branches balances global flow-context extrapolation and
  boundary-localized sensitivity.

---

## 5) Temporal attention head internals

Given node embedding `h_n in R^D`, the temporal head:

1. projects `h_n` to `T_out * D_t`,
2. reshapes to a sequence of `T_out` tokens,
3. adds learnable temporal positional embedding,
4. applies repeated MHSA + FFN + residual + LayerNorm blocks.

For each attention layer:

```text
z = LN(z + MHSA(z))
z = LN(z + FFN(z))
```

This explicitly models dependencies across future output timesteps for each
spatial point.

---

## 6) Spatial backbone options (brief technical summary)

### GAT backbone

- computes attention over each node's k-neighborhood,
- can inject edge-conditioned bias into attention logits,
- uses residual + FFN post-attention updates.

### MeshGraphNet-style backbone

- edge update from `(center_node, neighbor_node, edge_state)`,
- mean aggregation of edge messages to node updates,
- residual normalization at edge and node stages.

### Graph Transformer backbone

- local transformer attention on kNN neighborhoods,
- optional edge-conditioned value modulation,
- residual + FFN normalization structure.

All variants maintain shape compatibility with the same temporal head and decoder.

### Backbone internals (deeper view)

Each backbone consumes:

- node states `x in R^{N x D}`,
- neighbor table `nbr in Z^{N x k}`,
- relative coordinates `rel_pos in R^{N x k x 3}`,
- scalar distances `dists in R^{N x k}`.

#### GAT path

Edge features are encoded first, then attention is computed over each local
neighborhood:

```math
\alpha_{n,j,h}=\mathrm{softmax}_{j\in\mathcal{N}(n)}
\left(
\frac{\langle W_q^h x_n,\ W_k^h x_j\rangle}{\sqrt{d_h}} + b_{edge}(n,j,h)
\right)
```

```math
\tilde{x}_n=\sum_{j\in\mathcal{N}(n)}\alpha_{n,j}W_vx_j,\qquad
x_n\leftarrow \mathrm{LN}(x_n+\tilde{x}_n),\quad
x_n\leftarrow \mathrm{LN}(x_n+\mathrm{FFN}(x_n))
```

#### MeshGraphNet path

Edge state update and node update are both residual:

```math
e_{n,j}\leftarrow \mathrm{LN}\!\left(e_{n,j}+\mathrm{MLP}_e([x_n,x_j,e_{n,j}])\right)
```

```math
m_n=\frac{1}{k}\sum_{j\in\mathcal{N}(n)}e_{n,j},\qquad
x_n\leftarrow \mathrm{LN}\!\left(x_n+\mathrm{MLP}_x([x_n,m_n])\right)
```

#### Graph Transformer path

Similar local attention, but edge features modulate value projections:

```math
v_{n,j}^{(h)} = W_v^h x_j + W_e^h e_{n,j}
```

```math
\tilde{x}_n=\sum_{j\in\mathcal{N}(n)}\alpha_{n,j}v_{n,j},\quad
x_n\leftarrow \mathrm{LN}(x_n+\tilde{x}_n)\rightarrow \mathrm{LN}(x_n+\mathrm{FFN}(x_n))
```

---

## 7) Checkpoint and deployment behavior

The ensemble wrapper:

- loads two checkpoints (base and physfeat),
- freezes member parameters,
- runs both members in eval mode,
- averages outputs,
- optionally applies hard persistence fallback.

Device handling is runtime-aware: the wrapper resolves the current parameter
device and aligns inputs accordingly, preventing mixed-device failures.

Boundary index sanitization is applied before indexing operations to avoid
out-of-range issues.

---

## 8) Hard persistence fallback behavior

Fallback is triggered when:

- input norm is above a threshold (`in_norm_threshold`), and
- mean frame-to-frame input change is below a threshold (`in_step_mean_threshold`).

When triggered:

```text
u_hat[t] = velocity_in[last]  for all future t
```

followed by no-slip boundary projection.

This safeguard is meant for rare extreme regimes where conservative persistence
is more stable than extrapolation.

---

## 9) Final submitted model (clear statement)

The submitted predictor is a **composite system**, not a single branch:

- two pretrained full-resolution spatiotemporal GNN members,
- arithmetic mean ensembling,
- runtime device alignment,
- optional hard persistence fallback,
- strict no-slip output projection.

This design balances general graph dynamics (base branch) with explicit
surface-geometry conditioning (physfeat branch) while preserving robust
inference behavior under operational edge cases.
