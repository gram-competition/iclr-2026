# FiniteGraphV4

A two-hop directional finite-graph forecaster for the GRaM ICLR 2026
turbulence-prediction challenge. With a
learned temporal node encoder, directional second-hop neighbourhoods,
and a gated cross-direction fusion head.


## Entry point

```python
from models import FiniteGraphV4

model = FiniteGraphV4()                                    # loads weights.pt
velocity_out = model(t, pos, idcs_airfoil, velocity_in)    # (B, 5, 100k, 3)
```

## Architecture

Per non-surface centre node the model builds three directional graphs
(isotropic / upstream / downstream) with `k1` first-hop neighbours
each. For every first-hop neighbour it gathers `k2` **directional**
second-hop neighbours from a `k_pool` kNN pool (some kind of regularization to pool from larger amounts of nodes with the target to get independent on the mesh resolution - at least I thought so ;-) , so each
(centre, direction) pair carries `k1 × k2` two-hop paths that
preserve flow orientation at hop 2 as well as hop 1.

**Temporal node encoder.** The 26-channel input is first fed through a
`TemporalNodeEncoderV4` that reconstructs the 5 raw input velocity
snapshots from the delta (so delta velcoities) channels and encodes them with two Conv1d
blocks, then fuses the temporal summary with a static-feature projection
via a residual LayerNorm. The encoded latent (`hidden`) drives all
downstream message passing.

**Message passing.** Per-direction, messages flow `hop2 → hop1 → centre`
through two `MessagePassingLayerV4` blocks. Each block uses a
**centre-derived multi-head attention query** plus a **(attention, max,
variance) pool** — the variance channel makes local turbulence intensity
directly visible to the post-MLP - or lets hope so.

**Gated cross-direction fusion.** Iso / up / down latents are combined
by `GatedDirectionalFusion`: a centre-conditioned softmax produces three
branch weights, and the fused vector concatenates the weighted summary,
the inter-branch spread (variance) and the inter-branch maxima before
being decoded by the MLP head into `5 × 3 = 15` outputs.

Edge features are 39-channel and include:
- `rel_pos` / `rel_dist`;
- **Fourier positional encodings** (3 wavenumbers × 3 axes × sin/cos),
  non-dimensionalised by the per-chunk extent `L_ref`, to break the
  MLP's classical spectral bias on raw Cartesian inputs;
- wall-distance (neighbour + delta);
- flow-aligned / perpendicular components of the displacement;
- neighbour-vs-centre speed, velocity, vorticity, Q-criterion,
  strain magnitude and acceleration differences.

Outputs are five normalised residuals `r_k = v[5+k] − v_t4`. The package
adds `v_t4` back and zeros velocity at the airfoil indices to enforce
no-slip exactly.

Input features (26 channels): 4 velocity deltas against `v[0]` (12 ch),
`v_t4` (3), wall distance (1), signed wall distance (1), normalised
position (3), airfoil indicator (1), and at `t4` vorticity (3),
Q-criterion (1) and strain magnitude (1). Vorticity, Q and strain are
computed from the local kNN-LSQ velocity-gradient tensor at `t4`.

## Training

- **Split**: simulation-level, stratified by geometry prefix.
- **Loss**: weighted L1 + L2 on the normalised residual, with a mild
  near-wall upweight. Differentiable soft-divergence and Navier-Stokes
  momentum penalties with air properties (ρ = 1.225, μ = 1.81e-5) are
  available and were mixed in at low weight (~0.01) during training.
- **Optimiser**: AdamW with linear warmup + cosine decay.
- **Targets**: residuals `r_k = v[5+k] − v[4]`, k = 0..4.
- **Normalisation**: per-channel mean/std fit on the training split
  (the airfoil-indicator channel is kept unscaled). Stats travel with
  the checkpoint inside `weights.pt`.

Default model config (matches the training repo):

```python
FiniteGraphModelV4(
    in_ch=26, hidden=192, latent=192,
    k1=24, k2=12, n_attn_heads=4,
    out_heads=5, out_ch_per_head=3,
    shared_weights=False, dropout=0.05,
    temporal_hidden=96,
)
```

The checkpoint's own `model_config` is used at load time, so alternative
hyperparameters are picked up automatically.

## Dependencies

- PyTorch
- NumPy
- SciPy (for `cKDTree`)

No custom CUDA kernels; runs on CPU or GPU.

## File layout

```
models/finite_graph_v4/
├── __init__.py         — exposes FiniteGraphV4
├── model.py            — the competition-facing wrapper
├── net.py              — FiniteGraphModelV4 + inference wrapper
├── graph_utils.py      — kNN pool + directional graph construction
├── features.py         — (N, 26) feature assembly
├── weights.pt          — **user-supplied** checkpoint (state_dict + config + stats)
└── README.md           — this file
```
