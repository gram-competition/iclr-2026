# Reviewer Assessment: GRaM Challenge Submission
## TransolverResidual — Transient Airflow Prediction on 3D Point Clouds

**Reviewer perspective:** Workshop competition technical reviewer  
**Date:** April 11, 2026

---

## Summary Verdict

This is a competent, well-documented submission that demonstrates a genuine understanding of the problem. The polynomial residual framing is principled, the feature engineering is thoughtful, and the reported numbers are credible (rel-L2 ~0.07 vs. polynomial baseline of 0.2–1.3). However, several architectural choices are compromises rather than design decisions, and the submission has at least one known correctness bug that silently degrades inference. Below I detail both.

---

## Strong Points

### 1. Polynomial Residual Baseline — Correctly Motivated and Correctly Implemented

The core insight — that the polynomial baseline degrades super-linearly with horizon while the network correction stays flat — is the most important result in the report. The per-timestep table (t5: 0.069, t9: 0.084) confirms the model has learned the turbulent component, not just hedged toward the low-frequency prediction. This is non-trivial for a transient problem and the flat error curve across five future horizons is strong evidence of genuine dynamics learning.

The zero-initialized decoder is exactly right: the model starts as a pure polynomial baseline and the network learns corrections from zero. This is a clean inductive bias that makes training stable and graceful degradation guaranteed.

### 2. Feature Engineering

The `poly_residual` feature (input velocity minus polynomial fit at input times) is an excellent explicit turbulence proxy. The model is told, per point, where the polynomial is already failing in the input window — which directly predicts where it will fail in the output window. This is the kind of domain knowledge that separates informed submissions from blind ones.

The `dist_to_airfoil` + `upstream_dist` pair compactly encodes geometry. `upstream_dist` in particular is well-motivated: it encodes the asymmetry between upstream (clean flow) and downstream (wake/interference) relative to the surface.

### 3. Y-Flip Augmentation

The effect size here is dramatic and diagnostic: layer-0 slice entropy dropped from ~100% to 10.6% after augmentation. This means without augmentation, the model could not break y-symmetry and the slice assignments were uninformative. The fact that a single augmentation type had this much effect tells you the training set is y-biased (almost certainly the geometries are all pitched consistently), and the network was exploiting that bias rather than learning geometry.

### 4. Hard No-Slip Enforcement

Exact, free, correct. Every submission should do this. The fact that it also implicitly removes the model's need to learn to predict zeros at surface points focuses capacity on the interior.

### 5. Problem Scale is Correctly Calibrated

2.8M parameters for 905 training samples is the right order of magnitude. The submission avoids the common mistake of scaling up the model and overfitting instead of generalizing.

---

## Weak Points

### 1. The Physics-Attention Mechanism is Not Actually Working in the Middle Layers

This is the most serious architectural problem. The U-shaped entropy pattern (layer 0: 10.6%, layer 4: 94.2%, layer 7: 10.9%) is not a quirk — it is evidence that the model has learned to use only 2 of 8 Transolver blocks as the paper intends. The 6 middle blocks are largely doing global mean pooling (at 94.2% entropy, each slice weight is approximately 1/32 = 3.1% — a uniform distribution).

The submission acknowledges this. What it does not say clearly enough is: the model is paying the O(NMC) linear cost for 8 blocks but getting O(NMC) Physics-Attention behavior in only 2. This is a 4× waste of the dominant computational cost at inference time.

More importantly: the mode collapse to ~10 active slices (out of 32) in the functional layers means the model has fewer effective routing bins than designed. This should have been diagnosed earlier and addressed, either by reducing M to the actual effective count (M≈10) or by adding a slice diversity loss.

**Root cause:** The problem is the training signal. Residual connections mean the slice-assignment network sees a signal dominated by the skip path, not the attention-corrected path. The slice assignment learns from a nearly-unchanged representation at each intermediate layer, giving it no reason to maintain diversity. The original Transolver paper trains without a polynomial baseline and with a direct velocity prediction target — the residual learning strategy interferes with the attention routing dynamics.

### 2. Temporal Structure is Entirely Hand-Engineered

The 5 input timesteps are processed by flattening them into a 79-channel per-point feature vector. There is no temporal attention, no RNN, no explicit temporal inductive bias in the architecture. The time structure is instead encoded via hand-crafted features: `temporal_mean`, `temporal_std`, `temporal_deltas` (4 differences), `t_values`.

This works, but it is brittle. A 2-layer temporal self-attention over T=5 time tokens (per point, before spatial attention) would:
1. Let the model discover temporal patterns rather than requiring the user to enumerate them.
2. Give explicit temporal ordering information (which hand-engineering of deltas partially provides but incompletely).
3. Scale if the problem is extended to more input timesteps.

The fact that `temporal_deltas` improved validation loss by ~13% when added manually is direct evidence that the architecture is temporally underspecified and the gap is being patched through feature engineering.

### 3. Local Context is a Workaround, Not a Design

The `local_nbr_mean` feature (mean velocity of 8 nearest neighbours) is added because the Transolver, operating through global slice assignments, lacks local context for near-surface points. The 13% improvement from adding this feature is a red flag: it tells you the model without it was missing information that is spatially local.

The correct fix is an explicit local aggregation step (one or two GNN message-passing layers, or equivalently a local attention kernel with a small radius) before the global Transolver attention. Instead, the solution chosen is to precompute k-NN indices and pass them as features. This:
1. Creates an external dependency (the kNN cache) that must be managed at inference time.
2. Limits local context to one level of neighbours (k=8) with no learned aggregation weight.
3. Introduces the inference bug described below.

A learnable 1-hop message-passing step over the same k=8 graph would be strictly more expressive and would not require external caches.

### 4. k-NN Inference Dependency is Handled, but Carries a Latency Cost

The REPORT.md flags the k-NN inference dependency as a pending bug, but the code as submitted already fixes it. `_ensure_geometry_cache()` is called at the top of `forward()` whenever `knn_feats is None`, computes the k-NN via `scipy.cKDTree`, and caches the result by geometry fingerprint. Repeated calls on the same geometry (e.g. different time windows from the same simulation) hit the cache and are instant.

What remains is a **first-call latency penalty**: for a novel geometry not yet in the cache, the model must run a full cKDTree query over 100k points before it can produce output. This is not a correctness problem, but it is an inference-time surprise the evaluator may not expect. The fingerprint-based cache also has a theoretical hash-collision risk on 5 scattered position values, though in practice collisions across distinct CFD meshes are extremely unlikely.

### 5. Validation Strategy Does Not Test Geometric Generalization

The 90/10 random train/val split almost certainly mixes time windows from the same simulation across train and val. A sample from simulation `1023_16-2` (time window 2) in the val set will have the same `pos` and `idcs_airfoil` as samples `1023_16-0`, `1023_16-1`, etc. in the training set. The model has seen the geometry.

The held-out test set, by contrast, will contain entirely new geometries. The validation loss of 0.066 may be optimistic because the model has partially memorized per-geometry distance features. A geometry-held-out validation (withhold all 5 time windows from a set of simulation IDs) would give an honest estimate of test performance.

This matters most for the distance features: `dist_to_airfoil` and `upstream_dist` encode the per-geometry structure. The model has seen these exact distance feature arrays for all training geometries. On novel test geometries it must generalize from the distance features alone, without the benefit of having seen those exact arrays during training.

### 6. Geometry Encoding is a Single Scalar

The entire geometry of 1–3 F1 airfoils, with varying positions, chord lengths, and pitch angles, is encoded into two scalars per point: distance to the nearest surface point and x-offset to the nearest surface point. This discards:
- Which element a nearby surface point belongs to (element 1 vs. 2 vs. 3)
- The angle/orientation of the nearest surface
- Whether a point is in an inter-element gap (high-velocity accelerated flow) vs. open wake
- How many elements are present in this geometry

For a competition specifically about geometry-grounded representation, this is a notable gap. A PointNet over the airfoil surface points, outputting a per-point geometry embedding that is then concatenated to the feature vector, would provide geometry conditioning that the current approach cannot represent.

---

## What a Proper Architecture Would Look Like

The submission's three core problems — hand-engineered temporal features, k-NN feature hacks for local context, and Transolver slice collapse — all have the same root cause: the architecture is a **per-point MLP with a global context bolt-on**. There is no mechanism that natively understands sequences in time, local spatial neighbourhoods, or the interaction between the two. The recommended fixes are not patches to the existing design; they require replacing the encoder stack entirely.

### The Right Decomposition: Factorized Spatiotemporal Processing

The input is a spatiotemporal field: N points × T=5 timesteps × 3 velocity components. The output is N points × T=5 future timesteps × 3 components. A well-designed architecture should respect this structure explicitly, not flatten it into a per-point feature vector.

The correct decomposition is:

```
Input: (B, T_in, N, 3)

1. Temporal encoding:   (B, T_in, N, 3) → (B, N, D)
   — process the T=5 sequence per point, produce a temporal summary per point

2. Spatial encoding:    (B, N, D) → (B, N, D)
   — exchange information across space at multiple scales (local + global)

3. Temporal decoding:   (B, N, D) → (B, T_out, N, 3)
   — project the per-point representation into T_out future predictions
```

Each stage has a single well-defined responsibility. Hand-engineered temporal features (`temporal_mean`, `temporal_std`, `temporal_deltas`, `poly_residual`) exist precisely because step 1 is missing. The k-NN feature hack exists because step 2's local scale is missing.

---

### Stage 1: Temporal Encoding — Per-Point Temporal Transformer

Each point has a trajectory of T=5 velocity snapshots. The right model for a short sequence with no natural ordering ambiguity is a **small temporal transformer** applied independently to every point:

```
Input at point i:  [ (v_{t0}, t0), (v_{t1}, t1), ..., (v_{t4}, t4) ]  — T tokens of dim 3+1=4
Output at point i: one summary vector of dim D_t
```

This processes each token as `(velocity, time_value)` and produces a fixed-size temporal embedding per point. It naturally learns:
- the velocity trend (replacing `temporal_mean`, `temporal_deltas`)
- the turbulence intensity (replacing `poly_residual`)
- the temporal phase of the observation window (replacing `t_values`)

None of these need to be hand-specified. The transformer discovers the temporal basis that is most useful for predicting future states, which may not be the polynomial basis the current code hard-codes.

Memory: processing (B·N, T, D_t) requires holding a (B·N, T, T) attention matrix, which at B=1, N=100k, T=5 is 100k × 5 × 5 floats — 10MB, negligible.

---

### Stage 2: Spatial Encoding — Hierarchical Message Passing + Global Attention

The spatial stage needs two scales: local (sub-chord neighbourhoods) and global (wake, freestream, inter-element interactions). These are qualitatively different and should be handled by qualitatively different mechanisms.

**Local scale: learned message passing on the k-NN graph**

A proper **MPNN** (message-passing neural network) aggregation step over the k-NN graph:

```
m_{i} = Σ_{j ∈ N(i)}  MLP_edge( x_i, x_j, pos_i - pos_j )
x_i   ← MLP_node( x_i, m_i )
```

The edge MLP takes the relative position `pos_i - pos_j` as an explicit input, not as a feature baked into the source token. This gives the network proper equivariance information: it knows not just *what* a neighbour's velocity is, but *where* that neighbour is relative to the current point. This is geometrically meaningful — a neighbour upstream at distance δ carries different information than a neighbour at the same distance downstream.

The current k-NN feature (mean of neighbour velocities, with no position-relative weighting) is strictly weaker: it discards the relative positions entirely.

Two or three MPNN layers are sufficient. This is the correct local structure.

**Global scale: Perceiver-style cross-attention (not slice assignment)**

For global context, the Transolver slice mechanism should be replaced by a **Perceiver IO** cross-attention block:

```
Compress:  (B, N, D)  →  (B, L, D)    via cross-attention: L latents attend over N points
Process:   (B, L, D)  →  (B, L, D)    via standard self-attention among L latents
Broadcast: (B, L, D)  →  (B, N, D)    via cross-attention: N points attend over L latents
```

With L=256 and N=100k, the compress and broadcast steps are O(N·L·D) — same complexity as Transolver — but the mechanism is well understood and does not have a slice collapse failure mode. The L latent tokens are unconstrained: they learn to represent whatever global state is most useful without being forced into a softmax-weighted sum of point features.

Critically, this is not a workaround — it is strictly more expressive than the Transolver slice mechanism. The Transolver slice assignment is a special case of Perceiver cross-attention where the attention weights are constrained to sum to 1 per slice (softmax). Removing that constraint gives the latent tokens the freedom to attend selectively or broadly, per head, which is what a standard transformer does.

---

### Stage 3: Spatiotemporal Coupling — What is Actually Missing

Stages 1 and 2 as described are still factorized: temporal first, spatial second. This is not physically correct. Fluid dynamics couples space and time through **advection**: a vortex at position x at time t will arrive at a different position x' = x + u·Δt at time t+Δt. A factorized model that processes time independently per point cannot represent this: the vortex that arrives at point i at time t5 originated upstream at t4, but the per-point temporal encoder at point i only sees point i's own history.

The correct structure is a **spatiotemporal attention** layer that attends simultaneously over (N points × T timesteps) tokens:

```
Input:  (B, N × T_in, D)   — all N points at all T_in timesteps as a flat token sequence
Output: (B, N × T_in, D)   — each token updated in context of all other space-time tokens
```

At N=100k and T=5, this is 500k tokens — too expensive for direct self-attention. The Perceiver approach handles this: compress (N × T_in) to L latents, self-attend among latents, broadcast back. The L latents in the Perceiver **jointly** represent spatial and temporal context, which is what factorized processing cannot do.

This is not a theoretical nicety. It is the mechanism that lets the model answer: "the high turbulence I see at (x, t=4) — is it a vortex that originated upstream and will arrive at (x+u·Δt) at t=5, or is it local surface-generated turbulence that will dissipate?" A factorized model cannot distinguish these cases. A spatiotemporal Perceiver can, because the L latents learn to track coherent fluid structures across space and time jointly.

---

### Full Architecture Sketch

```
Input: velocity_in (B, T_in, N, 3), pos (B, N, 3), t (B, T_in + T_out)

1. Per-point temporal tokens:
   tokens_t  = TemporalTransformer( velocity_in, t[:, :T_in] )  → (B, N, D)

2. Local spatial exchange:
   tokens_l  = MPNN( tokens_t, pos, k=16, n_layers=2 )          → (B, N, D)

3. Spatiotemporal global context:
   tokens_st = Perceiver(
       input  = concat( tokens_l over T_in ) → (B, N*T_in, D),
       latents = L=256,
       n_self_attn_layers = 6,
   )                                                              → (B, N, D)

4. Polynomial baseline (unchanged):
   poly_pred = poly_extrapolate( velocity_in, t )                → (B, T_out, N, 3)

5. Correction decoder:
   correction = MLP( tokens_st )                                  → (B, T_out*3, N)

6. Output = poly_pred + correction, zeroed at idcs_airfoil
```

Parameter estimate: ~8–12M at D=256, L=256. Appropriate for 905 training samples with augmentation.

---

### Why Not Stay with Transolver?

The Transolver was a reasonable starting point given the time constraint, and the paper's results on steady-state PDE benchmarks are real. But this task has three properties that the Transolver was not designed for:

1. **Temporal sequences**, not snapshots. Transolver processes a single spatial field. Extending it to sequences by flattening time into features is the same mistake as using a CNN for text by treating characters as channels.

2. **Irregular 100k-point meshes with multi-scale structure**. The Transolver's slices collapse to ~10 active groups because the mechanism has no prior toward spatial coherence — it discovers spatial groupings by accident via gradient descent. A hierarchical architecture with explicit local aggregation imposes the right inductive bias directly.

3. **Turbulence is spatiotemporally correlated, not just spatially correlated**. A vortex is not a property of a point; it is a property of a region of space evolving over time. An architecture that cannot represent cross-space-time dependencies cannot properly model vortex dynamics. Factorized temporal + spatial processing is insufficient; spatiotemporal joint processing (Perceiver) is necessary.

---

## Numbers in Context

| Metric | Value | Commentary |
|--------|-------|------------|
| Val rel-L2 (mean) | 0.066 | Honest, but split may leak geometry |
| Val rel-L2 (t5) | 0.069 | First future step, should be easiest |
| Val rel-L2 (t9) | 0.084 | 5th future step, growth is modest (+22%) |
| Poly baseline (t9) | 1.277 | The model beats this by 15× |
| Parameters | 2.8M | Correct scale for 905 samples |
| Effective slices | ~10 | Paid for 32, got ~10 |
| Functional Transolver layers | 2 of 8 | Paid for 8, used 2 |

The model's t5→t9 error growth (+22%) is encouraging and suggests the model is extrapolating reasonably. The polynomial's t5→t9 growth is +514%, which means the model is doing qualitatively different physics from the baseline.

---

## Final Assessment

**What works:** The problem decomposition (residual on polynomial) is the right idea and is executed correctly. Feature engineering is more sophisticated than typical baseline submissions. The implementation is clean, well-documented, and avoids obvious pitfalls (no data leakage, hard no-slip enforcement, appropriate model scale).

**What doesn't work as intended:** The Transolver mechanism is underutilized (2 of 8 layers functional, ~10 of 32 slices active). The temporal structure is hand-engineered rather than learned. The local context problem is patched rather than solved, introducing an inference bug. The validation split does not measure generalization to novel geometries.

**What should have been done:** A temporal encoder, one GNN aggregation layer, entropy regularization on slice assignments, and a geometry-held-out validation split. None of these requires a complete rewrite — they are additive improvements to the existing architecture. The largest single-impact change would have been the geometry-held-out validation split, not because it improves the model, but because it would have revealed earlier whether the current feature set generalizes to novel geometries or not. That information would have driven all subsequent decisions.

The submission is above the median for a solo participant on a 14-day timeline. It is not competitive with a well-staffed team that had time for ablations and proper geometry encoding.
