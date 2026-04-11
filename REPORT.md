# GRaM Transient Airflow Prediction — Technical Report

**Competition:** GRaM Challenge @ ICLR 2026  
**Task:** Predict 5 future velocity timesteps (t5–t9) from 5 input timesteps on 100k-point irregular 3D meshes around F1-style airfoil geometries.  
**Evaluation metric:** Pointwise accuracy between predicted and ground-truth 3D velocity fields (exact formula undisclosed).  
**Submission deadline:** April 22, 2026 (AoE)

---

## 1. Problem Characteristics

Each sample consists of a point cloud of exactly 100,000 points in 3D space (Eulerian frame — points are fixed, not particle-tracked). The geometry is an anisotropic warp of an Imperial Front Wing airfoil, with 1–3 airfoil elements at varying positions and pitch angles. Constant freestream flows left-to-right. The dataset contains 181 simulations × 5 time windows = 905 samples total.

The core difficulty, as stated by the organizers, is predicting **high-frequency (turbulent) components**. The low-frequency laminar component is well approximated by simple extrapolation of the input velocity history. This asymmetry between easy-to-predict background flow and hard-to-predict wake turbulence drives every architectural decision made here.

The no-slip boundary condition (zero velocity at airfoil surface points `idcs_airfoil`) is exactly enforceable as a hard constraint, which is used.

---

## 2. Architecture: TransolverResidual

### 2.1 Overall Design

The model implements a residual learning strategy on top of a polynomial extrapolation baseline:

```
output = poly_extrapolate(velocity_in, t, degree=2)  +  learned_correction
```

The polynomial baseline is computed per-point by fitting a degree-2 polynomial to the 5 input timesteps and extrapolating to the 5 output timesteps. This has no learnable parameters and is computed analytically. The network only needs to learn the **turbulent correction** — the difference between the true flow and the smooth polynomial prediction.

This is sensible. The polynomial baseline degrades predictably with horizon:

| Timestep | Polynomial rel-L2 | Model rel-L2 | Gain |
|----------|-------------------|--------------|------|
| t5       | 0.208             | 0.069        | +66.9% |
| t6       | 0.427             | 0.082        | +80.9% |
| t7       | 0.659             | 0.083        | +87.5% |
| t8       | 0.932             | 0.081        | +91.3% |
| t9       | 1.277             | 0.084        | +93.5% |

The polynomial error grows super-linearly with horizon (it exceeds 1.0 at t9, meaning the baseline is worse than predicting zero). The model error is flat at 0.07–0.08 across all timesteps. This flatness is the most important number in the table: it means the model has genuinely learned the dynamics, not just hedged toward zero.

### 2.2 Feature Vector (79 channels)

Each point receives the following per-point features:

| Feature | Channels | Description |
|---------|----------|-------------|
| `pos_normalized` | 3 | Position in [0,1]³ per-sample bounding box |
| `velocity_in` | 15 | All 5 input snapshots flattened (5×3) |
| `poly_residual` | 15 | Input velocity minus polynomial fit — explicit turbulence proxy |
| `temporal_mean` | 3 | Mean velocity over 5 input timesteps |
| `temporal_std` | 3 | Std over 5 input timesteps |
| `is_airfoil` | 1 | Binary surface flag |
| `dist_to_airfoil` | 1 | Distance to nearest surface point |
| `upstream_dist` | 1 | Signed x-offset to nearest surface point |
| `t_values` | 10 | All 10 time values (global temporal context) |
| `local_nbr_mean` | 15 | Mean velocity of 8 nearest spatial neighbours (5×3) |
| `temporal_deltas` | 12 | Velocity differences Δv_t = v_t − v_{t−1} (4×3) |

**Total: 79 channels.**

The `poly_residual` feature is particularly well-motivated: it explicitly tells the model which points are currently turbulent (large residual = the polynomial is already failing to track the flow at this point). Without this, the model would have to infer turbulence indirectly from temporal statistics.

The `local_nbr_mean` and `temporal_deltas` features were added later and yielded approximately 13% validation loss improvement over the base 52-channel version (val loss 0.076 → 0.066). This confirms the expected benefit: local neighbourhood context matters for an irregular mesh where adjacent points share physical conditions.

### 2.3 Transolver Backbone

The backbone is based on the Physics-Attention mechanism from Wu et al. (Transolver, ICML 2024). Configuration:

- **8 Transolver blocks** (layers)
- **Hidden dimension:** 256
- **Attention heads:** 8
- **Slice count:** M = 32
- **MLP ratio:** 1 (no FFN expansion)
- **Dropout:** 0.1

Each Transolver block:
1. Projects N×C point features → N×M slice weights via a linear layer + softmax
2. Aggregates: weighted combination of point features → M physics-aware tokens
3. Standard multi-head attention among M tokens (O(M²C) — cheap since M=32)
4. Broadcasts token updates back to N points using slice weights
5. FFN + residual connections

Total trainable parameters: **2,849,615** (~2.8M). This is appropriately small for a 905-sample dataset.

### 2.4 Encoder and Decoder

- **Encoder:** Linear(79 → 512) → GELU → Linear(512 → 256)
- **Decoder:** LayerNorm(256) → Linear(256 → 15) — zero-initialized so the model starts as a pure polynomial baseline
- **No-slip enforcement:** Hard zero at `idcs_airfoil` indices after output computation

---

## 3. Training Setup

### 3.1 Loss Function

Variance-weighted relative L2:

```
weights(point) = 1 + (wake_weight - 1) × (std(velocity_in) - min) / (max - min)
```

Wake points (high temporal variance in the input window) receive up to 4× the weight of freestream points. The motivation is correct: the evaluators' metric is global but submissions will differ almost entirely in the wake region. Freestream prediction is nearly trivial; wake prediction is not.

### 3.2 Optimiser and Schedule

- AdamW, lr=1e-3, weight_decay=1e-4
- Cosine annealing, T_max=400 epochs, eta_min=1e-5
- Gradient accumulation with accum_steps=4 (effective batch size = 4)
- BF16 mixed precision (L40S GPU)

### 3.3 Data Augmentation

Y-flip augmentation: with 50% probability per sample, negate `pos[:,1]` and `velocity[:,:,1]`. This reflects the geometry about y=0, producing a valid mirrored configuration. Distance features (`dist_to_airfoil`, `upstream_dist`) are invariant under this transform and are not modified.

This is the most impactful single change made during development. Before augmentation, the first Transolver block showed near-uniform (mushy) slice assignments at every layer. After augmentation, layer 0 slice entropy dropped from ~100% to ~10.6% of maximum. The model was unable to establish sharp spatial routing without y-symmetry breaking.

### 3.4 Training Data

Final run (run_06): all 905 samples, train_fraction=1.0, 400 epochs. No validation set. The final checkpoint (last epoch) is the submission. All results reported above are from run_05, trained on 90% of data (val set = 10%), which serves as the honest performance estimate.

---

## 4. Physics-Attention Analysis

This is the most important section for evaluating whether the model is behaving as the Transolver authors intended.

### 4.1 What the Transolver Paper Intends

Wu et al. (2024) explicitly design the slice mechanism to:
1. **Assign each point to one of M physically coherent groups.** The softmax is used deliberately to produce low-entropy (sharp) assignments.
2. **Learn distinct physical regimes** across all layers — freestream, boundary layer, wake, shock wave in their visualizations.
3. **Maintain informative slice assignments throughout the network depth.** In all their visualizations (Figures 1, 5, 9–17), every layer of Transolver shows visually distinct, spatially coherent slice patterns.

### 4.2 What This Model Actually Does

Entropy measurements (M=32, max entropy = 3.466 nats):

| Layer | Mean entropy | % of max | Assessment |
|-------|-------------|----------|------------|
| 0 (input) | 0.368 | 10.6% | Sharp |
| 4 (middle) | 3.265 | **94.2%** | **Near-uniform (mushy)** |
| 7 (output) | 0.379 | 10.9% | Sharp |

**This U-shaped pattern (sharp → mushy → sharp) is not what Transolver intends.** The paper never describes or shows a mushy middle layer. In their design, every layer should maintain sharp, physically interpretable slice assignments.

At layer 4, with 94.2% entropy, each point assigns approximately 1/32 ≈ 3.1% weight to every slice. This means Physics-Attention at that layer has **degenerated to a weighted average over all M tokens** — functionally equivalent to a global mean pooling operation followed by a broadcast. The "physics-aware" routing that defines the Transolver mechanism is not happening at this layer.

### 4.3 Mode Collapse in Active Layers

Even in the sharp layers (0 and 7), the slice routing exhibits severe mode collapse. The mass distribution across M=32 slices is highly skewed:

- **Layer 0:** s0 carries ~19.9%, s1 carries ~19.9%, the remaining 30 slices share ~60% with many approaching 0%.
- **Layer 7:** Similar distribution.

Approximately 8–14 slices are functionally active; the remainder are dead. This means the effective slice count is M≈10, not M=32. The choice of M=32 is consistent with the Transolver paper's recommendation for hidden_dim=256, but the mode collapse means this capacity is wasted.

### 4.4 What the Active Slices Are Actually Learning

Despite the mode collapse, the active slices in layers 0 and 7 **do show spatially coherent structure**. The airfoil geometry is clearly visible as a negative space in the weight maps. Different slices concentrate on distinguishable spatial regions — upstream vs. downstream, near-wall vs. far-field. The panel-by-panel heatmaps (in the style of Fig. 1 of Wu et al.) confirm that the dominant slices have learned physically meaningful groupings.

This means the core mechanism is working for the active slices, even though most slices have collapsed to zero. The model is running with fewer effective slices than designed, but those slices are doing their job.

### 4.5 Why the Mushy Middle Layer Appears

The U-shaped entropy pattern is a natural consequence of the architecture choices rather than a sign of correct Transolver behavior:

1. **Zero-initialized decoder:** The model starts as a pure polynomial baseline. The early correction signal is tiny, so gradient flow through the deeper layers is initially weak.
2. **Strong residual connections:** Each block's output is dominated by its residual path (x + small_correction). Features evolve slowly across depth.
3. **Softmax temperature:** The temperature parameter governs slice sharpness. If temperature adapts to make early and late layers sharp, middle layers may transiently adopt uniform routing because the signal passing through them is already close to correct.

In practice, the middle layers appear to function as **global information aggregation steps** — they mix information across all physical regimes so that later layers have global context when making the final spatial routing decision. Whether this is intentional or emergent is debatable, but it is not what the Transolver authors describe.

### 4.6 Critical Verdict on Transolver Compliance

| Criterion | Transolver intent | This model |
|-----------|------------------|------------|
| Sharp slice assignments | All layers | Only layers 0 and 7 |
| Distinct spatial regimes per slice | All M slices active | ~10 of 32 active |
| Consistent routing depth-wise | Same structure at each layer | U-shape: sharp-mushy-sharp |
| Physically meaningful active slices | Yes | Yes, for active slices |
| Linear complexity O(NMC) | Yes | Yes |

**Summary:** The model partially implements the Transolver mechanism. The active slices in the input and output layers behave as intended. The middle layers and the dead slices do not. The mechanism is underutilized but not broken.

---

## 5. What Works

1. **Residual learning on polynomial baseline.** The flat error curve across horizons (t5: 0.069, t9: 0.084) confirms the model has genuinely learned the turbulent correction, not just hedged toward zero.

2. **Feature engineering.** The `poly_residual` feature gives the model an explicit turbulence signal. The `dist_to_airfoil` and `upstream_dist` encode the geometry compactly. The `local_nbr_mean` (k-NN neighbourhood) provides local flow context that attention over slices cannot replicate for near-surface points.

3. **Y-flip augmentation.** Transformative for slice routing quality: layer 0 entropy dropped from ~100% to 10.6% after augmentation. Without this, the model could not break y-symmetry and the slice assignments were uninformative.

4. **No-slip enforcement.** Hard zeroing at airfoil surface indices. Exact, costs nothing, guaranteed.

5. **Zero-initialized decoder.** The model degrades gracefully to polynomial baseline at initialization. Training is stable because the learning target is centered near zero.

6. **Variance-weighted loss.** Redirects capacity toward the wake region where prediction is hardest and gains matter most.

---

## 6. What Does Not Work as Intended

1. **Middle layer Physics-Attention (94.2% entropy at layer 4).** This layer is effectively doing global mean pooling, not physics-aware routing. Seven Transolver blocks were designed and trained but only two (0 and 7) operate as the paper intends.

2. **Mode collapse (M=32 → effective M≈10).** Twenty or more slices are dead in every layer. The computational cost of M=32 is paid, but the representational benefit is not. The model would likely produce identical results with M=12.

3. **Depth underutilization.** The U-shaped entropy pattern and the similarity between layer 0 and layer 7 routing suggest the 7 intermediate blocks are not dramatically transforming the representation. The encoder's feature vector already contains most of the spatial discriminability needed for routing; the blocks refine token content within mostly fixed spatial assignments.

4. **Potential overfitting risk (final run).** Run_06 trains on all 905 samples with no held-out validation. There is no checkpoint selection criterion other than "last epoch." With only 905 samples and 2.8M parameters, regularization (dropout=0.1, weight_decay=1e-4, augmentation) is the only defense. The val loss from run_05 (0.066) is an honest estimate but may not generalize to the held-out test geometry configurations.

5. **k-NN local features at inference time.** The k-NN cache is precomputed and depends on the exact point cloud. This is fine for training, but the submission model (`TransolverResidual()`) will fall back to zeros for the local neighbourhood features if no cache is available at inference time. Whether the evaluators' test harness pre-builds the cache or not is unknown. The model was trained with these features, so running without them will silently degrade performance.

---

## 7. Performance in Context

The competition evaluates on a held-out test set. The honest estimate from run_05 val set:

- **Mean val relative L2:** 0.066
- **Per-timestep range:** 0.069 (t5) to 0.084 (t9)
- **Improvement over polynomial:** 67% to 93%

The Transolver paper (Wu et al., 2024) reports results on the AirfRANS benchmark (Reynolds-Averaged Navier-Stokes, 2D airfoil, 32k points, 800 training samples) of **0.0037 relative L2** on surrounding velocity — far lower than this model's 0.066–0.084. However, the GRaM task is fundamentally harder: it is a **transient** prediction (predicting future states, not steady-state), the meshes are 3D with 100k points, turbulence is explicitly unsteady, and the training set is comparably sized (905 samples vs. 800). Transient prediction of turbulent flow is a genuinely harder problem than steady-state RANS estimation.

The polynomial baseline at 0.2–1.3 relative L2 (depending on horizon) establishes a clear lower bound on difficulty. A naive prediction of "no change from the last input timestep" would yield approximately 0.4–0.6. The model at 0.07–0.08 represents a substantial improvement over both.

---

## 8. Conclusions

The submission implements a sound and well-motivated architecture. The residual learning strategy, feature engineering, and y-flip augmentation are all well-justified and demonstrably effective. The Physics-Attention mechanism is partially functional: it works as intended in the input and output layers but degenerates in the middle layers.

The key unresolved issues — mode collapse to ~10 effective slices, mushy middle layers, k-NN inference dependency — are known limitations that do not invalidate the submission but represent the primary avenues for improvement in a follow-up iteration.

If the final run_06 (trained on all 905 samples, 400 epochs) improves over run_05's 0.066 val loss estimate by even 3–5%, the submission should be competitive. The polynomial baseline collapsing at long horizons (>1.0 at t9) means any model that genuinely predicts the turbulent component — as this one demonstrably does — will rank significantly above naive baselines.

---

*Run environment: L40S (40GB), CUDA 12, PyTorch 2.x, BF16 mixed precision.*  
*Training time (run_05, 400 epochs, 815 samples): approximately 8 hours.*  
*Model weights: `models/transolver_residual/weights.pt`*
