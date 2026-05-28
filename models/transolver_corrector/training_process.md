# CorrectedTransolver — Training Process

## Architecture

Two-stage model for 3D velocity field prediction on irregular point clouds:

1. **Transolver+ backbone** — global predictions via Physics-Attention with Eidetic States (Wu et al., 2025). Time conditioning via FiLM (Feature-wise Linear Modulation) on LayerNorm layers, driven by a sinusoidal time encoder. Operates as a flow map: conditions on the full 5-step input window and predicts residuals on top of the last input step.

2. **Wake Corrector GNN** — local refinement via EdgeConv V2 (sum aggregation + edge features) on variance-selected wake points. A variance gate selects the top 40% of points by temporal velocity variance, builds a k-NN graph (k=16), and runs 10 layers of message passing to predict corrections to the backbone output.

### Key features
- **Distance-to-airfoil**: log(1 + min_dist) as additional input feature
- **Variance-gated point selection**: focuses corrector computation on turbulent wake regions
- **No-slip boundary condition**: enforced by zeroing velocity at airfoil points

## Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Backbone | hidden_dim | 192 |
| Backbone | n_layers | 3 |
| Backbone | n_heads | 8 |
| Backbone | slice_num | 32 |
| Backbone | dropout | 0.21 |
| Backbone | time_n_frequencies | 16 |
| Corrector | hidden_dim | 192 |
| Corrector | n_layers | 10 |
| Corrector | k (kNN) | 16 |
| Corrector | top_fraction | 0.4 |
| Corrector | backbone | EdgeConv V2 |

## Training

- **Loss**: Wake-weighted L1 (alpha=1.0) — per-point L1 weighted by local velocity variance
- **Optimizer**: AdamW, lr=5e-4, weight_decay=1e-5
- **Scheduler**: Cosine annealing
- **Batch size**: 2 per GPU, 3x A100-80GB (DDP)
- **Epochs**: ~400
- **Preprocessing**: none (raw physical units for both velocity and position)
- **Augmentations**: y/z axis reflections (50% each), variance-scaled noise (base_std=0.005, scale=0.02, zeroed at airfoil)