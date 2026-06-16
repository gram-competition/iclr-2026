# Training Process — harshitsinghsnu

## Architecture
Point-wise MLP operating on **all 100k points** independently (embarrassingly parallel).

| Component | Detail |
|---|---|
| Layers | 4 × ResidualBlock (256 → 512 → 512 → 256) |
| Activation | GELU |
| Normalisation | LayerNorm after each linear |
| Skip connection | velocity_in → output (strong physical prior) |
| Inference | Chunked (8 192 pts/chunk) to fit any GPU |

## Input features (29 channels per point)
| Feature | Channels | Notes |
|---|---|---|
| `pos` | 3 | xyz coordinate |
| `velocity_in` | 15 | 5 timesteps × 3, flattened |
| `t` | 10 | broadcast to every point |
| `dist_airfoil` | 1 | normalised distance to airfoil centroid |

## Training
- **Loss**: MSE on `velocity_out`
- **Optimizer**: AdamW (lr=1e-3, weight\_decay=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Point subsampling**: 4 096 points per cloud during training (memory)
- **Environment**: Google Colab T4 GPU

## Reproducing
Run the notebook `GRaM_Competition_Colab.ipynb` top-to-bottom.