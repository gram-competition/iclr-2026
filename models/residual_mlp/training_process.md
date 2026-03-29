# Residual MLP — Training Process

## Architecture
- 4-layer MLP with residual connections, LayerNorm, GELU activation
- Input: 22 features per point (pos + flattened velocity + airfoil mask + velocity magnitude + surface proxy + time delta)
- Output: velocity delta (5 time steps x 3 components)
- Residual prediction: velocity_out = velocity_in[-1] + learned_delta
- No-slip boundary enforcement on airfoil surface
- 209,167 parameters

## Training
- Framework: Candle (pure Rust ML framework by HuggingFace)
- Data: 810 samples, 729 train / 81 val
- Normalization: velocity / 50.0
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
- Loss: MSE in normalized space
- Validation metric: L2 in original scale
- 3 epochs on CPU (preliminary — GPU training pending)

## Results (3 epochs, CPU)
| Epoch | Train MSE | Val L2 |
|-------|-----------|--------|
| 1     | 0.2575    | 30.95  |
| 2     | 0.0193    | 27.47  |
| 3     | 0.0124    | 24.90  |

## Key Design Choices
- **Residual prediction**: strong inductive bias for temporally smooth fluid flow
- **Per-point MLP**: avoids O(N^2) attention on 100K points; fast and memory-efficient
- **Explicit no-slip enforcement**: zeroes out airfoil surface velocity in output
- **Pure Rust training pipeline**: data loading (npyz + rayon), graph construction (kiddo kd-tree), model training (candle)
