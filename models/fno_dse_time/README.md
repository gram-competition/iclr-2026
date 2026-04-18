
# FNO_DSE_TIME Competition Model

## General Description

FNO3d_dse_v2 — enhanced version of FNO3d_dse with two targeted improvements:

1. **Sinusoidal time embedding**: The actual time values t are encoded as sin/cos embeddings and added to every point's feature vector. The model now has an explicit clock rather than treating all timesteps as anonymous channels.
2. **Velocity residual skip connection**: The last input timestep of velocity_in is used as a direct skip to the output. The FNO learns to predict the *residual* on top of this prior. Low-frequency laminar flow is essentially free; all model capacity is focused on the high-frequency turbulent residual.

Everything else (VFT3d, SpectralConv3d_dse, training configs) is unchanged. Same forward() signature as FNO3d_dse.

## Architecture

- **Input channels:** 19 (positions, velocities, airfoil mask, time embedding)
- **Output channels:** 15 (velocity predictions for 5 timesteps × 3 components)
- **Modes:** 10 (spectral modes per dimension)
- **Width:** 32 (hidden feature size)
- **Blocks:** 4 spectral convolution blocks (SpectralConv3d_dse + 1x1 Conv1d skip)
- **Time embedding:** Sinusoidal, 16 dimensions (default)
- **Residual skip:** Last input velocity projected and added to output
- **Other layers:** Two fully connected layers (128 hidden units, then output)

## Training

- **Epochs:** 200
- **Learning rate:** 1e-3 (cosine annealing to 1e-5)
- **Batch size:** 4
- **Weight decay:** 1e-4 (default)
- **Loss:** MSE (mean squared error)
- **Validation split:** 10%
- **Mixed precision:** Enabled if CUDA is available
- **Model selection:** Best checkpoint by validation MAE
- **Augmentation:** Random subsampling of points (max_points=100000)
- **Distributed:** NCCL backend supported
- **Dataset:** Warped-IFW HDF5 dataset, with normalization of positions and velocities.
- **Loss function:** Mean Squared Error (MSE) between predicted and target velocities.
- **Optimizer:** Adam
- **Learning rate schedule:** CosineAnnealingLR, initial lr=1e-3, min lr=1e-5, 200 epochs
- **Weight decay:** 1e-4 (default, sometimes tuned)
- **Batch size:** 4 (default)
- **Validation split:** 10% of data for validation

