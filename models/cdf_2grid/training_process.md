# CDFDoubleGridNet

## Overview
**CDFDoubleGridNet** is a multi-scale voxel U-Net with temporal attention and
CDF grid warping in each axis.

---

## Architecture

| Component | Detail |
|-----------|--------|
| **Temporal encoder** | Per-point linear projection (3→32) → 4-head self-attention across T=5 frames, applied independently at each spatial point |
| **Fourier positional encoding** | 8-band sinusoidal encoding of physically-normalized coordinates (`(pos - pos_mean) / pos_std`); frequencies 2^k·π, k=0…7 |
| **SDF embedding** | 2-layer MLP mapping `(sdf/5, log1p(sdf·10)/2.4)` → 16-D geometry-aware feature |
| **Multi-scale voxel U-Net** | Two-level CDF-warped voxel pyramid (32³ coarse + 80³ fine); each level: scatter-mean into voxels → 1×1 Conv3D context fusion → 3-level U-Net → trilinear sample-back |
| **CDF grid warping** | Gaussian-smoothed CDF mapping warps physical positions into uniform computational coordinates, concentrating voxel resolution near the airfoil surface and high-density flow regions |
| **Residual prediction** | Output = last input frame + predicted delta |
| **No-slip enforcement** | Airfoil surface points are zeroed in the output (hard physical constraint) |
| **TTA** | Y-axis flip test-time averaging (two forward passes averaged at inference) |

**Parameter count:** 33.78 M

---

## Normalization (handled automatically inside model.py)

Two separate normalizations are applied internally:

1. **Global stats** (`vel_mean`, `vel_std`, `pos_mean`, `pos_std`) — computed on
   the training set, stored as registered buffers, and saved inside
   `state_dict.pt`. Applied to velocity and position before the Fourier encoder
   and temporal attention.
2. **Per-sample CDF normalization** — positions are rescaled to `[0, 1]`
   per-sample at runtime inside `_forward_single`, then passed through the
   Gaussian-smoothed CDF map to produce uniform computational coordinates for
   the voxel grid. This is computed on the fly and not stored.

The model accepts raw, unscaled data directly in the competition signature:

```python
from models import CDFDoubleGridNet
model = CDFDoubleGridNet() # Weights are downloaded automatically from a public HuggingFace repository — no HF token required.
velocity_out = model(t, pos, idcs_airfoil, velocity_in)
```

or

```python
from models.cdf_2grid.model import Model
model = Model()   
velocity_out = model(t, pos, idcs_airfoil, velocity_in)
```

No caller pre-processing is needed.

---

## Training objective

Relative L2 loss + 0.01 × absolute L2 loss.
Autoregressive pushforward training on 50% of batches.

---

## Hardware requirement

A CUDA GPU is strongly recommended. The SDF computation performs chunked `cdist` over up to
100k × 20k points, and TTA runs two full forward passes per sample. CPU
inference is slow for any practical evaluation.

Approximate inference time: ~2s per sample with TTA tested on a RTX2080Ti (11GB)

---

## Dependencies

```
torch >= 2.1
numpy
huggingface_hub
```