# PT-PointNet

A lightweight PointNet-style baseline with local Point-Transformer attention over 
k-nearest neighbors. Global context is carried by a single max-pool vector concatenated 
back onto each point. Simpler and smaller than the AB-UPT entry, but competitive on this
task.

## Architecture

Per-batch sample:

1. **Point embedding** — position, Fourier features of position (frequencies
   `1, 2, 4, 8, 16, 32, 64, 128`), the five input velocity fields (normalized), and the
   start time are concatenated and projected to a hidden representation.
2. **k-NN neighborhood** — k=16 nearest neighbors are computed once on
   positions and reused across all blocks.
3. **Point-Transformer blocks (×2)** — subtraction-form attention over the k
   neighbors with a learned relative-position encoding δ added to both keys
   and values; attention weights come from an MLP γ(q − k + δ) producing
   per-head softmax weights over the neighborhood. FFN + pre-LN + residual
   around each sub-block.
4. **Global pooling** — per-sample max over points gives a single global
   feature vector, broadcast back onto every point.
5. **Decoder** — concatenation of [point embedding, neighborhood feature,
   global feature] is projected to per-point 5-step velocity deltas.
6. **Head** — deltas are added to the last input frame, denormalized, and
   hard-masked to zero on airfoil points.

Inputs are normalized with per-component mean/std stored in `norm_stats.pt`.

## Training

- Dataset: y-axis mirror augmentation of the competition training split
  (spanwise symmetry of the front-wing flow).
- Optimizer: Adam, cosine LR schedule (peak 5e-4, min 0), 150 epochs,
  batch size 4.
- Precision: bf16 autocast (with activation checkpointing during training
  to fit on a single 24 GB GPU).
- EMA on model weights (decay 0.999, linear warmup over 100 steps); the
  best EMA checkpoint on the held-out split is shipped here.

## Files

- `model.py` — `PTPointNet` class (zero-arg, loads weights and norm stats
  from this directory in `__init__`, moves to CUDA if available, runs
  forward under bf16 autocast + inference_mode by default).
- `state_dict.pt` — trained EMA weights (~1.8 MB).
- `norm_stats.pt` — position and velocity normalization buffers.
