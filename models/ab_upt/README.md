# AB-UPT (pragmatic variant)

A stripped-down and simplified implementation of the Anchored-Branched Universal Physics
Transformer ([Alkin et al., 2025](https://arxiv.org/abs/2502.09692)), adapted
to this competition's setting. The goal was a practical single-GPU submission
rather than a faithful reproduction of the original architecture. Several
components were simplified and hyperparameters were kept modest.

## Architecture

Per-batch sample:

1. **Point embedding** — position, Fourier features of position, the five
   input velocity fields, the start time, and an is-surface flag are
   concatenated and projected to a hidden representation.
2. **Supernode sampling** — a small set of anchor points is drawn per sample:
   surface anchors (from airfoil indices), wake anchors (top-variance points
   in the volume, where "variance" is per-point temporal variance of the
   input velocity), and far-field anchors (the rest of the volume).
3. **kNN message passing** — each supernode mean-pools messages from its k nearest volume
   neighbors (each message is an MLP of neighbor feature + relative position). 
4. **Branched transformer** — surface and volume supernode streams run in
   parallel with shared self-attention / FFN weights; a cross-branch
   attention step every other block lets the streams communicate.
5. **Perceiver decoder** — every point attends to the final supernode
   latents (two cross-attention blocks).
6. **Head** — per-point 5-step velocity deltas added to the last input frame,
   hard-masked to zero on airfoil points.

Inputs are normalized with per-component mean/std stored in `norm_stats.pt`.

## Training

- Dataset: y-axis mirror augmentation of the competition training split
  (spanwise symmetry of the front-wing flow). The test split is not
  augmented.
- Optimizer: AdamW, cosine LR schedule (peak 5e-4, min 0), 600 epochs.
- Precision: bf16 autocast.
- EMA on model weights (decay 0.999, linear warmup over 100 steps); the
  best EMA checkpoint on the held-out split is shipped here.

## Files

- `model.py` — `ABUPT` class (zero-arg, loads weights and norm stats from
  this directory in `__init__`, moves to CUDA if available, runs forward
  under bf16 autocast + inference_mode by default).
- `state_dict.pt` — trained EMA weights (~21 MB).
- `norm_stats.pt` — position and velocity normalization buffers.
