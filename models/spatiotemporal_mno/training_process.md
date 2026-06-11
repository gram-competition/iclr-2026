# Spatiotemporal MNO

This is the basic Multiscale Neural Operator submission.

> An earlier GRU-based latent-temporal variant also lived in this folder, but it
> was **not** used for the submission. The shipped model is the basic MNO
> documented below, and its trained weights (`state_dict.pt`) are loaded
> automatically by the constructor.

## Idea

The full 5-frame velocity history is flattened into one per-point feature
vector. A shared encoder lifts it to a latent field, a stack of geometry-aware
MNO blocks mixes it spatially, and a decoder produces velocity residuals for all
future frames at once around a persistence baseline.

## Why this matches the challenge

- The airfoil geometry is fixed across the 10 timestamps, so a single spatial
  pass over the point cloud captures the relevant structure.
- The output is a per-point velocity field, so the decoder keeps point alignment
  and the no-slip boundary constraint.

## Implementation details

File: `models/spatiotemporal_mno/model.py`

- Inputs (per point):
  - `pos`
  - flattened velocity history (all observed frames)
  - Fourier embedding of every timestamp (input + output)
  - airfoil mask
  - wall distance
  - local surface frame
- Backbone:
  - encoder MLP
  - stack of MNO blocks over the point cloud
- Decoder:
  - pointwise decoder MLP to all future residual frames
  - per-output-step horizon gating from the output-timestamp embedding
- Physics prior:
  - persistence baseline from the last observed frame
  - hard no-slip enforcement on airfoil points in scaled space
- Pretrained weights:
  - `state_dict.pt` ships in this folder and is auto-loaded in the constructor
    (`load_pretrained=True` by default; pass `load_pretrained=False` to train
    from scratch).

## Training workflow

Train with the shared CLI:

```bash
python scripts/train.py --config config/spatiotemporal_mno.yaml
```

Or override from the command line:

```bash
python scripts/train.py \
  --model-name spatiotemporal_mno \
  --config config/spatiotemporal_mno.yaml \
  --dataset-dir dataset_huggingface/warped-ifw
```

Evaluate:

```bash
python scripts/evaluate.py \
  --config config/spatiotemporal_mno.yaml \
  --model-name spatiotemporal_mno \
  --checkpoint-path outputs/runs/<run_name>/checkpoints/best.pt \
  --split test
```

## Practical notes

- This model keeps the same training loop, loss functions, dataset pipeline, and checkpoint format as the baseline.
- It is heavier than the baseline because it applies MNO blocks before and after temporal forecasting.
- The temporal module is direct multi-step forecasting, not autoregressive rollout, to limit compounding error over the 5 predicted steps.
