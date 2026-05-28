# PerceiverFlow

Geometry-conditioned spatiotemporal flow predictor for 3D airfoil velocity fields.

## Architecture
- Perceiver IO spatial encoder (256 latent queries)
- Temporal Transformer (3 layers) conditioned on geometry embedding
- Perceiver IO spatial decoder

## Training
- 810 samples, 50 epochs, AdamW lr=3e-4
- Freestream-normalized direct velocity prediction
- Physics-aware loss with near-surface upweighting

## Weights
Hosted on HuggingFace: `hd-hg/perceiver-flow-weights`

## For Inference
model = PerceiverFlow()   # this should print "PerceiverFlow: weights loaded from HuggingFace"
model.eval()

# Dummy forward pass matching exact competition signature
B, N = 1, 1000   # use 1000 points instead of 100k to keep it fast
dummy_t            = torch.zeros(B, 10)
dummy_pos          = torch.randn(B, N, 3)
dummy_idcs_airfoil = [torch.randint(0, N, (200,))]
dummy_vel_in       = torch.randn(B, 5, N, 3)

with torch.no_grad():
    out = model(dummy_t, dummy_pos, dummy_idcs_airfoil, dummy_vel_in)

print(f"Output shape: {out.shape}")   # should be torch.Size([1, 5, 1000, 3])