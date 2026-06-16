# TransolverAR

**Authors:** Alex Colagrande, Theofanis Ifaistos, Anthony Kalaydjian

Transolver (Wu et al., ICML 2024) adapted for autoregressive velocity rollout. The backbone uses slice-based physics attention (64 slices, 8 heads, hidden dim 256, 6 layers) with a 5-step input window predicting 5 future steps.

Each training sample is subsampled uniformly to 20 000 mesh nodes, making the model mesh-size agnostic — at inference it runs directly on the full ~100 000-node mesh. Teacher forcing is applied per rollout step and decays linearly from 1.0 to 0.0 over the first 50% of training, after which the model is trained fully autoregressive. Training uses AdamW with cosine annealing (lr 2e-4, 500 epochs) on an 80/20 geometry-level train/val split.
