# SpatioTemporalGNN (fmgreco_stnn)

- **PR narrative (code-aligned):** see [`PR22_TECHNICAL_BRIEF.md`](PR22_TECHNICAL_BRIEF.md) for greppable claims, backbone wording, and SDPA notes.
- **Architecture:** Full-resolution point cloud → `scipy` k-NN graph → **residual** MeshGraphNet blocks (or optional SO(3) / steerable backbone) → **TransformerEncoder** over the `t_out` time tokens per node (`norm_first`, GELU FFN) → **residual Δ** (decoder predicts change w.r.t. last/mean input frame) → **`apply_hard_no_slip`** on `idcs_airfoil` (hard zero on boundary). Optional **pressure** node features (`use_pressure`); optional **train** loss on **Δ-velocity** vs last frame (`train_loss_on_velocity_delta`); val/test can stay **full-field** RL2.
- **Training CLI:** `latent_dim` → `hidden_dim`, `num_blocks` → spatial GNN depth, `num_temporal_layers` → temporal Transformer depth. Example: `python main.py --config train_config.yaml --dataset-dir /path/to/npz`.
- **Defaults:** **hidden_dim=128**, **num_layers=12**, **num_fourier=8**, **num_attn_layers=2**. Retrain after architecture changes; bundled weights may only partially load (`strict=False`).
- **Optional SO(3) backbone:** `use_so3_backbone=True` (CLI `--use-so3-backbone`) replaces MeshGraphNet with k-NN equivariant vector channels (parallel/perp decomposition along edges). Does **not** require pressure in the tensors; invariant channels are learned. Chiral / mirror-asymmetric flows are **not** forced to be identical under reflection.
- **Weights:** Optional `models/fmgreco_stnn/state_dict.pt` (full `state_dict` or `{"model_state_dict": ...}`). Without it, random init (for development only).
- **Dependencies:** `scipy` strongly recommended for k-NN at 100k points (`pip install scipy`).
- **Note:** Training on the official Hugging Face dataset is done offline; this package is the inference graph only.
