# SpatioTemporalGNN (fmgreco_stnn)

- **Architecture:** Full-resolution point cloud → `scipy` k-NN graph → MeshGraphNet-style message passing → temporal attention head → velocity delta + residual from last input time → **no-slip** (zero velocity on `idcs_airfoil`).
- **Weights:** Optional `models/fmgreco_stnn/state_dict.pt` (full `state_dict` or `{"model_state_dict": ...}`). Without it, random init (for development only).
- **Dependencies:** `scipy` strongly recommended for k-NN at 100k points (`pip install scipy`).
- **Note:** Training on the official Hugging Face dataset is done offline; this package is the inference graph only.
