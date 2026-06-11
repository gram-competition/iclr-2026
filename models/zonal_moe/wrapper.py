import os
import torch
from torch_geometric.nn import knn_graph
from .model import ZonalMoE
from .preprocessing import compute_wall_distance

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer(
            "vel_mean",
            torch.tensor(
                [37.750118255615234, 0.5372318625450134, 2.009599447250366],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "vel_std",
            torch.tensor(
                [19.8649845123291, 7.343273639678955, 9.551141738891602],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "pos_mean",
            torch.tensor(
                [0.8507418036460876, -6.422636200653642e-09, 0.37120404839515686],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "pos_std",
            torch.tensor(
                [0.40274253487586975, 0.07883177697658539, 0.2320450097322464],
                dtype=torch.float32,
            ),
        )
        # Initialize the core model
        self.model = ZonalMoE(
            wall_dist_scale=0.28871151953935625, vorticity_scale=10.57309174537657
        )

        # Load weights
        base_path = os.path.dirname(__file__)
        weights_path = os.path.join(base_path, "weights.pt")

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            # Extract just the model state dict if it was saved as a checkpoint dict
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict)
            print(f"Loaded ZonalMoE weights from {weights_path}")
        else:
            print(f"Warning: ZonalMoE weights not found at {weights_path}")
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        t: torch.Tensor,  # time tensor — intentionally unused; our physics-based model derives temporal structure from vel_in directly
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for evaluation.
        Args:
            t: (B, 1) or (B, T) - not strictly used by our model directly
            pos: (B, N, 3) - unnormalized positions
            idcs_airfoil: list of length B, where each element is (M,) airfoil indices
            velocity_in: (B, T, N, 3) - unnormalized input velocity (T=5)
        Returns:
            velocity_out: (B, T_out, N, 3) - absolute predicted velocity (T_out=5)
        """
        # Process each sample independently — N=100k points with k=32 graph
        # makes true batching an OOM risk, so a per-sample loop is correct here.
        outputs = []
        for b in range(pos.shape[0]):
            pos_b = pos[b]           # (N, 3)
            vel_in_b = velocity_in[b]  # (T, N, 3)
            idx_b = idcs_airfoil[b]  # (M,)

            # 1. On-the-fly physical features
            wall_dist = compute_wall_distance(pos_b, idx_b)
            is_airfoil = torch.zeros(pos_b.shape[0], dtype=torch.bool, device=pos_b.device)
            is_airfoil[idx_b] = True

            # 2. Normalize inputs
            pos_norm = (pos_b - self.pos_mean) / (self.pos_std + 1e-8)
            vel_in_norm = (vel_in_b - self.vel_mean) / (self.vel_std + 1e-8)

            edge_index = knn_graph(pos_norm, k=16, loop=False)
            edge_index_dense = knn_graph(pos_norm, k=32, loop=False)

            vel_out_norm = self.model.predict(
                vel_in=vel_in_norm,
                edge_index=edge_index,
                wall_dist=wall_dist,
                is_airfoil=is_airfoil,
                batch=None,
                edge_index_dense=edge_index_dense,
            )

            vel_out_raw = vel_out_norm * self.vel_std + self.vel_mean
            vel_out_raw[:, idx_b, :] = 0.0
            outputs.append(vel_out_raw)

        return torch.stack(outputs, dim=0)  # (B, T_out, N, 3)
