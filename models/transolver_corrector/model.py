"""Competition submission wrapper for CorrectedTransolver.

Self-contained model: no-arg constructor, downloads weights from
Hugging Face, handles all preprocessing (time encoding, airfoil mask)
internally.  Velocity and position are passed through **unnormalized**
(the checkpoint was trained with ``preprocessors: []``).
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .transolver import TransolverPlusFlowModel
from .wake_corrector import WakeCorrectorGNN


N_INPUT_T = 5


class TransolverCorrector(nn.Module):
    """Two-stage model: global Transolver backbone + local GNN corrector.

    Instantiated without arguments for competition submission.
    Weights are downloaded from Hugging Face at init time.
    """

    def __init__(self) -> None:
        super().__init__()

        self.backbone = TransolverPlusFlowModel(
            n_hidden=192, n_layers=3, n_head=8, slice_num=32,
            mlp_ratio=2, dropout=0.21, use_dist_to_airfoil=True,
            time_n_frequencies=16, time_omega_min=10.0, time_omega_max=1000.0,
            input_cond_n_frequencies=8,
        )
        self.corrector = WakeCorrectorGNN(
            node_dim=10, hidden_dim=192, n_layers=10, k=16,
            top_fraction=0.4, mask_sharpness=5.0,
        )

        path = hf_hub_download(
            repo_id="MoosChance/TranssolverGram",
            filename="state_dict_weights_only.pt",
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
        else:
            sd = checkpoint
        clean = OrderedDict()
        for k, v in sd.items():
            clean[k.removeprefix("module.")] = v
        self.load_state_dict(clean, strict=True)

        self.eval()

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        """Predict future velocity fields.

        Parameters
        ----------
        t : (B, 10)
        pos : (B, N, 3)
        idcs_airfoil : list of B variable-length index tensors
        velocity_in : (B, 5, N, 3)

        Returns
        -------
        torch.Tensor
            (B, 5, N, 3) — predicted velocity in physical units.
        """
        B, N = pos.shape[0], pos.shape[1]
        device = pos.device

        # --- Build airfoil mask ---
        airfoil_mask = torch.zeros(B, N, device=device)
        for b in range(B):
            airfoil_mask[b, idcs_airfoil[b]] = 1.0

        # --- Compute time offsets ---
        t_inputs = t[:, :N_INPUT_T]           # (B, 5)
        t_seed = t[:, N_INPUT_T - 1]          # (B,)
        t_outputs = t[:, N_INPUT_T:]           # (B, 5)

        delta_t = t_outputs - t_seed.unsqueeze(1)              # (B, 5)
        delta_t_inputs = t_outputs.unsqueeze(2) - t_inputs.unsqueeze(1)  # (B, 5, 5)

        # --- Forward through model (raw/unnormalized data) ---
        data = {
            "velocity_in": velocity_in,
            "delta_t": delta_t,
            "delta_t_inputs": delta_t_inputs,
            "pos": pos,
            "airfoil_mask": airfoil_mask,
        }

        u_base = self.backbone(data)
        velocity_out = self.corrector(
            u_base=u_base, pos=pos,
            velocity_in=velocity_in, airfoil_mask=airfoil_mask,
        )

        return velocity_out
