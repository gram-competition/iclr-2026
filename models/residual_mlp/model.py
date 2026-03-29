"""
Residual MLP for transient airflow prediction.
Trained in Rust (candle), weights loaded from safetensors.

Architecture: 4-layer MLP with residual connections + LayerNorm + GELU.
Predicts velocity delta from last input time step.
Enforces no-slip boundary on airfoil surface.
"""

import os

import torch
import torch.nn as nn
from safetensors.torch import load_file


VEL_SCALE = 50.0
POS_SCALE = 1.0
HIDDEN_DIM = 256
N_LAYERS = 4
INPUT_DIM = 22
OUTPUT_DIM = 15  # 5 time steps * 3 components


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Build layers matching Rust candle architecture
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: 22 -> 256
        self.linears.append(nn.Linear(INPUT_DIM, HIDDEN_DIM))
        self.norms.append(nn.LayerNorm(HIDDEN_DIM))

        # Hidden layers: 256 -> 256 with residual
        for _ in range(1, N_LAYERS):
            self.linears.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
            self.norms.append(nn.LayerNorm(HIDDEN_DIM))

        # Output: 256 -> 15
        self.output = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

        # Load weights from safetensors (candle format)
        path = os.path.join(os.path.dirname(__file__), "weights.safetensors")
        st = load_file(path)
        self._load_candle_weights(st)
        self.eval()

    def _load_candle_weights(self, st: dict):
        """Map candle VarMap keys to PyTorch state dict."""
        for i in range(N_LAYERS):
            self.linears[i].weight.data = st[f"l{i}.weight"]
            self.linears[i].bias.data = st[f"l{i}.bias"]
            self.norms[i].weight.data = st[f"ln{i}.weight"]
            self.norms[i].bias.data = st[f"ln{i}.bias"]
        self.output.weight.data = st["out.weight"]
        self.output.bias.data = st["out.bias"]

    def _build_features(self, pos, velocity_in, airfoil_mask, time_delta):
        """Build (n_points, 22) feature tensor per sample."""
        n_points = pos.shape[0]

        # Normalize
        pos_norm = pos / POS_SCALE
        vel_norm = velocity_in / VEL_SCALE  # (5, n_points, 3)

        # Flatten velocity: (n_points, 15)
        vel_flat = vel_norm.permute(1, 0, 2).reshape(n_points, 15)

        # Last velocity magnitude: (n_points, 1)
        last_vel = vel_norm[4]  # (n_points, 3)
        vel_mag = last_vel.norm(dim=1, keepdim=True)

        # Distance proxy: 1 - mask
        dist_proxy = 1.0 - airfoil_mask

        # Time delta
        t_delta = torch.full((n_points, 1), time_delta * 100.0, device=pos.device)

        return torch.cat([pos_norm, vel_flat, airfoil_mask, vel_mag, dist_proxy, t_delta], dim=1)

    def _forward_single(self, features):
        """Forward pass on (n_points, 22) features."""
        n_points = features.shape[0]
        h = features

        for i, (linear, norm) in enumerate(zip(self.linears, self.norms)):
            out = torch.nn.functional.gelu(norm(linear(h)))
            if i > 0:
                h = out + h
            else:
                h = out

        delta = self.output(h)  # (n_points, 15)
        delta = delta.view(n_points, 5, 3).permute(1, 0, 2)  # (5, n_points, 3)
        return delta

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = t.shape[0]
        results = []

        for b in range(batch_size):
            n_points = pos.shape[1]

            # Build airfoil mask
            mask = torch.zeros(n_points, 1, device=pos.device)
            mask[idcs_airfoil[b]] = 1.0

            # Time delta
            if t.shape[1] >= 6:
                td = (t[b, 5] - t[b, 4]).item()
            else:
                td = 0.002

            # Build features
            features = self._build_features(pos[b], velocity_in[b], mask, td)

            # Predict delta
            delta = self._forward_single(features)  # (5, n_points, 3)

            # Residual: last input velocity + delta
            last_vel = velocity_in[b, 4] / VEL_SCALE  # (n_points, 3)
            pred = last_vel.unsqueeze(0).expand(5, -1, -1) + delta

            # Denormalize
            pred = pred * VEL_SCALE

            # Enforce no-slip
            flow_mask = (1.0 - mask).unsqueeze(0).expand(5, -1, -1)  # (5, n_points, 1)
            pred = pred * flow_mask

            results.append(pred)

        return torch.stack(results, dim=0)
