"""Physics-Attention with Eidetic States (Transolver++).

Reference: Wu et al. (2025) "Transolver++: An Accurate Neural Solver for
PDEs on Million-Scale Geometries".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gumbel_softmax(logits: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y = (logits + gumbel_noise) / tau
    return F.softmax(y, dim=-1)


class PhysicsAttention1DEidetic(nn.Module):
    def __init__(self, dim: int, n_head: int = 8, slice_num: int = 32,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.dim_head = dim // n_head
        self.n_head = n_head

        self.bias = nn.Parameter(torch.ones(1, n_head, 1, 1) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(self.dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU(),
        )
        self.in_project_x = nn.Linear(dim, self.dim_head * n_head)
        self.in_project_slice = nn.Linear(self.dim_head, slice_num)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.dim_head * n_head, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        x_mid = self.in_project_x(x)
        x_mid = (x_mid.reshape(B, N, self.n_head, self.dim_head)
                 .permute(0, 2, 1, 3).contiguous())

        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = temperature.clamp(min=0.01)
        slice_weights = _gumbel_softmax(self.in_project_slice(x_mid), temperature)

        slice_norm = slice_weights.sum(dim=2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights)
        slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1e-5)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(q, k, v)

        out = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.to_out(out)
