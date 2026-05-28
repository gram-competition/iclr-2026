from __future__ import annotations

import torch
from torch import nn


class WaveletFiLM(nn.Module):
    def __init__(self, context_dim: int, feature_dim: int, scale: float = 0.25) -> None:
        super().__init__()
        self.scale = scale
        self.proj = nn.Linear(context_dim, 2 * feature_dim)

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma_raw, beta_raw = self.proj(context).chunk(2, dim=-1)
        gamma = 1.0 + self.scale * torch.tanh(gamma_raw)
        beta = self.scale * torch.tanh(beta_raw)

        if features.ndim == 3:
            # (B, N, C)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        elif features.ndim == 5:
            # (B, C, X, Y, Z)
            gamma = gamma[:, :, None, None, None]
            beta = beta[:, :, None, None, None]
        else:
            raise ValueError(f"Unsupported feature shape for WaveletFiLM: {tuple(features.shape)}")

        return features * gamma + beta


class WaveletResidualBlock(nn.Module):
    """Residual 3D block modulated by a geometry context vector."""

    def __init__(self, hidden_dim: int, context_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, hidden_dim)
        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, hidden_dim)
        self.act = nn.GELU()
        self.film = WaveletFiLM(context_dim, hidden_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = self.act(self.norm1(self.conv1(x)))
        hidden = self.norm2(self.conv2(hidden))
        hidden = self.film(hidden, context)
        return self.act(residual + hidden)
