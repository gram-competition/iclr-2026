"""Temporal mixing on top of per-node spatial embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttentionHead(nn.Module):
    """
    Per-node temporal mixing: each point gets a length-``t_out`` sequence, then a
    small Transformer encoder runs **along time** (batch = number of nodes).

    This lets future frames attend to each other so predictions stay temporally
    coherent, while spatial structure is already encoded in ``x`` from the GNN.

    **Attention backend:** ``nn.TransformerEncoderLayer`` uses
    ``nn.MultiheadAttention``, which on PyTorch 2.x dispatches through
    ``torch.nn.functional.scaled_dot_product_attention``. The actual kernel
    (math vs memory-efficient vs vendor-specific fused attention) depends on
    dtype, head dim, sequence length, and your build — including ROCm wheels
    that expose an efficient SDPA path. For GRaM-style horizons (``t_out`` is
    small, e.g. 5), cost is dominated by the **spatial** GNN over ``N`` nodes,
    not this temporal attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        t_out: int = 5,
        temporal_dim: int | None = None,
        num_heads: int = 4,
        num_attn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t_out = t_out
        td = int(temporal_dim or hidden_dim)
        if td % num_heads != 0:
            raise ValueError(
                f"temporal_dim ({td}) must be divisible by num_heads ({num_heads})."
            )
        self.td = td
        self.in_proj = (
            nn.Identity()
            if td == hidden_dim
            else nn.Linear(hidden_dim, td, bias=True)
        )

        self.time_query = nn.Parameter(torch.randn(1, t_out, td) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=td,
            nhead=num_heads,
            dim_feedforward=td * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attn_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, hidden_dim) -> (N, t_out, td)"""
        x = self.in_proj(x)
        h = x.unsqueeze(1).expand(-1, self.t_out, -1).contiguous()
        h = h + self.time_query
        return self.transformer(h)
