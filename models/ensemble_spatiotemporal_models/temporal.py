
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttentionHead(nn.Module):


    def __init__(self, hidden_dim: int, t_out: int = 5,
                 temporal_dim: int | None = None, num_heads: int = 4,
                 num_attn_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.t_out = t_out
        self.td = temporal_dim or hidden_dim

        self.proj = nn.Linear(hidden_dim, t_out * self.td)

        self.attn_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        for _ in range(num_attn_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(self.td, num_heads,
                                      dropout=dropout, batch_first=True)
            )
            self.ff_layers.append(nn.Sequential(
                nn.Linear(self.td, self.td * 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(self.td * 2, self.td),
                nn.Dropout(dropout),
            ))
            self.norm1.append(nn.LayerNorm(self.td))
            self.norm2.append(nn.LayerNorm(self.td))

        # Learnable temporal position embeddings
        self.time_pe = nn.Parameter(torch.randn(1, t_out, self.td) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, hidden_dim)

        Returns:
            (N, T_out, temporal_dim)
        """
        N = x.size(0)
        h = self.proj(x).view(N, self.t_out, self.td)  # (N, T, td)
        h = h + self.time_pe

        for attn, ff, n1, n2 in zip(
            self.attn_layers, self.ff_layers, self.norm1, self.norm2
        ):
            res = h
            h, _ = attn(h, h, h)
            h = n1(res + h)
            h = n2(h + ff(h))

        return h  # (N, T_out, td)
