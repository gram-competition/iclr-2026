"""
Transolver block with Physics-Attention for irregular 3-D meshes.

Directly adapted from the official Transolver repo:
    github.com/thuml/Transolver  —  Physics_Attention_Irregular_Mesh

Key idea
--------
Instead of attending over all N points (O(N²)), project each point into one of
M "physics slices" via a learned soft assignment, run standard attention only
among the M slice tokens (O(M²)), then broadcast back to N points.

Complexity: O(N·M·C + M²·C)  — linear in N since M is a small constant.

The slices learn to separate physical regimes (freestream / boundary layer /
wake / inter-airfoil gaps) without explicit supervision.
"""

import torch
import torch.nn as nn
from einops import rearrange


# ---------------------------------------------------------------------------
# Physics-Attention (irregular mesh)
# ---------------------------------------------------------------------------

class PhysicsAttention(nn.Module):
    """
    Multi-head physics-attention for irregular meshes.

    Args:
        dim       : token dimension (= hidden_dim)
        heads     : number of attention heads
        dim_head  : dimension per head  (inner_dim = heads × dim_head)
        slice_num : number of physics slices M
        dropout   : dropout probability
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        slice_num: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads    = heads
        self.dim_head = dim_head
        self.scale    = dim_head ** -0.5

        # Learnable temperature for slice softmax (clamped for stability)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

        # Point → inner_dim projections (two separate paths: positional & value)
        self.proj_x  = nn.Linear(dim, inner_dim)   # used to compute slice weights
        self.proj_fx = nn.Linear(dim, inner_dim)   # used as "values" to aggregate

        # Slice assignment: dim_head → M  (orthogonal init for diversity)
        self.proj_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.proj_slice.weight)

        # Standard Q K V projections on the M slice tokens
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N, C)
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape

        # ── (1) Project to multi-head space ──────────────────────────────────
        # x_mid  — determines which slice each point belongs to
        x_mid  = self.proj_x(x).reshape(B, N, self.heads, self.dim_head) \
                     .permute(0, 2, 1, 3)                   # (B, H, N, dim_head)
        # fx_mid — the "content" to aggregate into slices
        fx_mid = self.proj_fx(x).reshape(B, N, self.heads, self.dim_head) \
                     .permute(0, 2, 1, 3)                   # (B, H, N, dim_head)

        # ── (2) Slice: soft-assign each point to M physics slices ────────────
        temp = self.temperature.clamp(0.1, 5.0)
        slice_weights = self.softmax(
            self.proj_slice(x_mid) / temp
        )                                                   # (B, H, N, M)

        # Weighted aggregate → M slice tokens
        slice_norm  = slice_weights.sum(dim=2)              # (B, H, M)
        slice_token = torch.einsum("bhnc,bhnm->bhmc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-5).unsqueeze(-1)
        #                                                     (B, H, M, dim_head)

        # ── (3) Attention among M slice tokens ───────────────────────────────
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, M, M)
        attn = self.dropout(self.softmax(dots))
        out_token = torch.matmul(attn, v)                  # (B, H, M, dim_head)

        # ── (4) Deslice: broadcast slice tokens back to N points ─────────────
        out_x = torch.einsum("bhmc,bhnm->bhnc", out_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")   # (B, N, inner_dim)

        return self.to_out(out_x)


# ---------------------------------------------------------------------------
# Transolver Block
# ---------------------------------------------------------------------------

class TransolverBlock(nn.Module):
    """
    One Transolver encoder block:
        x ← x + PhysicsAttention(LayerNorm(x))
        x ← x + FFN(LayerNorm(x))

    Args:
        dim       : hidden dimension
        heads     : attention heads
        slice_num : number of physics slices M
        mlp_ratio : FFN hidden = dim × mlp_ratio
        dropout   : dropout probability
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        slice_num: int = 64,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = PhysicsAttention(
            dim=dim,
            heads=heads,
            dim_head=dim // heads,
            slice_num=slice_num,
            dropout=dropout,
        )

        ffn_dim = dim * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
