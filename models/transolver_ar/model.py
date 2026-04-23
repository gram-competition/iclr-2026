"""
TransolverAR submission for GRaM @ ICLR 2026.

Architecture: slice-based Transformer for irregular meshes (Transolver, Wu et al. ICML 2024),
adapted for autoregressive velocity rollout.

Trained on warped-ifw with 20k-node subsampling; generalises to full 100k meshes
(val nRMSE 0.1145 full mesh vs 0.1164 subsampled, epoch 20).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_


# ── Internal architecture ─────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 n_layers: int = 1, res: bool = True):
        super().__init__()
        act = nn.GELU
        self.res = res
        self.linear_pre  = nn.Sequential(nn.Linear(n_in, n_hidden), act())
        self.linears     = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), act())
            for _ in range(n_layers)
        ])
        self.linear_post = nn.Linear(n_hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)
        for lin in self.linears:
            x = (lin(x) + x) if self.res else lin(x)
        return self.linear_post(x)


class _PhysicsAttention(nn.Module):
    """Slice-based attention for unstructured meshes — O(N·G) complexity."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0., slice_num: int = 64):
        super().__init__()
        inner_dim        = dim_head * heads
        self.dim_head    = dim_head
        self.heads       = heads
        self.scale       = dim_head ** -0.5
        self.softmax     = nn.Softmax(dim=-1)
        self.dropout     = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

        self.proj_x      = nn.Linear(dim, inner_dim)
        self.proj_fx     = nn.Linear(dim, inner_dim)
        self.proj_slice  = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.proj_slice.weight)

        self.to_q        = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k        = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v        = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out      = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [B, N, C]
        B, N, _ = x.shape
        H, D    = self.heads, self.dim_head

        fx = self.proj_fx(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
        xm = self.proj_x (x).reshape(B, N, H, D).permute(0, 2, 1, 3)

        sw  = self.softmax(self.proj_slice(xm) / self.temperature)  # B H N G
        sn  = sw.sum(2)                                               # B H G
        st  = torch.einsum("bhnc,bhng->bhgc", fx, sw)
        st  = st / (sn + 1e-5).unsqueeze(-1)                         # B H G D

        q, k, v = self.to_q(st), self.to_k(st), self.to_v(st)
        attn    = self.dropout(self.softmax(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale))
        out     = torch.matmul(attn, v)                              # B H G D

        out_nodes = rearrange(
            torch.einsum("bhgc,bhng->bhnc", out, sw),
            "b h n d -> b n (h d)")
        return self.to_out(out_nodes)


class _TransolverBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float,
                 mlp_ratio: int = 2, slice_num: int = 64,
                 last_layer: bool = False, out_dim: int = 3):
        super().__init__()
        self.last_layer = last_layer
        self.ln1  = nn.LayerNorm(hidden_dim)
        self.attn = _PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num)
        self.ln2  = nn.LayerNorm(hidden_dim)
        self.mlp  = _MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                         n_layers=0, res=False)
        if last_layer:
            self.ln3  = nn.LayerNorm(hidden_dim)
            self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.ln1(x)) + x
        x = self.mlp (self.ln2(x)) + x
        if self.last_layer:
            x = self.proj(self.ln3(x))
        return x


@dataclass
class _TransolverCoreConfig:
    window_size: int   = 5
    hidden_dim:  int   = 128
    n_layers:    int   = 6
    n_heads:     int   = 8
    slice_num:   int   = 64
    mlp_ratio:   int   = 2
    dropout:     float = 0.0
    n_rollout:   int   = 5


class _TransolverCore(nn.Module):
    """
    Autoregressive Transolver backbone.

    predict_step(pos [B,N,3], v_window [B,N,W,3])  →  v_next [B,N,3]
    rollout(pos, v_init [B,N,W,3], n_steps)         →  preds  [B,n_steps,N,3]

    All inputs and outputs are in normalised space.
    """

    def __init__(self, cfg: _TransolverCoreConfig):
        super().__init__()
        c_in             = 3 + cfg.window_size * 3
        self.cfg         = cfg
        self.encoder     = _MLP(c_in, cfg.hidden_dim * 2, cfg.hidden_dim,
                                n_layers=0, res=False)
        self.placeholder = nn.Parameter(torch.rand(cfg.hidden_dim) / cfg.hidden_dim)
        self.blocks      = nn.ModuleList([
            _TransolverBlock(
                hidden_dim=cfg.hidden_dim, num_heads=cfg.n_heads,
                dropout=cfg.dropout, mlp_ratio=cfg.mlp_ratio,
                slice_num=cfg.slice_num,
                last_layer=(i == cfg.n_layers - 1), out_dim=3,
            )
            for i in range(cfg.n_layers)
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def predict_step(self, pos: torch.Tensor,
                     v_window: torch.Tensor) -> torch.Tensor:
        """pos [B,N,3], v_window [B,N,W,3]  →  v_next [B,N,3]"""
        B, N = pos.shape[:2]
        feat = torch.cat([pos, v_window.reshape(B, N, -1)], dim=-1)
        h    = self.encoder(feat) + self.placeholder
        for block in self.blocks:
            h = block(h)
        return h

    def rollout(self, pos: torch.Tensor, v_init: torch.Tensor,
                n_steps: int = None) -> torch.Tensor:
        """
        pos    [B,N,3]
        v_init [B,N,W,3]  normalised initial window, newest last
        returns preds [B,n_steps,N,3]  normalised
        """
        n_steps = n_steps or self.cfg.n_rollout
        window  = v_init.clone()
        preds   = []
        for _ in range(n_steps):
            v_next = self.predict_step(pos, window)
            preds.append(v_next)
            window = torch.cat([window[:, :, 1:], v_next.unsqueeze(2)], dim=2)
        return torch.stack(preds, dim=1)


# ── Submission wrapper ────────────────────────────────────────────────────────

class TransolverAR(nn.Module):
    """
    Submission model for GRaM @ ICLR 2026.

    Instantiates without arguments; loads weights and norm stats from the
    same directory as this file.

    Callable interface:
        forward(t, pos, idcs_airfoil, velocity_in) -> velocity_out
          t             (batch, 10)          — unused
          pos           (batch, N, 3)        — raw mesh coordinates
          idcs_airfoil  list[Tensor]         — unused
          velocity_in   (batch, 5, N, 3)     — raw input velocity window
          returns       (batch, 5, N, 3)     — predicted next 5 steps (raw)
    """

    def __init__(self):
        super().__init__()
        cfg = _TransolverCoreConfig(
            window_size=5, hidden_dim=256, n_layers=6, n_heads=8,
            slice_num=64, mlp_ratio=2, dropout=0.1, n_rollout=5,
        )
        self._inner = _TransolverCore(cfg)

        root = Path(__file__).parent
        sd = torch.load(root / "state_dict.pt", map_location="cpu", weights_only=True)
        self._inner.load_state_dict(sd)

        ns = torch.load(root / "norm_stats.pt", map_location="cpu", weights_only=False)
        self.register_buffer("pos_mean", ns["pos_mean"])  # (3,)
        self.register_buffer("pos_std",  ns["pos_std"])
        self.register_buffer("vel_mean", ns["vel_mean"])
        self.register_buffer("vel_std",  ns["vel_std"])

        self.eval()

    def forward(
        self,
        t:             torch.Tensor,
        pos:           torch.Tensor,
        idcs_airfoil,
        velocity_in:   torch.Tensor,
    ) -> torch.Tensor:
        pos_n   = (pos - self.pos_mean) / self.pos_std           # (B, N, 3)
        v_n     = (velocity_in.permute(0, 2, 1, 3)               # (B, N, 5, 3)
                   - self.vel_mean) / self.vel_std
        preds_n = self._inner.rollout(pos_n, v_n, n_steps=5)    # (B, 5, N, 3)
        return preds_n * self.vel_std + self.vel_mean
