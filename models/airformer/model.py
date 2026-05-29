"""
FlowTransolver v11 — Anchor-Slice Architecture
"""


import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from einops import rearrange

import numpy as np
from scipy.spatial import cKDTree
_HAS_SCIPY = True

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_



# ═══════════════════════════════════════════════════════════════════════════════
# Anchor selection: stratified spatial coverage
# ═══════════════════════════════════════════════════════════════════════════════

def select_anchors_stratified(pos, n_anchors, surface_idx=None,
                               surface_frac=0.30, seed=None):
    """Sample M anchors with spatial coverage + guaranteed surface representation.

    Pure random sampling under-represents the surface boundary layer (where
    error is highest).  Pure FPS is too slow at N=100k.  We use a hybrid:
        - Pick surface_frac of anchors from the surface
        - Pick the rest from non-surface points (uniform random)

    Returns:
        anchor_idx: (M,) LongTensor — indices into pos
    """
    N = pos.shape[0]
    device = pos.device
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    n_surf = int(n_anchors * surface_frac) if surface_idx is not None else 0
    n_field = n_anchors - n_surf

    out = []
    if n_surf > 0 and surface_idx is not None and surface_idx.numel() > 0:
        n_take = min(n_surf, surface_idx.shape[0])
        perm = torch.randperm(surface_idx.shape[0], generator=g, device=device)
        out.append(surface_idx[perm[:n_take]])
        n_field = n_anchors - n_take

    # Non-surface pool
    if surface_idx is not None:
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[surface_idx] = False
        field_idx = mask.nonzero(as_tuple=False).squeeze(-1)
    else:
        field_idx = torch.arange(N, device=device)

    n_take = min(n_field, field_idx.shape[0])
    perm = torch.randperm(field_idx.shape[0], generator=g, device=device)
    out.append(field_idx[perm[:n_take]])

    return torch.cat(out, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# k-NN utilities (cKDTree-based, only concept retained from graph methods)
# ═══════════════════════════════════════════════════════════════════════════════

def query_anchors(query_pos, anchor_pos, k):
    """For each query point, find k nearest anchors.

    Args:
        query_pos:  (Nq, 3)
        anchor_pos: (M, 3)
        k:          number of anchors per query
    Returns:
        anchor_idx: (Nq, k)
        dists:      (Nq, k)
    """
    device = query_pos.device
    if _HAS_SCIPY:
        a = anchor_pos.detach().cpu().float().numpy()
        q = query_pos.detach().cpu().float().numpy()
        tree = cKDTree(a)
        d, i = tree.query(q, k=k, workers=-1)
        if k == 1:
            i = i[:, None]
            d = d[:, None]
        anchor_idx = torch.from_numpy(i.astype(np.int64)).to(device)
        dists = torch.from_numpy(d.astype(np.float32)).to(device)
    else:
        pw = torch.cdist(query_pos, anchor_pos)
        dists, anchor_idx = pw.topk(k, largest=False, dim=-1)
    return anchor_idx, dists


def aggregate_to_anchors(full_features, anchor_idx_in_full, full_pos, k_per_anchor):
    """Initialize anchor features by mean-pooling k nearest full-res points."""
    anchor_pos = full_pos[anchor_idx_in_full]
    nn_idx, _ = query_anchors(anchor_pos, full_pos, k_per_anchor)
    return full_features[nn_idx].mean(dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# PhysicsAttention: multi-head attention with slice-based sparse attention pattern
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsAttention(nn.Module):
    """Physics-Attention for irregular meshes (point clouds).

    Args:
        dim:       hidden dimension of input/output features.
        heads:     number of attention heads.
        dim_head:  dimension per head (inner_dim = heads * dim_head).
        dropout:   dropout rate on attention weights and output projection.
        slice_num: G — number of learnable physical-state slices.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Learnable temperature per head — controls sharpness of slice assignment
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

        # Dual projections: x_mid drives slice routing, fx_mid carries features
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)

        # Slice assignment: per-head, per-point → soft distribution over G slices
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        # Standard QKV on slice tokens
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (B, M, C) per-point features.
        Returns:
            (B, M, C) updated features after physics-aware global attention.
        """
        B, M, C = x.shape
        H, D = self.heads, self.dim_head

        # ── (1) Slice: aggregate M points → G slice tokens ──
        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, M, H, D)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # (B, H, M, D)
        x_mid = (
            self.in_project_x(x)
            .reshape(B, M, H, D)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # (B, H, M, D)

        # Soft assignment weights with learnable temperature
        slice_weights = self.softmax(
            self.in_project_slice(x_mid)
            / torch.clamp(self.temperature, min=0.1, max=5.0)
        )  # (B, H, M, G)

        slice_norm = slice_weights.sum(dim=2)  # (B, H, G)

        # Weighted aggregation into slice tokens
        slice_token = torch.einsum("bhmc,bhmg->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            slice_norm[:, :, :, None].expand_as(slice_token) + 1e-5
        )  # (B, H, G, D)

        # ── (2) Attend: standard self-attention among G slice tokens ──
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        out_slice = torch.matmul(attn, v)  # (B, H, G, D)

        # ── (3) Deslice: broadcast back to M points ──
        out_x = torch.einsum("bhgc,bhmg->bhmc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h m d -> b m (h d)")
        return self.to_out(out_x)


# ═══════════════════════════════════════════════════════════════════════════════
# Position encoding
# ═══════════════════════════════════════════════════════════════════════════════

class FourierPE(nn.Module):
    """Multi-scale Fourier positional encoding."""

    def __init__(self, num_freq=8):
        super().__init__()
        self.num_freq = num_freq
        self.register_buffer("freqs", 2.0 ** torch.arange(num_freq).float() * math.pi)

    @property
    def out_dim(self):
        return 3 + 6 * self.num_freq

    def forward(self, pos):
        proj = pos.unsqueeze(-1) * self.freqs
        return torch.cat([pos, proj.sin().flatten(-2), proj.cos().flatten(-2)], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Anchor PhysicsAttention block
# ═══════════════════════════════════════════════════════════════════════════════

class AnchorPhysicsBlock(nn.Module):
    """Pre-norm PhysicsAttention on anchors. At M=4096 and G=64 slices,
    the slice mechanism has ~64 points/slice ratio."""

    def __init__(self, d_model, n_heads=8, slice_num=64, dropout=0.05, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PhysicsAttention(
            d_model, heads=n_heads, dim_head=d_model // n_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Per-point readout: gather from anchors via inverse-distance attention pool
# ═══════════════════════════════════════════════════════════════════════════════

class AnchorReadout(nn.Module):
    """Per-point feature gathering from k nearest anchors via
    inverse-distance attention pooling (NOT multi-head attention).

    Skip connection with per-point local features guarantees gradient flow."""

    def __init__(self, d_anchor, d_local, d_out, dropout=0.05):
        super().__init__()
        # Learnable temperature for inverse-distance weighting
        self.log_temp = nn.Parameter(torch.tensor(0.0))

        self.combine = nn.Sequential(
            nn.LayerNorm(d_anchor + d_local),
            nn.Linear(d_anchor + d_local, d_out * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out * 2, d_out),
        )

    def forward(self, anchor_feats, local_feats, anchor_idx_per_point, dists):
        """
        anchor_feats:          (M, d_anchor)
        local_feats:           (N, d_local)
        anchor_idx_per_point:  (N, k)
        dists:                 (N, k)
        Returns:               (N, d_out)
        """
        # Inverse-distance softmax weights with learned temperature
        temp = torch.exp(self.log_temp).clamp(min=0.5, max=200.0)
        logits = -temp * dists                                    # (N, k)
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)         # (N, k, 1)

        # Gather and pool
        gathered = anchor_feats[anchor_idx_per_point]             # (N, k, d_a)
        pooled = (weights * gathered).sum(dim=1)                  # (N, d_a)

        # Skip connection with local features
        combined = torch.cat([pooled, local_feats], dim=-1)
        return self.combine(combined)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-step temporal head
# ═══════════════════════════════════════════════════════════════════════════════

class PerStepHead(nn.Module):
    """Independent decoder MLPs per output timestep.

    Each step has its own bias vector and decoder — forces upstream features
    to encode multi-step information that different decoders extract
    differently (no timestep collapse).
    """

    def __init__(self, d_model, t_out=5, dropout=0.05):
        super().__init__()
        self.t_out = t_out
        self.step_biases = nn.Parameter(torch.randn(t_out, d_model) * 0.02)
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2), nn.GELU(),
                nn.Linear(d_model // 2, 3),
            )
            for _ in range(t_out)
        ])

    def forward(self, x):
        outs = []
        for t in range(self.t_out):
            outs.append(self.decoders[t](x + self.step_biases[t]))
        return torch.stack(outs, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════════════════════

class AirFormer(nn.Module):
    """Anchor-Slice architecture for full-resolution airflow prediction."""

    # ── Defaults ──
    D_LOCAL       = 128
    D_ANCHOR      = 256
    N_ANCHORS     = 4096
    N_LAYERS      = 12
    N_HEADS       = 16
    SLICE_NUM     = 128
    K_QUERY       = 8
    K_AGG         = 16
    SURFACE_FRAC  = 0.50
    DROPOUT       = 0.05
    NUM_FOURIER   = 8
    T_IN          = 5
    T_OUT         = 5

    def __init__(self, **kwargs):
        super().__init__()
        cfg = {
            "d_local": self.D_LOCAL, "d_anchor": self.D_ANCHOR,
            "n_anchors": self.N_ANCHORS, "n_layers": self.N_LAYERS,
            "n_heads": self.N_HEADS, "slice_num": self.SLICE_NUM,
            "k_query": self.K_QUERY, "k_agg": self.K_AGG,
            "surface_frac": self.SURFACE_FRAC, "dropout": self.DROPOUT,
            "num_fourier": self.NUM_FOURIER,
            "t_in": self.T_IN, "t_out": self.T_OUT,
        }
        cfg.update(kwargs)

        # Store config as attributes
        self.d_local = cfg["d_local"]
        self.d_anchor = cfg["d_anchor"]
        self.n_anchors = cfg["n_anchors"]
        self.k_query = cfg["k_query"]
        self.k_agg = cfg["k_agg"]
        self.surface_frac = cfg["surface_frac"]
        self.t_in = cfg["t_in"]
        self.t_out = cfg["t_out"]
        # Legacy alias for training scripts that expect d_model
        self.d_model = cfg["d_anchor"]

        # Fourier PE
        self.pos_enc = FourierPE(cfg["num_fourier"])
        in_dim = self.pos_enc.out_dim + cfg["t_in"] * 3 + 1

        # Per-point local encoder
        self.local_enc = nn.Sequential(
            nn.Linear(in_dim, cfg["d_local"]), nn.GELU(),
            nn.Linear(cfg["d_local"], cfg["d_local"]),
            nn.LayerNorm(cfg["d_local"]),
        )

        # Project local → anchor dim for anchor init
        self.local_to_anchor = nn.Sequential(
            nn.Linear(cfg["d_local"], cfg["d_anchor"]), nn.GELU(),
            nn.Linear(cfg["d_anchor"], cfg["d_anchor"]),
            nn.LayerNorm(cfg["d_anchor"]),
        )

        # PhysicsAttention layers on anchors
        self.anchor_layers = nn.ModuleList([
            AnchorPhysicsBlock(
                cfg["d_anchor"], cfg["n_heads"],
                cfg["slice_num"], cfg["dropout"],
            )
            for _ in range(cfg["n_layers"])
        ])
        self.anchor_norm = nn.LayerNorm(cfg["d_anchor"])

        # Per-point readout
        self.readout = AnchorReadout(
            cfg["d_anchor"], cfg["d_local"], cfg["d_anchor"],
            dropout=cfg["dropout"],
        )

        # Per-step temporal head
        self.head = PerStepHead(
            cfg["d_anchor"], cfg["t_out"], dropout=cfg["dropout"],
        )

        self._init_weights()
        self._try_load_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _try_load_weights(self):
        p = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.isfile(p):
            print(f"Loading pretrained weights from {p}")
            self.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))

    def _run_anchor_layer(self, layer, x):
        return layer(x)

    def _forward_single(self, pos, vel_in, airfoil_idx):
        """Process one sample."""
        N = pos.shape[0]
        device = pos.device
        use_ckpt = self.training

        # ── Per-point local features ──
        pos_feat = self.pos_enc(pos)
        vel_flat = vel_in.permute(1, 0, 2).reshape(N, -1)
        mask = torch.zeros(N, 1, device=device)
        mask[airfoil_idx] = 1.0
        local_in = torch.cat([pos_feat, vel_flat, mask], dim=-1)
        local_feats = self.local_enc(local_in)                    # (N, d_local)

        # ── Anchor selection ──
        with torch.no_grad():
            anchor_idx_in_full = select_anchors_stratified(
                pos, self.n_anchors,
                surface_idx=airfoil_idx,
                surface_frac=self.surface_frac,
            )
            anchor_pos = pos[anchor_idx_in_full]
            anchor_idx_per_point, query_dists = query_anchors(
                pos, anchor_pos, self.k_query,
            )

        # ── Initialize anchor features ──
        anchor_init = self.local_to_anchor(local_feats)
        anchor_feats = aggregate_to_anchors(
            anchor_init, anchor_idx_in_full, pos, self.k_agg,
        )                                                         # (M, d_anchor)

        # ── PhysicsAttention on anchors ──
        anchor_feats = anchor_feats.unsqueeze(0)                  # (1, M, d_anchor)
        for layer in self.anchor_layers:
            if use_ckpt:
                anchor_feats = checkpoint(
                    self._run_anchor_layer, layer, anchor_feats,
                    use_reentrant=False,
                )
            else:
                anchor_feats = layer(anchor_feats)
        anchor_feats = self.anchor_norm(anchor_feats).squeeze(0)  # (M, d_anchor)

        # ── Per-point readout ──
        x = self.readout(
            anchor_feats, local_feats,
            anchor_idx_per_point, query_dists,
        )                                                         # (N, d_anchor)

        # ── Per-step head ──
        offsets = self.head(x)                                    # (N, 5, 3)

        # ── Residual + no-slip ──
        v_last = vel_in[-1]
        pred = offsets + v_last.unsqueeze(1)
        pred[airfoil_idx] = 0.0
        return pred.permute(1, 0, 2)

    def forward(self, t, pos, idcs_airfoil, velocity_in):
        """Competition interface."""
        B = velocity_in.shape[0]
        outs = []
        for b in range(B):
            outs.append(self._forward_single(
                pos[b], velocity_in[b], idcs_airfoil[b].to(pos.device),
            ))
        return torch.stack(outs)