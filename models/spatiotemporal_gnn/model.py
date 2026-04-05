"""
Architecture: Graph Transformer backbone with Fourier positional encoding,
temporal self-attention head, and residual velocity prediction on 100k-point
3D airfoil meshes.

"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import numpy as np
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# -----------------------------------------------------------------------
# Graph utilities
# -----------------------------------------------------------------------

def knn_graph(pos: torch.Tensor, k: int):
    """Build k-NN graph in dense format using scipy.cKDTree."""
    if _HAS_SCIPY:
        pos_np = pos.detach().cpu().float().contiguous().numpy()
        tree = cKDTree(pos_np)
        dists_np, idx_np = tree.query(pos_np, k=k + 1, workers=-1)
        idx_np = idx_np[:, 1:]
        dists_np = dists_np[:, 1:]
        nn_idx = torch.from_numpy(np.asarray(idx_np)).to(device=pos.device, dtype=torch.long)
        dists_t = torch.from_numpy(np.asarray(dists_np)).to(device=pos.device, dtype=pos.dtype)
        rel_pos = pos[nn_idx] - pos.unsqueeze(1)
        return nn_idx, rel_pos, dists_t

    pw = torch.cdist(pos, pos)
    _, nn_idx = pw.topk(k + 1, largest=False, dim=-1)
    nn_idx = nn_idx[:, 1:]
    rel_pos = pos[nn_idx] - pos.unsqueeze(1)
    dists = rel_pos.norm(dim=-1)
    return nn_idx, rel_pos, dists


# -----------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mult, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EdgeEncoder(nn.Module):
    def __init__(self, hidden_dim, edge_in=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_in, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, rel_pos, dists):
        raw = torch.cat([rel_pos, dists.unsqueeze(-1)], dim=-1)
        return self.net(raw)


class FourierPosEnc(nn.Module):
    def __init__(self, in_dim=3, num_freqs=8):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)
        self.out_dim = in_dim + in_dim * num_freqs * 2

    def forward(self, x):
        proj = x.unsqueeze(-1) * self.freqs
        return torch.cat([x, proj.sin().flatten(-2),
                          proj.cos().flatten(-2)], dim=-1)


# -----------------------------------------------------------------------
# Graph Transformer backbone
# -----------------------------------------------------------------------

class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, edge_dim=None):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_e = nn.Linear(edge_dim, dim, bias=False) if edge_dim else None
        self.W_o = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, neighbors, edge_feat):
        N, k = neighbors.shape
        H, d = self.heads, self.head_dim

        q = self.W_q(x).view(N, 1, H, d)
        k_nbr = self.W_k(x)[neighbors].view(N, k, H, d)
        v_nbr = self.W_v(x)[neighbors].view(N, k, H, d)

        if self.W_e is not None and edge_feat is not None:
            v_nbr = v_nbr + self.W_e(edge_feat).view(N, k, H, d)

        attn = (q * k_nbr).sum(-1) / self.scale
        attn = F.softmax(attn, dim=1)
        attn = self.attn_drop(attn)

        out = (attn.unsqueeze(-1) * v_nbr).sum(1).reshape(N, -1)
        out = self.W_o(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


class GraphTransformerBackbone(nn.Module):
    def __init__(self, hidden_dim, num_layers=6, heads=4, dropout=0.1):
        super().__init__()
        self.edge_enc = EdgeEncoder(hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, heads, dropout, edge_dim=hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, neighbors, rel_pos, dists):
        ef = self.edge_enc(rel_pos, dists)
        for layer in self.layers:
            x = layer(x, neighbors, ef)
        return x


# -----------------------------------------------------------------------
# Temporal attention head
# -----------------------------------------------------------------------

class TemporalAttentionHead(nn.Module):
    def __init__(self, hidden_dim, t_out=5, temporal_dim=None,
                 num_heads=4, num_attn_layers=2, dropout=0.1):
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
        self.time_pe = nn.Parameter(torch.randn(1, t_out, self.td) * 0.02)

    def forward(self, x):
        N = x.size(0)
        h = self.proj(x).view(N, self.t_out, self.td)
        h = h + self.time_pe
        for attn, ff, n1, n2 in zip(
            self.attn_layers, self.ff_layers, self.norm1, self.norm2
        ):
            res = h
            h, _ = attn(h, h, h)
            h = n1(res + h)
            h = n2(h + ff(h))
        return h


# -----------------------------------------------------------------------
# Main model (no-arg constructor for competition)
# -----------------------------------------------------------------------

class SpatioTemporalGNN(nn.Module):
    """Competition-ready SpatioTemporalGNN with hardcoded hyperparameters.

    Trained with:
        backbone=graph_transformer, hidden_dim=256, num_layers=10, heads=8,
        k=24, use_fourier=True, num_fourier=8, dropout=0.05, t_in=5, t_out=5
    """

    K = 24
    T_IN = 5
    T_OUT = 5
    HIDDEN_DIM = 256
    NUM_LAYERS = 10
    HEADS = 8
    DROPOUT = 0.05
    NUM_FOURIER = 8

    def __init__(self):
        super().__init__()
        self.k = self.K

        self.pos_enc = FourierPosEnc(3, self.NUM_FOURIER)
        pos_dim = self.pos_enc.out_dim

        in_dim = pos_dim + self.T_IN * 3 + 1
        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, self.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.LayerNorm(self.HIDDEN_DIM),
        )

        self.spatial = GraphTransformerBackbone(
            self.HIDDEN_DIM, self.NUM_LAYERS, self.HEADS, self.DROPOUT,
        )

        self.temporal = TemporalAttentionHead(
            self.HIDDEN_DIM, t_out=self.T_OUT, temporal_dim=self.HIDDEN_DIM,
            num_heads=self.HEADS, dropout=self.DROPOUT,
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(self.HIDDEN_DIM, 3),
        )

        weights_path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state)
            self.to("cpu")

    def _forward_single(self, pos, vel_in, airfoil_idx):
        N = pos.size(0)
        mask = pos.new_zeros(N, 1)
        mask[airfoil_idx] = 1.0

        neighbors, rel_pos, dists = knn_graph(pos, self.k)

        pos_feat = self.pos_enc(pos)
        vel_flat = vel_in.permute(1, 0, 2).reshape(N, -1)
        x = torch.cat([pos_feat, vel_flat, mask], dim=-1)
        x = self.node_enc(x)

        x = self.spatial(x, neighbors, rel_pos, dists)
        x = self.temporal(x)
        x = self.decoder(x)

        last_vel = vel_in[-1]
        x = x + last_vel.unsqueeze(1)

        x_full = x.permute(1, 0, 2)
        x_full[:, airfoil_idx] = 0.0
        return x_full

    def forward(self, t, pos, idcs_airfoil, velocity_in):
        B = velocity_in.size(0)
        outputs = []
        for b in range(B):
            out = self._forward_single(
                pos[b], velocity_in[b],
                idcs_airfoil[b].to(pos.device),
            )
            outputs.append(out)
        return torch.stack(outputs)
