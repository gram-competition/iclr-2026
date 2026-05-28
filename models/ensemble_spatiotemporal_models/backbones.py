import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeEncoder(nn.Module):
    """Encode raw edge features [rel_pos(3) + dist(1)] into hidden_dim."""
    def __init__(self, hidden_dim: int, edge_in: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, rel_pos: torch.Tensor, dists: torch.Tensor) -> torch.Tensor:
        raw = torch.cat([rel_pos, dists.unsqueeze(-1)], dim=-1)
        return self.net(raw)


class GATLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1,
                 edge_dim: int | None = None):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim)

        self.edge_proj = nn.Linear(edge_dim, heads) if edge_dim else None

        self.attn_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, neighbors, edge_feat):
        """
        x:         (N, D)
        neighbors: (N, k)
        edge_feat: (N, k, edge_dim) or None
        """
        N, k = neighbors.shape
        H, d = self.heads, self.head_dim

        q = self.W_q(x).view(N, 1, H, d)                        # (N,1,H,d)
        k_all = self.W_k(x)                                      # (N,D)
        v_all = self.W_v(x)

        k_nbr = k_all[neighbors].view(N, k, H, d)               # (N,k,H,d)
        v_nbr = v_all[neighbors].view(N, k, H, d)

        # GATv2: LeakyReLU on sum then dot with learned vec → simplified
        attn = (q * k_nbr).sum(-1) / self.scale                  # (N,k,H)

        if self.edge_proj is not None and edge_feat is not None:
            attn = attn + self.edge_proj(edge_feat)               # edge bias

        attn = F.softmax(attn, dim=1)                             # softmax over k
        attn = self.attn_drop(attn)

        out = (attn.unsqueeze(-1) * v_nbr).sum(dim=1)            # (N,H,d)
        out = out.reshape(N, -1)
        out = self.W_o(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


class GATBackbone(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 6,
                 heads: int = 4, dropout: float = 0.1,
                 use_edge_feat: bool = True):
        super().__init__()
        edge_dim = hidden_dim if use_edge_feat else None
        self.edge_enc = EdgeEncoder(hidden_dim) if use_edge_feat else None
        self.layers = nn.ModuleList([
            GATLayer(hidden_dim, heads, dropout, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, neighbors, rel_pos, dists):
        ef = self.edge_enc(rel_pos, dists) if self.edge_enc is not None else None
        for layer in self.layers:
            x = layer(x, neighbors, ef)
        return x


# -----------------------------------------------------------------------
# 2. MeshGraphNet-style (encode-process-decode with edge updates)
# -----------------------------------------------------------------------

class MGNLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.edge_norm = nn.LayerNorm(dim)
        self.node_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, neighbors, edge_feat):
        """
        x:         (N, D)
        neighbors: (N, k)
        edge_feat: (N, k, D)
        """
        N, k, D = edge_feat.shape
        x_i = x.unsqueeze(1).expand(-1, k, -1)            # (N,k,D)  center
        x_j = x[neighbors]                                 # (N,k,D)  neighbor

        # --- edge update (residual) ---
        inp = torch.cat([x_i, x_j, edge_feat], dim=-1)     # (N,k,3D)
        edge_feat = self.edge_norm(edge_feat + self.drop(self.edge_mlp(inp)))

        # --- node update: mean-aggregate edge messages ---
        msg = edge_feat.mean(dim=1)                         # (N,D)
        x = self.node_norm(x + self.drop(self.node_mlp(torch.cat([x, msg], dim=-1))))

        return x, edge_feat


class MeshGraphNetBackbone(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 6,
                 dropout: float = 0.1, **_kw):
        super().__init__()
        self.edge_enc = EdgeEncoder(hidden_dim)
        self.layers = nn.ModuleList([
            MGNLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, neighbors, rel_pos, dists):
        ef = self.edge_enc(rel_pos, dists)
        for layer in self.layers:
            x, ef = layer(x, neighbors, ef)
        return x


# -----------------------------------------------------------------------
# 3. Graph Transformer (local attention with edge-conditioned bias)
# -----------------------------------------------------------------------

class GraphTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1,
                 edge_dim: int | None = None):
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

        # edge-conditioned value bias
        if self.W_e is not None and edge_feat is not None:
            v_nbr = v_nbr + self.W_e(edge_feat).view(N, k, H, d)

        attn = (q * k_nbr).sum(-1) / self.scale   # (N,k,H)
        attn = F.softmax(attn, dim=1)
        attn = self.attn_drop(attn)

        out = (attn.unsqueeze(-1) * v_nbr).sum(1).reshape(N, -1)
        out = self.W_o(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


class GraphTransformerBackbone(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 6,
                 heads: int = 4, dropout: float = 0.1,
                 use_edge_feat: bool = True):
        super().__init__()
        edge_dim = hidden_dim if use_edge_feat else None
        self.edge_enc = EdgeEncoder(hidden_dim) if use_edge_feat else None
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, heads, dropout, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, neighbors, rel_pos, dists):
        ef = self.edge_enc(rel_pos, dists) if self.edge_enc is not None else None
        for layer in self.layers:
            x = layer(x, neighbors, ef)
        return x


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------

BACKBONES = {
    "gat": GATBackbone,
    "meshgraphnet": MeshGraphNetBackbone,
    "graph_transformer": GraphTransformerBackbone,
}


def build_backbone(name: str, **kwargs) -> nn.Module:
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {list(BACKBONES)}")
    return BACKBONES[name](**kwargs)
