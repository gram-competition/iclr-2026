"""
GatedEGNOMeanResModel — self-contained submission for GRaM @ ICLR 2026.

Architecture: EGNO-style E(n)-equivariant GNN with spectral temporal mixing and
a one-shot Δv decoder referenced against the temporal mean of the input frames
(Reynolds decomposition).

References:
    - Satorras, Hoogeboom, Welling. "E(n) Equivariant Graph Neural Networks."
      ICML 2021. arXiv:2102.09844.
      Base message-passing layer + edge-inference gating (Eq. 3.3).
    - Xu, Han, Lou, Kossaifi, Ramanathan, Azizzadenesheli, Leskovec, Ermon,
      Anandkumar. "Equivariant Graph Neural Operator for Modeling 3D
      Dynamics." ICML 2024. arXiv:2401.11037.
      Block structure + spectral temporal mixing (TimeConv / TimeConvX).
    - Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar.
      "Fourier Neural Operator for Parametric Partial Differential
      Equations." ICLR 2021. arXiv:2010.08895.
      Underlying `SpectralConv` (from `neuraloperator` library).
    - Vaswani et al. "Attention Is All You Need." NeurIPS 2017.
      arXiv:1706.03762. Sinusoidal positional embedding for frame indices.
    - Reynolds, O. "On the Dynamical Theory of Incompressible Viscous Fluids
      and the Determination of the Criterion." Philosophical Transactions of
      the Royal Society of London A, 1895.
      Classical mean/fluctuation decomposition motivating the residual target
      `Δv = v_out − mean(v_in)` used by this model.



Interface:
    model = GatedEGNOMeanResModel()
    velocity_out = model(t, pos, idcs_airfoil, velocity_in)
    # shapes: (B, 5, N, 3)

External dependencies:
    torch, torch_scatter, scipy, neuraloperator
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_scatter import scatter_add, scatter_mean
from neuralop.layers.spectral_convolution import SpectralConv
from scipy.spatial import cKDTree


# ═════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░░░░░░░░░  DATA PROCESSING CODE  ░░░░░░░░░░░░░░░░░░░░
# ═════════════════════════════════════════════════════════════════
# Feature computation from raw (pos, idcs_airfoil).

def _chunked_nn_search(pts: Tensor, surface_pts: Tensor, chunk: int = 2000
                       ) -> tuple[Tensor, Tensor]:
    """Nearest surface point per query. One distance matrix pass, chunked.

    Args:
        pts:         (N, 3) query points
        surface_pts: (M, 3) airfoil surface points

    Returns:
        min_dists:   (N,)   minimum distance to any surface point
        nearest_pts: (N, 3) coordinates of nearest surface point
    """
    min_dists, nearest_pts = [], []
    for i in range(0, pts.shape[0], chunk):
        d = torch.cdist(pts[i:i + chunk], surface_pts)  # (chunk, M)
        mn, idx = d.min(dim=1)
        min_dists.append(mn)
        nearest_pts.append(surface_pts[idx])
    return torch.cat(min_dists), torch.cat(nearest_pts)


@torch.no_grad()
def compute_features_for_sample(
    pos_b: Tensor,
    idcs_b: Tensor,
    udf_d_max: float = 0.5,
    knn_k: int = 16,
) -> tuple[Tensor, Tensor]:
    """Compute (point_features, knn_graph) for a single sample on the fly.

    Args:
        pos_b:  (N, 3) coordinates
        idcs_b: (M,)   airfoil surface index tensor

    Returns:
        point_features: (N, 4) = [udf_truncated(1), udf_gradient(3)]
        knn_graph:      (N, knn_k) int64 neighbor indices
    """
    device = pos_b.device
    # Move airfoil indices to same device as pos before indexing (critical
    # when model is on CUDA but idcs_airfoil came in as CPU list of tensors).
    idcs_b = idcs_b.to(device)
    surface = pos_b[idcs_b]  # (M, 3)

    # UDF + UDF gradient via one chunked cdist pass (stays on pos device).
    min_dists, nearest = _chunked_nn_search(pos_b, surface)
    udf_trunc = min_dists.clamp(max=udf_d_max).unsqueeze(1)       # (N, 1)
    udf_grad = F.normalize(nearest - pos_b, dim=1)                # (N, 3)
    point_features = torch.cat([udf_trunc, udf_grad], dim=1)      # (N, 4)

    # kNN graph: scipy cKDTree is CPU-only, round-trip.
    pos_np = pos_b.detach().cpu().numpy()
    tree = cKDTree(pos_np)
    _, idx = tree.query(pos_np, k=knn_k + 1)                      # +1 includes self
    knn_graph = torch.from_numpy(idx[:, 1:]).long().to(device)     # (N, knn_k)

    return point_features, knn_graph


# ═════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░░░░░░░░  MODEL ARCHITECTURE CODE  ░░░░░░░░░░░░░░░░░░
# ═════════════════════════════════════════════════════════════════
# Base EGNN layer + equivariant decoder.

class FixedEGNNLayer(nn.Module):
    """EGNN layer; 5 velocity frames enter as edge-projected invariant scalars.

    "Fixed": an earlier decoder zero-initialized its 2-layer MLP heads to
    force Δv = 0 at init, creating a dying-neurons dead zone. Fixed by
    using PyTorch default init everywhere (see EquivariantDecoder).
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0,
                 update_coords: bool = False):
        super().__init__()
        self.update_coords = update_coords
        # a_ij: dist2(1) + vel_proj_src(5) + vel_proj_dst(5) = 11 invariant scalars
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden_dim + 11, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: Tensor, x: Tensor, vel_all: Tensor, edge_index: Tensor
                ) -> tuple[Tensor, Tensor, Tensor]:
        # Equation numbers below refer to Satorras, Hoogeboom, Welling (2021),
        # "E(n) Equivariant Graph Neural Networks" (arXiv:2102.09844).
        src, dst = edge_index
        rel = x[src] - x[dst]
        dist2 = (rel * rel).sum(-1, keepdim=True)
        dist = dist2.sqrt().clamp(min=1e-8)
        r_hat = rel / dist

        # a_ij edge attributes: velocity frames projected onto the edge
        # direction at each endpoint (invariant under E(n)).
        vel_proj_src = (vel_all[src] * r_hat.unsqueeze(1)).sum(-1)  # (E, 5)
        vel_proj_dst = (vel_all[dst] * r_hat.unsqueeze(1)).sum(-1)  # (E, 5)

        # Eq. 3:  m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2, a_ij)
        m_ij = self.phi_e(
            torch.cat([h[src], h[dst], dist2, vel_proj_src, vel_proj_dst], dim=-1)
        )

        # Eq. 4:  x_i' = x_i + C * sum_{j != i} (x_i - x_j) * phi_x(m_ij)
        # (disabled by default — the flow prediction happens in the velocity
        # channel, so we keep the mesh coordinates fixed.)
        if self.update_coords:
            coord_w = self.phi_x(m_ij)
            delta_x = scatter_mean(rel * coord_w, dst, dim=0, dim_size=x.size(0))
            x = x + delta_x

        # Eq. 5:  m_i = sum_{j in N(i)} m_ij   (unweighted aggregation)
        agg = scatter_add(m_ij, dst, dim=0, dim_size=h.size(0))
        # Eq. 6:  h_i' = phi_h(h_i, m_i)   (with residual + LayerNorm)
        h = h + self.drop(self.phi_h(torch.cat([h, agg], dim=-1)))
        h = self.norm(h)
        return h, x, m_ij


class FixedEGNNGatedLayer(FixedEGNNLayer):
    """EGNN layer with multi-head sigmoid gating on aggregated messages.

    Replaces Eq. 5 of the EGNN paper (unweighted aggregation) with its
    edge-inferred variant from the "Inferring Edges" section (Eq. 7 and
    Eq. 8 of arXiv:2102.09844):

        Eq. 8:  ẽ_ij = phi_inf(m_ij) = sigmoid(Linear(m_ij))
        Eq. 7:  m_i  = sum_{j in N(i)} ẽ_ij * m_ij

    In the paper, ẽ_ij is interpreted as an edge-existence inference — a
    soft indicator of whether an edge should contribute to message
    passing. Here we adapt the same operation as a per-edge gate on a
    fixed k-nearest-neighbour graph where the graph topology is kept, and
    ẽ_ij becomes a learned weight that modulates how
    strongly each neighbour contribute independently. The math is identical; the
    interpretation moves from "should this edge exist?" to "how much
    does this existing edge matter here?".

    The multi-head extension below splits the hidden dimension into H
    slices and applies an independent Eq. 8 gate to each slice.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0,
                 update_coords: bool = False, heads: int = 1):
        super().__init__(hidden_dim, dropout=dropout, update_coords=update_coords)
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by heads={heads}"
            )
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.gate = nn.Linear(hidden_dim, heads)
        # Zero-init so sigmoid(gate) = 0.5 uniformly per head at step 0 —
        # neutral gate, equivalent to plain aggregation.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h: Tensor, x: Tensor, vel_all: Tensor, edge_index: Tensor
                ) -> tuple[Tensor, Tensor, Tensor]:
        src, dst = edge_index
        rel = x[src] - x[dst]
        dist2 = (rel * rel).sum(-1, keepdim=True)
        dist = dist2.sqrt().clamp(min=1e-8)
        r_hat = rel / dist

        vel_proj_src = (vel_all[src] * r_hat.unsqueeze(1)).sum(-1)
        vel_proj_dst = (vel_all[dst] * r_hat.unsqueeze(1)).sum(-1)

        m_ij = self.phi_e(
            torch.cat([h[src], h[dst], dist2, vel_proj_src, vel_proj_dst], dim=-1)
        )

        if self.update_coords:
            coord_w = self.phi_x(m_ij)
            delta_x = scatter_mean(rel * coord_w, dst, dim=0, dim_size=x.size(0))
            x = x + delta_x

        # Eq. 8:  ẽ_ij = sigmoid(Linear(m_ij))   (per head)
        # Eq. 7:  m_i  = sum_j ẽ_ij * m_ij       (per head's hidden slice)
        g = torch.sigmoid(self.gate(m_ij))                      # (E, H)
        g_wide = g.repeat_interleave(self.head_dim, dim=1)       # (E, hidden)
        agg = scatter_add(g_wide * m_ij, dst, dim=0, dim_size=h.size(0))

        h = h + self.drop(self.phi_h(torch.cat([h, agg], dim=-1)))
        h = self.norm(h)
        return h, x, m_ij


class EquivariantDecoder(nn.Module):
    """Δv_i = Σ_t α_i^(t) · v_i^(t) + Σ_j w(m_ij) · (x_j − x_i).

    Invariant-scalar weights applied to equivariant vectors → E(n) equivariant.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Default PyTorch init — zero-init on both MLP layers would kill
        # gradients through the inner layer (see FixedEGNNLayer docstring).
        self.edge_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.vel_gates = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),   # one gate per input velocity frame
        )

    def forward(self, h: Tensor, m_ij: Tensor, x: Tensor,
                vel_all: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        rel = x[src] - x[dst]
        w = self.edge_weight(m_ij)                                 # (E, 1)
        geom_delta = scatter_mean(rel * w, dst, dim=0, dim_size=h.size(0))

        alpha = self.vel_gates(h).unsqueeze(-1)                    # (N, 5, 1)
        vel_combo = (alpha * vel_all).sum(dim=1)                   # (N, 3)

        return vel_combo + geom_delta                              # (N, 3)


# ─────────────────────────────────────────────────────────────────
# EGNO temporal spectral convolutions + block
# ─────────────────────────────────────────────────────────────────

def sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    """Diffusion-style sin/cos positional embedding for frame indices."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeConv(nn.Module):
    """Spectral conv along the time axis on scalar hidden features."""

    def __init__(self, channels: int, n_modes: int = 3):
        super().__init__()
        self.spec = SpectralConv(channels, channels, n_modes=(n_modes,))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, h: Tensor) -> Tensor:
        # (T, BN, C) → (BN, C, T) → spectral → back
        T, BN, C = h.shape
        x = h.permute(1, 2, 0).contiguous()
        x = self.spec(x)
        x = x.permute(2, 0, 1).contiguous()
        return h + self.act(x)


class TimeConvX(nn.Module):
    """Equivariant spectral conv along time on velocity (D=3 absorbed into batch)."""

    def __init__(self, channels: int, n_modes: int = 3):
        super().__init__()
        self.channels = channels
        self.spec = SpectralConv(channels, channels, n_modes=(n_modes,))

    def forward(self, v: Tensor) -> Tensor:
        # (T, BN, D, C) — D=3 spatial components
        T, BN, D, C = v.shape
        assert C == self.channels
        x = v.permute(1, 2, 3, 0).reshape(BN * D, C, T)
        x = self.spec(x)
        x = x.reshape(BN, D, C, T).permute(3, 0, 1, 2).contiguous()
        return v + x


class GatedEGNOBlock(nn.Module):
    """One EGNO block: TimeConv(h) + TimeConvX(vel) + gated EGNN message passing."""

    def __init__(self, hidden_dim: int, n_modes: int = 3,
                 dropout: float = 0.0, update_coords: bool = False,
                 heads: int = 1):
        super().__init__()
        self.time_conv_h = TimeConv(hidden_dim, n_modes=n_modes)
        self.time_conv_v = TimeConvX(channels=1, n_modes=n_modes)
        self.gnn = FixedEGNNGatedLayer(
            hidden_dim, dropout=dropout, update_coords=update_coords, heads=heads,
        )

    def forward(self, h: Tensor, x: Tensor, vel_all: Tensor,
                edge_index: Tensor) -> tuple[Tensor, Tensor]:
        T, BN, C = h.shape

        # Temporal mixing on scalars (invariant).
        h = self.time_conv_h(h)

        # Temporal mixing on velocity (equivariant — D kept as preserved axis).
        vel_td = vel_all.transpose(0, 1).unsqueeze(-1)            # (T, BN, 3, 1)
        vel_td = self.time_conv_v(vel_td)
        vel_all_new = vel_td.squeeze(-1).transpose(0, 1).contiguous()  # (BN, T, 3)

        # Run GNN once on a batched (T·BN)-node graph: replicate edges with
        # cumulative offsets so the T time slices don't share neighbors.
        E = edge_index.shape[1]
        device = edge_index.device
        h_batched = h.reshape(T * BN, C)
        x_batched = x.repeat(T, 1)
        vel_all_batched = vel_all_new.repeat(T, 1, 1)
        offsets = (torch.arange(T, device=device) * BN).repeat_interleave(E)
        edges_batched = edge_index.repeat(1, T) + offsets.unsqueeze(0)

        h_out, _, _ = self.gnn(h_batched, x_batched, vel_all_batched, edges_batched)
        h_new = h_out.reshape(T, BN, C)
        return h_new, vel_all_new


# ─────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────

class GatedEGNOMeanResModel(nn.Module):
    """E(n)-equivariant spectral-temporal GNN for airflow prediction.

    One-shot prediction of 5 output velocity frames from 5 input frames,
    geometry-conditioned via UDF features + kNN message passing, and
    decoded as a residual against the temporal mean of the input window
    (Reynolds decomposition).
    """

    # Hyperparameters are class attributes for no-argument construction.
    hidden_dim = 96      # width of node / edge MLPs and block channels
    depth = 4            # number of EGNO blocks
    n_modes = 3          # Fourier modes along time (max = T//2 + 1 = 3 for T=5)
    heads = 1            # gate heads (1 = scalar gate per EGNN Eq 3.3)
    no_slip_mask = True  # zero velocity at airfoil indices at every output step
    knn_k = 16           # kNN graph connectivity
    udf_d_max = 0.5      # UDF truncation threshold

    per_frame_input_dim = 3   # vel_mag(1) + udf_trunc(1) + |udf_grad|(1)

    def __init__(self):
        super().__init__()

        self.t_emb_dim = max(8, self.hidden_dim // 2)
        self.per_frame_encoder = nn.Sequential(
            nn.Linear(self.per_frame_input_dim + self.t_emb_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            GatedEGNOBlock(
                self.hidden_dim, n_modes=self.n_modes,
                dropout=0.0, update_coords=False, heads=self.heads,
            )
            for _ in range(self.depth)
        ])

        self.decoder = EquivariantDecoder(self.hidden_dim)

        # Auto-load weights if state_dict.pt is present next to this file.
        weights_path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(weights_path):
            state = torch.load(weights_path, weights_only=True, map_location="cpu")
            self.load_state_dict(state)

    # ── Graph-edge index construction (batched kNN → COO) ──
    def _build_edge_index(self, knn_graph: Tensor) -> Tensor:
        B, N, k = knn_graph.shape
        device = knn_graph.device
        batch_offsets = torch.arange(B, device=device) * N
        rows = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
        rows = rows + batch_offsets.view(B, 1, 1)
        cols = knn_graph + batch_offsets.view(B, 1, 1)
        valid = knn_graph >= 0
        return torch.stack([rows[valid], cols[valid]], dim=0)

    # ── Feature computation for a batch (loop over samples) ──
    def _compute_batch_features(self, pos: Tensor, idcs_airfoil: list[Tensor]
                                 ) -> tuple[Tensor, Tensor]:
        batch_feats, batch_knn = [], []
        for b in range(pos.shape[0]):
            feat, knn = compute_features_for_sample(
                pos[b], idcs_airfoil[b],
                udf_d_max=self.udf_d_max, knn_k=self.knn_k,
            )
            batch_feats.append(feat)
            batch_knn.append(knn)
        return torch.stack(batch_feats), torch.stack(batch_knn)

    # ── Forward ──
    def forward(self, t: Tensor, pos: Tensor, idcs_airfoil: list[Tensor],
                velocity_in: Tensor) -> Tensor:
        """
        Args:
            t:            (B, 10) — timestamps (unused by this model)
            pos:          (B, N, 3)
            idcs_airfoil: list[Tensor], length B, each (M_b,)
            velocity_in:  (B, 5, N, 3)

        Returns:
            velocity_out: (B, 5, N, 3)
        """
        point_features, knn_graph = self._compute_batch_features(pos, idcs_airfoil)

        B, T, N, _ = velocity_in.shape
        BN = B * N
        edge_index = self._build_edge_index(knn_graph)
        pos_flat = pos.reshape(BN, 3)

        # Per-frame invariant scalars: [|v|, udf_trunc, |udf_grad|]
        vel_mag = velocity_in.norm(dim=-1).transpose(1, 2)         # (B, N, T)
        udf_t = point_features[..., 0:1]                           # (B, N, 1)
        udf_g = point_features[..., 1:4].norm(dim=-1, keepdim=True)  # (B, N, 1)

        vm = vel_mag.unsqueeze(-1)                                 # (B, N, T, 1)
        ut = udf_t.unsqueeze(2).expand(-1, -1, T, -1)              # (B, N, T, 1)
        ug = udf_g.unsqueeze(2).expand(-1, -1, T, -1)              # (B, N, T, 1)
        per_frame = torch.cat([vm, ut, ug], dim=-1)                # (B, N, T, 3)
        per_frame = per_frame.permute(2, 0, 1, 3).reshape(T, BN, 3)

        t_emb = sinusoidal_time_embedding(
            torch.arange(T, device=pos.device), self.t_emb_dim,
        )
        t_emb = t_emb.unsqueeze(1).expand(T, BN, self.t_emb_dim)

        h_in = torch.cat([per_frame, t_emb], dim=-1)
        h = self.per_frame_encoder(h_in.reshape(T * BN, -1)).reshape(T, BN, self.hidden_dim)

        # Velocity channel (equivariant) — per-point 5-frame history
        vel_all = velocity_in.permute(0, 2, 1, 3).reshape(BN, T, 3)

        # Stacked EGNO blocks (spectral-time + spatial-GNN).
        for block in self.blocks:
            h, vel_all = block(h, pos_flat, vel_all, edge_index)

        # One-shot decode: batched graph copies for all T output frames.
        E = edge_index.shape[1]
        h_batched = h.reshape(T * BN, self.hidden_dim)
        x_batched = pos_flat.repeat(T, 1)
        vel_all_batched = vel_all.repeat(T, 1, 1)
        offsets = (torch.arange(T, device=pos.device) * BN).repeat_interleave(E)
        edges_batched = edge_index.repeat(1, T) + offsets.unsqueeze(0)

        last_gnn = self.blocks[-1].gnn
        _, _, m_ij_last = last_gnn(h_batched, x_batched, vel_all_batched, edges_batched)
        delta_flat = self.decoder(
            h_batched, m_ij_last, x_batched, vel_all_batched, edges_batched,
        )
        delta = delta_flat.reshape(T, B, N, 3).permute(1, 0, 2, 3).contiguous()

        # Reynolds-decomposition residual: add predicted Δv to the temporal
        # mean of the input window to get the output velocity field.
        mean_frame = velocity_in.mean(dim=1, keepdim=True)         # (B, 1, N, 3)
        v_out = mean_frame + delta                                  # (B, T, N, 3)

        # No-slip boundary: airfoil surface velocity = 0 at every output frame.
        if self.no_slip_mask:
            vol_mask = torch.ones(B, N, 1, device=pos.device, dtype=velocity_in.dtype)
            for b, idc in enumerate(idcs_airfoil):
                idc = idc.to(pos.device)
                vol_mask[b, idc, :] = 0.0
            v_out = v_out * vol_mask.unsqueeze(1)

        return v_out
