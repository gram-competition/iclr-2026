"""Two-hop directional finite-graph network v4 (competition-inference copy).

This is the same ``FiniteGraphModelV4`` architecture that was trained in
the source repo, reproduced here with local imports only so the submission
folder is self-contained. Only the pieces needed for inference are kept —
training code, precomputation caches and logging are omitted.

Upgrades over v3
----------------
1. **Temporal node encoder.** The 26-channel input is first encoded by a
   Conv1d stack over the reconstructed 5-snapshot velocity sequence,
   fused with a static-feature projection, and passed through a residual
   LayerNorm. All graph message passing operates on the encoded latent.
2. **Directional second-hop neighbourhoods.** Upstream and downstream
   branches now select flow-aware hop-2 neighbours (not a plain
   isotropic kNN lookup as in v3).
3. **Gated cross-direction fusion.** Iso / up / down latents are fused
   through centre-conditioned gate weights plus summary / spread /
   extrema pools, rather than a plain concatenation.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .features import (
    V_T4_CH_START, AIRFOIL_IND_CH, WALL_DIST_CH,
    VORT_CH_START, Q_CRIT_CH, STRAIN_MAG_CH, DT,
)
from .graph_utils import build_directional_graphs_np, build_knn_pool_np


# ---------------------------------------------------------------------------
# Node info (16 ch) — raw physics state of one node, shared between layer 1
# and layer 2 edge-feature computations.
#   pos(3) | v_t4(3) | wall_dist(1) | speed(1) | vort(3) | Q(1) | strain(1)
#   | accel(3)
# ---------------------------------------------------------------------------

POS_S = slice(0, 3)
V_T4_S = slice(3, 6)
WD_I = 6
SPD_I = 7
VORT_S = slice(8, 11)
Q_I = 11
STR_I = 12
ACC_S = slice(13, 16)
NODE_INFO_DIM = 16


def _split_info(info: torch.Tensor):
    return (
        info[..., POS_S],
        info[..., V_T4_S],
        info[..., WD_I],
        info[..., SPD_I],
        info[..., VORT_S],
        info[..., Q_I],
        info[..., STR_I],
        info[..., ACC_S],
    )


def _build_node_info_tensor(
    features_raw: torch.Tensor, pos: torch.Tensor, dt: float,
) -> torch.Tensor:
    v_t4 = features_raw[:, V_T4_CH_START:V_T4_CH_START + 3]
    wd = features_raw[:, WALL_DIST_CH:WALL_DIST_CH + 1]
    speed = v_t4.norm(dim=-1, keepdim=True)
    vort = features_raw[:, VORT_CH_START:VORT_CH_START + 3]
    q = features_raw[:, Q_CRIT_CH:Q_CRIT_CH + 1]
    strain = features_raw[:, STRAIN_MAG_CH:STRAIN_MAG_CH + 1]
    delta3 = features_raw[:, 6:9]
    delta4 = features_raw[:, 9:12]
    accel = (delta4 - delta3) / dt
    return torch.cat([pos, v_t4, wd, speed, vort, q, strain, accel], dim=-1)


# ---------------------------------------------------------------------------
# Edge features (39 channels, same layout as v3)
# ---------------------------------------------------------------------------

FOURIER_FREQS = (1.0, 4.0, 16.0)

EDGE_FEAT_DIM = (
    3
    + 1
    + len(FOURIER_FREQS) * 3 * 2
    + 1
    + 1
    + 1
    + 1
    + 1
    + 1
    + 3
    + 3
    + 1
    + 1
    + 3
)  # = 39


def compute_edge_features_v4(
    centre_info: torch.Tensor,
    nbr_info: torch.Tensor,
    L_ref,
) -> torch.Tensor:
    pos_c, v_t4_c, wd_c, spd_c, vort_c, q_c, str_c, acc_c = _split_info(centre_info)
    pos_n, v_t4_n, wd_n, spd_n, vort_n, q_n, str_n, acc_n = _split_info(nbr_info)

    rel_pos = pos_n - pos_c.unsqueeze(1)
    rel_dist = rel_pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    freqs = torch.as_tensor(
        FOURIER_FREQS, device=rel_pos.device, dtype=rel_pos.dtype,
    )
    if isinstance(L_ref, torch.Tensor):
        L = L_ref.to(rel_pos.dtype).view(-1, 1, 1, 1).clamp(min=1e-3)
    else:
        L = float(L_ref)
    scaled = (2.0 * math.pi) * rel_pos.unsqueeze(-1) / L * freqs
    fourier = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
    B_, k_, _, _ = fourier.shape
    fourier = fourier.reshape(B_, k_, -1)

    spd_c_safe = spd_c.clamp(min=1e-8)
    v_hat = v_t4_c / spd_c_safe.unsqueeze(-1)
    v_par = (rel_pos * v_hat.unsqueeze(1)).sum(dim=-1, keepdim=True)
    v_par_norm = v_par / rel_dist
    perp_sq = (rel_dist ** 2 - v_par ** 2).clamp(min=0.0)
    v_perp_norm = perp_sq.sqrt() / rel_dist

    delta_wall  = (wd_n - wd_c.unsqueeze(-1)).unsqueeze(-1)
    speed_delta = (spd_n - spd_c.unsqueeze(-1)).unsqueeze(-1)
    v_diff      = v_t4_n - v_t4_c.unsqueeze(1)
    vort_diff   = vort_n - vort_c.unsqueeze(1)
    q_diff      = (q_n - q_c.unsqueeze(-1)).unsqueeze(-1)
    strain_diff = (str_n - str_c.unsqueeze(-1)).unsqueeze(-1)
    accel_diff  = acc_n - acc_c.unsqueeze(1)

    out = torch.cat([
        rel_pos,
        rel_dist,
        fourier,
        wd_n.unsqueeze(-1),
        delta_wall,
        v_par_norm,
        v_perp_norm,
        spd_n.unsqueeze(-1),
        speed_delta,
        v_diff,
        vort_diff,
        q_diff,
        strain_diff,
        accel_diff,
    ], dim=-1)
    assert out.shape[-1] == EDGE_FEAT_DIM, out.shape
    return out


# ---------------------------------------------------------------------------
# Temporal node encoder — learned embedding from raw 26-ch features.
# ---------------------------------------------------------------------------

DELTA_SLICE = slice(0, 12)
VT4_SLICE = slice(12, 15)
STATIC_SLICE = slice(15, 26)


def _reconstruct_velocity_sequence(flat_features: torch.Tensor) -> torch.Tensor:
    """Reconstruct the 5 raw input velocities from feature channels."""
    deltas = flat_features[:, DELTA_SLICE].reshape(-1, 4, 3)
    v_t4 = flat_features[:, VT4_SLICE]
    v0 = v_t4 - deltas[:, 3]
    seq = torch.stack([
        v0,
        v0 + deltas[:, 0],
        v0 + deltas[:, 1],
        v0 + deltas[:, 2],
        v_t4,
    ], dim=1)
    return seq


class TemporalNodeEncoderV4(nn.Module):
    """Encode raw node features into a learned graph state."""

    def __init__(
        self,
        in_ch: int,
        node_ch: int,
        temporal_hidden: int = 96,
        dropout: float = 0.0,
    ):
        super().__init__()
        static_ch = STATIC_SLICE.stop - STATIC_SLICE.start
        self.temporal = nn.Sequential(
            nn.Conv1d(3, temporal_hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(temporal_hidden, temporal_hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.temporal_proj = nn.Sequential(
            nn.Linear(2 * temporal_hidden, temporal_hidden),
            nn.SiLU(inplace=True),
        )
        self.static_proj = nn.Sequential(
            nn.LayerNorm(static_ch),
            nn.Linear(static_ch, temporal_hidden),
            nn.SiLU(inplace=True),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Sequential(
            nn.Linear(2 * temporal_hidden, node_ch),
            nn.SiLU(inplace=True),
            nn.Linear(node_ch, node_ch),
        )
        self.res = nn.Linear(in_ch, node_ch, bias=False)
        self.norm = nn.LayerNorm(node_ch)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        orig_shape = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        vel_seq = _reconstruct_velocity_sequence(flat).transpose(1, 2)
        temp_h = self.temporal(vel_seq)
        temp_feat = torch.cat([temp_h.mean(dim=-1), temp_h[..., -1]], dim=-1)
        temp_feat = self.temporal_proj(temp_feat)
        static_feat = self.static_proj(flat[:, STATIC_SLICE])
        fused = torch.cat([temp_feat, static_feat], dim=-1)
        out = self.out(self.drop(fused))
        out = self.norm(out + self.res(flat))
        return out.reshape(*orig_shape, -1)


# ---------------------------------------------------------------------------
# Directional second-hop selection (torch).
# ---------------------------------------------------------------------------

_SENTINEL = 1e30


def select_directional_second_hop_torch(
    node_info: torch.Tensor,
    first_hop_idx: torch.Tensor,
    nbrs_pool: torch.Tensor,
    k2: int,
    direction: str,
) -> torch.Tensor:
    flat_idx = first_hop_idx.reshape(-1)
    cand = nbrs_pool[flat_idx]
    if k2 > cand.shape[1]:
        raise ValueError(f"k2={k2} exceeds candidate pool size {cand.shape[1]}")
    iso = cand[:, :k2].clone()
    if direction == "iso":
        return iso.reshape(*first_hop_idx.shape, k2)

    centre = node_info[flat_idx]
    centre_pos = centre[:, POS_S]
    centre_v = centre[:, V_T4_S]
    speed = centre_v.norm(dim=-1)
    stagnation = speed < 1e-6
    v_hat = torch.zeros_like(centre_v)
    moving = ~stagnation
    v_hat[moving] = centre_v[moving] / speed[moving].unsqueeze(-1)

    cand_pos = node_info[cand][..., POS_S]
    dp = cand_pos - centre_pos.unsqueeze(1)
    dist = dp.norm(dim=-1)
    proj = (dp * v_hat.unsqueeze(1)).sum(dim=-1)
    sentinel = dist.new_full(dist.shape, _SENTINEL)

    if direction == "up":
        masked = torch.where(proj < 0.0, dist, sentinel)
    elif direction == "down":
        masked = torch.where(proj > 0.0, dist, sentinel)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    sel_dist, order = torch.topk(masked, k2, dim=-1, largest=False, sorted=True)
    out = cand.gather(-1, order)
    invalid = sel_dist[:, -1] >= _SENTINEL
    bad = invalid | stagnation
    if bad.any():
        out[bad] = iso[bad]
    return out.reshape(*first_hop_idx.shape, k2)


# ---------------------------------------------------------------------------
# Message passing and fusion.
# ---------------------------------------------------------------------------


class MessagePassingLayerV4(nn.Module):
    def __init__(
        self,
        centre_ch: int,
        nbr_ch: int,
        edge_ch: int,
        hidden: int,
        out_ch: int,
        n_attn_heads: int = 4,
    ):
        super().__init__()
        assert hidden % n_attn_heads == 0, (hidden, n_attn_heads)

        msg_in = centre_ch + nbr_ch + edge_ch
        self.msg = nn.Sequential(
            nn.Linear(msg_in, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
        )
        self.q_proj = nn.Linear(centre_ch, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.attn_out = nn.Linear(hidden, hidden)

        self.n_heads = n_attn_heads
        self.head_dim = hidden // n_attn_heads
        self.scale = math.sqrt(self.head_dim)

        self.post = nn.Sequential(
            nn.Linear(3 * hidden, out_ch),
            nn.SiLU(inplace=True),
        )
        self.norm = nn.LayerNorm(out_ch)
        self.res = (
            nn.Identity() if centre_ch == out_ch
            else nn.Linear(centre_ch, out_ch, bias=False)
        )

    def forward(
        self,
        centre: torch.Tensor,
        nbr: torch.Tensor,
        edge: torch.Tensor,
    ) -> torch.Tensor:
        B, k, _ = nbr.shape

        centre_exp = centre.unsqueeze(1).expand(-1, k, -1)
        msg_in = torch.cat([centre_exp, nbr, edge], dim=-1)
        h = self.msg(msg_in.reshape(B * k, -1)).reshape(B, k, -1)

        q = self.q_proj(centre).view(B, self.n_heads, 1, self.head_dim)
        kk = self.k_proj(h).view(B, k, self.n_heads, self.head_dim).transpose(1, 2)
        vv = self.v_proj(h).view(B, k, self.n_heads, self.head_dim).transpose(1, 2)
        attn = ((q @ kk.transpose(-1, -2)) / self.scale).softmax(dim=-1)
        attn_pool = (attn @ vv).squeeze(2).reshape(B, -1)
        attn_pool = self.attn_out(attn_pool)
        max_pool = h.max(dim=1).values
        var_pool = h.var(dim=1, unbiased=False)
        pooled = torch.cat([attn_pool, max_pool, var_pool], dim=-1)
        out = self.post(pooled)
        return self.norm(out + self.res(centre))


class TwoHopDirectionalConvV4(nn.Module):
    def __init__(
        self,
        node_ch: int,
        edge_ch: int,
        hidden: int,
        latent: int,
        n_attn_heads: int = 4,
    ):
        super().__init__()
        self.l2_mp = MessagePassingLayerV4(
            centre_ch=node_ch,
            nbr_ch=node_ch,
            edge_ch=edge_ch,
            hidden=hidden,
            out_ch=hidden,
            n_attn_heads=n_attn_heads,
        )
        self.l1_mp = MessagePassingLayerV4(
            centre_ch=node_ch,
            nbr_ch=hidden,
            edge_ch=edge_ch,
            hidden=hidden,
            out_ch=latent,
            n_attn_heads=n_attn_heads,
        )

    def forward(
        self,
        centre_feat: torch.Tensor,
        l1_feat: torch.Tensor,
        l2_feat: torch.Tensor,
        l2_edge: torch.Tensor,
        l1_edge: torch.Tensor,
    ) -> torch.Tensor:
        B, k1, k2, C = l2_feat.shape
        l1_flat = l1_feat.reshape(B * k1, C)
        l2_flat = l2_feat.reshape(B * k1, k2, C)
        l2_edge_flat = l2_edge.reshape(B * k1, k2, -1)
        l1_updated = self.l2_mp(l1_flat, l2_flat, l2_edge_flat).reshape(B, k1, -1)
        return self.l1_mp(centre_feat, l1_updated, l1_edge)


class GatedDirectionalFusion(nn.Module):
    """Fuse iso / up / down latents with centre-conditioned gates."""

    def __init__(
        self,
        centre_ch: int,
        latent: int,
        hidden: int,
        out_ch: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        gate_in = centre_ch + 3 * latent
        self.gates = nn.Sequential(
            nn.Linear(gate_in, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fuse = nn.Sequential(
            nn.Linear(centre_ch + 3 * latent, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        centre: torch.Tensor,
        h_iso: torch.Tensor,
        h_up: torch.Tensor,
        h_down: torch.Tensor,
    ) -> torch.Tensor:
        gate_input = torch.cat([centre, h_iso, h_up, h_down], dim=-1)
        weights = self.gates(gate_input).softmax(dim=-1)
        stack = torch.stack([h_iso, h_up, h_down], dim=1)
        summary = (weights.unsqueeze(-1) * stack).sum(dim=1)
        spread = stack.var(dim=1, unbiased=False)
        maxima = stack.max(dim=1).values
        fused_in = torch.cat([centre, summary, spread, maxima], dim=-1)
        return self.fuse(self.drop(fused_in))


# ---------------------------------------------------------------------------
# Full v4 model.
# ---------------------------------------------------------------------------


class FiniteGraphModelV4(nn.Module):
    """Two-hop directional graph forecaster with temporal encoding."""

    def __init__(
        self,
        in_ch: int = 26,
        hidden: int = 192,
        latent: int = 192,
        k1: int = 24,
        k2: int = 12,
        n_attn_heads: int = 4,
        out_heads: int = 5,
        out_ch_per_head: int = 3,
        shared_weights: bool = False,
        dropout: float = 0.05,
        temporal_hidden: int = 96,
    ):
        super().__init__()
        self._config = dict(
            in_ch=in_ch,
            hidden=hidden,
            latent=latent,
            k1=k1,
            k2=k2,
            n_attn_heads=n_attn_heads,
            out_heads=out_heads,
            out_ch_per_head=out_ch_per_head,
            shared_weights=shared_weights,
            dropout=dropout,
            temporal_hidden=temporal_hidden,
        )
        self.in_ch = in_ch
        self.hidden = hidden
        self.latent = latent
        self.k1 = k1
        self.k2 = k2
        self.out_heads = out_heads
        self.out_ch_per_head = out_ch_per_head
        self.shared_weights = shared_weights

        self.node_encoder = TemporalNodeEncoderV4(
            in_ch=in_ch,
            node_ch=hidden,
            temporal_hidden=temporal_hidden,
            dropout=dropout,
        )

        def _make() -> TwoHopDirectionalConvV4:
            return TwoHopDirectionalConvV4(
                node_ch=hidden,
                edge_ch=EDGE_FEAT_DIM,
                hidden=hidden,
                latent=latent,
                n_attn_heads=n_attn_heads,
            )

        if shared_weights:
            self.conv = _make()
        else:
            self.iso_conv = _make()
            self.up_conv = _make()
            self.down_conv = _make()

        self.dir_fusion = GatedDirectionalFusion(
            centre_ch=hidden,
            latent=latent,
            hidden=2 * latent,
            out_ch=latent,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(latent, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, out_heads * out_ch_per_head),
        )

    @property
    def k(self) -> int:
        return self.k1

    def _get_conv(self, which: str) -> TwoHopDirectionalConvV4:
        if self.shared_weights:
            return self.conv
        return getattr(self, f"{which}_conv")

    def forward_nodes(
        self,
        center_feat: torch.Tensor,
        iso_l1_feat: torch.Tensor,
        iso_l1_edge: torch.Tensor,
        iso_l2_feat: torch.Tensor,
        iso_l2_edge: torch.Tensor,
        up_l1_feat: torch.Tensor,
        up_l1_edge: torch.Tensor,
        up_l2_feat: torch.Tensor,
        up_l2_edge: torch.Tensor,
        down_l1_feat: torch.Tensor,
        down_l1_edge: torch.Tensor,
        down_l2_feat: torch.Tensor,
        down_l2_edge: torch.Tensor,
    ) -> torch.Tensor:
        center_enc = self.node_encoder(center_feat)
        iso_l1_enc = self.node_encoder(iso_l1_feat)
        iso_l2_enc = self.node_encoder(iso_l2_feat)
        up_l1_enc = self.node_encoder(up_l1_feat)
        up_l2_enc = self.node_encoder(up_l2_feat)
        down_l1_enc = self.node_encoder(down_l1_feat)
        down_l2_enc = self.node_encoder(down_l2_feat)

        h_iso = self._get_conv("iso")(
            center_enc, iso_l1_enc, iso_l2_enc, iso_l2_edge, iso_l1_edge,
        )
        h_up = self._get_conv("up")(
            center_enc, up_l1_enc, up_l2_enc, up_l2_edge, up_l1_edge,
        )
        h_down = self._get_conv("down")(
            center_enc, down_l1_enc, down_l2_enc, down_l2_edge, down_l1_edge,
        )

        fused = self.dir_fusion(center_enc, h_iso, h_up, h_down)
        out = self.head(fused)
        return out.view(-1, self.out_heads, self.out_ch_per_head)


# ---------------------------------------------------------------------------
# Full-chunk inference wrapper.
# ---------------------------------------------------------------------------


class FiniteGraphInferenceWrapperV4(nn.Module):
    def __init__(
        self,
        model: FiniteGraphModelV4,
        k_pool: int = 128,
        inference_batch: int = 768,
    ):
        super().__init__()
        self.model = model
        self.k_pool = k_pool
        self.inference_batch = inference_batch
        self._feat_mean: Optional[torch.Tensor] = None
        self._feat_std: Optional[torch.Tensor] = None

    def set_stats(self, feat_mean: torch.Tensor, feat_std: torch.Tensor):
        self._feat_mean = feat_mean
        self._feat_std = feat_std

    @property
    def k(self) -> int:
        return self.model.k1

    def forward(
        self,
        features: torch.Tensor,   # (N, 26) — normalised if stats are set
        pos: torch.Tensor,        # (N, 3) raw
    ) -> torch.Tensor:
        device = features.device
        k1 = self.model.k1
        k2 = self.model.k2
        n_pts = pos.shape[0]

        if self._feat_mean is not None:
            fm = self._feat_mean.to(device)
            fs = self._feat_std.to(device)
            features_raw = features * fs + fm
        else:
            features_raw = features

        pos_np = pos.detach().cpu().numpy().astype(np.float32)
        pool_np = build_knn_pool_np(pos_np, self.k_pool)
        pool_t = torch.from_numpy(pool_np.copy()).to(device)

        v_t4_np = features_raw[:, V_T4_CH_START:V_T4_CH_START + 3].detach().cpu().numpy()
        airfoil_np = features_raw[:, AIRFOIL_IND_CH].detach().cpu().numpy() > 0.5
        graphs = build_directional_graphs_np(
            pos_np, v_t4_np, airfoil_np, pool_np, k=k1,
        )
        node_ids = torch.from_numpy(graphs["node_ids"]).to(device)
        iso_n1 = torch.from_numpy(graphs["iso"]).to(device)
        up_n1 = torch.from_numpy(graphs["up"]).to(device)
        down_n1 = torch.from_numpy(graphs["down"]).to(device)

        node_info = _build_node_info_tensor(features_raw, pos, DT)
        l_ref_val = float(
            (pos.max(0).values - pos.min(0).values).max().clamp(min=1e-3).item()
        )

        n_valid = node_ids.shape[0]
        out_all = torch.zeros(
            n_pts, self.model.out_heads, self.model.out_ch_per_head,
            device=device, dtype=features.dtype,
        )

        for start in range(0, n_valid, self.inference_batch):
            end = min(start + self.inference_batch, n_valid)
            nids = node_ids[start:end]
            bsz = nids.shape[0]
            center = features_raw[nids]
            centre_info = node_info[nids]
            l_ref_t = torch.full((bsz,), l_ref_val, device=device, dtype=centre_info.dtype)

            packed: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            dir_sets = {
                "iso": iso_n1[start:end],
                "up": up_n1[start:end],
                "down": down_n1[start:end],
            }
            for name, n1_idx in dir_sets.items():
                n2_idx = select_directional_second_hop_torch(
                    node_info, n1_idx, pool_t, k2, name,
                )
                l1_feat = features_raw[n1_idx]
                l2_feat = features_raw[n2_idx]
                l1_info = node_info[n1_idx]
                l2_info = node_info[n2_idx]

                l1_edge = compute_edge_features_v4(centre_info, l1_info, l_ref_t)
                l1_info_flat = l1_info.reshape(bsz * k1, -1)
                l2_info_flat = l2_info.reshape(bsz * k1, k2, -1)
                l_flat = l_ref_t.unsqueeze(1).expand(-1, k1).reshape(bsz * k1)
                l2_edge_flat = compute_edge_features_v4(
                    l1_info_flat, l2_info_flat, l_flat,
                )
                l2_edge = l2_edge_flat.reshape(bsz, k1, k2, -1)
                packed[name] = (l1_feat, l1_edge, l2_feat, l2_edge)

            pred = self.model.forward_nodes(
                center,
                packed["iso"][0], packed["iso"][1], packed["iso"][2], packed["iso"][3],
                packed["up"][0], packed["up"][1], packed["up"][2], packed["up"][3],
                packed["down"][0], packed["down"][1], packed["down"][2], packed["down"][3],
            )
            out_all[nids] = pred

        return out_all.permute(1, 0, 2)  # (5, N, 3)
