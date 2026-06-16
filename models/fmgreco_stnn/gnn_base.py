"""Spatiotemporal GNN: k-NN graph, MGN backbone, temporal head, residual + no-slip."""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn

from .backbones import MeshGraphNetBackbone
from .equivariant import (
    EquivariantVelocityLift,
    SO3EquivariantBackbone,
    SO3ToHiddenMerge,
)
from .graph_utils import airfoil_boundary_features, knn_graph
from .temporal import TemporalAttentionHead


def _sanitize_airfoil_idx(
    idx: torch.Tensor, num_points: int, device: torch.device
) -> torch.Tensor:
    if idx.numel() == 0:
        return idx.to(device=device, dtype=torch.long)
    return idx.to(device=device, dtype=torch.long).clamp_(0, num_points - 1)


def _load_state_dict_shape_safe(module: nn.Module, state: dict[str, Any]) -> None:
    """Load only tensors that exist on ``module`` and match shape (``strict=False`` is not enough)."""
    model_sd = module.state_dict()
    compatible = {
        k: v
        for k, v in state.items()
        if k in model_sd and model_sd[k].shape == v.shape
    }
    module.load_state_dict(compatible, strict=False)
    skipped = [k for k in state if k not in compatible]
    if skipped and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        warnings.warn(
            f"Bundled state_dict.pt: skipped {len(skipped)} key(s) (missing or shape mismatch vs "
            f"current architecture). Example keys: {skipped[:6]!r}",
            stacklevel=2,
        )


def _coerce_t_vector(t: torch.Tensor, t_dim: int, *, ref: torch.Tensor) -> torch.Tensor:
    """Reshape ``t`` to length ``t_dim`` (pad or trim) and match device/dtype of ``ref``."""
    flat = t.reshape(-1)
    d = t_dim
    if flat.numel() >= d:
        out = flat[:d]
    else:
        out = torch.nn.functional.pad(flat, (0, d - int(flat.numel())))
    return out.to(device=ref.device, dtype=ref.dtype)


def _scaled_velocity_zero(
    velocity_mean: Optional[torch.Tensor],
    velocity_std: Optional[torch.Tensor],
    *,
    ref: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Standardized velocity that corresponds to physical :math:`\\mathbf{v}=0` (per component)."""
    if velocity_mean is None or velocity_std is None:
        return None
    m = velocity_mean.flatten()[:3]
    s = velocity_std.flatten()[:3].clamp_min(1e-12)
    return (-m / s).to(device=ref.device, dtype=ref.dtype)


def apply_hard_no_slip(
    pred_velocity_txn3: torch.Tensor,
    airfoil_idx: torch.Tensor,
    *,
    scaled_physical_zero: Optional[torch.Tensor] = None,
) -> None:
    """
    Enforce a **hard** no-slip condition: surface velocity is exactly zero.

    ``pred_velocity_txn3`` is shaped ``(T, N, 3)``; ``airfoil_idx`` indexes
    boundary nodes along ``N``. Updates the tensor **in place**.

    When inputs are standardized (``(v - mean) / std``), pass ``scaled_physical_zero``
    so boundary values match physical zero (same tensor as ``assert_no_slip_boundary``).
    Otherwise defaults to literal ``0`` in tensor space (e.g. unscaled smoke tests).
    """
    if airfoil_idx.numel() == 0:
        return
    idx = airfoil_idx.to(device=pred_velocity_txn3.device, dtype=torch.long)
    if scaled_physical_zero is None:
        pred_velocity_txn3[:, idx] = 0.0
    else:
        z = scaled_physical_zero.flatten()[:3].to(
            device=pred_velocity_txn3.device, dtype=pred_velocity_txn3.dtype
        )
        pred_velocity_txn3[:, idx, :] = z


class FourierPosEnc(nn.Module):
    def __init__(self, in_dim: int = 3, num_freqs: int = 8):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)
        self.out_dim = in_dim + in_dim * num_freqs * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x.unsqueeze(-1) * self.freqs
        return torch.cat(
            [x, proj.sin().flatten(-2), proj.cos().flatten(-2)], dim=-1
        )


class DeltaDecoder(nn.Module):
    """Maps (N, t_out, td) -> (N, t_out, 3) optional SIREN-style sin stack on the delta."""

    def __init__(
        self,
        td: int,
        t_out: int,
        hidden_dim: int,
        *,
        use_siren: bool,
        siren_omega0: float,
        dropout: float,
    ):
        super().__init__()
        self.use_siren = use_siren
        self.siren_omega0 = siren_omega0
        if use_siren:
            h = max(32, hidden_dim)
            self.w1 = nn.Linear(td, h)
            self.w2 = nn.Linear(h, h)
            self.out = nn.Linear(h, 3)
            self.drop = nn.Dropout(dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(td, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self.use_siren:
            return self.mlp(h)
        x = self.siren_omega0 * self.w1(h)
        x = torch.sin(x)
        x = self.drop(x)
        x = torch.sin(self.w2(x))
        return self.out(x)


class SpatioTemporalGNN(nn.Module):
    """
    Competition entry: full-res k-NN + MeshGraphNet + temporal attention.
    Constructor must be callable with no args; loads ``state_dict.pt`` if present.

    The public API passes ``t`` of shape ``(B, 10)`` (see ``main.run_smoke``). When
    ``use_t_embedding`` is True, ``t`` is embedded with a small MLP and concatenated
    to every node (same idea as a typical reference MLP that conditions on global
    time / sequence metadata). Set ``use_t_embedding=False`` to match older
    checkpoints that were trained with ``t`` ignored.

    **Residual delta prediction:** the decoder outputs a correction; outputs are
    ``baseline + delta`` where ``baseline`` is the last input frame (default) or
    the mean over input times. The network does not re-predict the entire field from
    scratch, which stabilizes learning for small inter-frame changes.

    **No-slip:** airfoil indices zero out velocity in the encoded sequence and in the
    residual baseline so the surface is pinned to zero before the decoder adds corrections;
    outputs are hard-masked to zero on ``idcs_airfoil``.

    Incremental features (graph path):
    - Airfoil boundary features: log1p(min dist) + unit direction to nearest surface
      sample (chunked cdist; surface may be subsampled).
    - Fourier features on positions for high-frequency spatial encoding.
    - Optional standardized **pressure** for each input time step (scalar per node).
    - Optional **relative velocity inputs** ``v_t - v_last`` (after surface zeroing).
    - Optional sin-heavy head on the predicted delta (SIREN-style first layer).

    Deeper ``num_layers`` increases the number of k-NN message-passing hops (wider
    receptive field along the graph). Use e.g. 12 on large-GPU runs if memory allows.

    Set ``use_so3_backbone=True`` for SO(3)-equivariant vector message passing (optional).

    Set ``use_steerable_backbone=True`` for full e3nn steerable irreps + spherical-harmonic
    k-NN tensor products (optional; mutually exclusive with ``use_so3_backbone``).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 12,
        heads: int = 4,
        k: int = 16,
        t_in: int = 5,
        t_out: int = 5,
        use_fourier: bool = True,
        num_fourier: int = 8,
        dropout: float = 0.1,
        use_boundary_features: bool = True,
        max_airfoil_samples: int = 4096,
        boundary_feature_chunk: int = 8192,
        residual_baseline: str = "last",
        use_siren_head: bool = False,
        siren_omega0: float = 30.0,
        num_heads: Optional[int] = None,
        num_attn_layers: int = 2,
        use_so3_backbone: bool = False,
        use_steerable_backbone: bool = False,
        steerable_lmax: int = 2,
        use_pressure: bool = False,
        use_relative_velocity_inputs: bool = False,
        num_t_in: Optional[int] = None,
        num_t_out: Optional[int] = None,
        load_bundled_weights: bool = True,
        use_t_embedding: bool = False,
        t_dim: int = 10,
        t_embed_dim: int = 32,
        **kwargs: Any,
    ):
        super().__init__()
        _ = kwargs  # trainer may pass num_modes, knn_query_chunk_size, etc.
        if num_heads is not None:
            heads = num_heads
        if num_t_in is not None:
            t_in = num_t_in
        if num_t_out is not None:
            t_out = num_t_out
        self.use_t_embedding = bool(use_t_embedding)
        self.t_dim = int(t_dim)
        self.t_embed_dim = int(t_embed_dim)

        self.k = k
        self.t_in = t_in
        self.t_out = t_out
        self.use_boundary_features = use_boundary_features
        self.max_airfoil_samples = max_airfoil_samples
        self.boundary_feature_chunk = boundary_feature_chunk
        if residual_baseline not in ("last", "mean"):
            raise ValueError("residual_baseline must be 'last' or 'mean'")
        self.residual_baseline = residual_baseline
        self.use_siren_head = use_siren_head
        self.use_so3_backbone = bool(use_so3_backbone)
        self.use_steerable_backbone = bool(use_steerable_backbone)
        self.steerable_lmax = int(steerable_lmax)
        if self.use_so3_backbone and self.use_steerable_backbone:
            raise ValueError(
                "use_so3_backbone and use_steerable_backbone are mutually exclusive"
            )
        self.d_v = max(8, hidden_dim // 8)
        self.use_pressure = bool(use_pressure)
        self.use_relative_velocity_inputs = bool(use_relative_velocity_inputs)

        if use_fourier:
            self.pos_enc = FourierPosEnc(3, num_fourier)
            pos_dim = self.pos_enc.out_dim
        else:
            self.pos_enc = None
            pos_dim = 3

        b_dim = 9 if use_boundary_features else 0
        p_dim = t_in if self.use_pressure else 0
        if self.use_t_embedding:
            self.t_mlp = nn.Sequential(
                nn.Linear(self.t_dim, self.t_embed_dim),
                nn.LayerNorm(self.t_embed_dim),
                nn.GELU(),
                nn.Linear(self.t_embed_dim, self.t_embed_dim),
            )
        else:
            self.t_mlp = None
        t_extra = self.t_embed_dim if self.use_t_embedding else 0
        in_dim = pos_dim + t_in * 3 + p_dim + 1 + b_dim + t_extra
        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        if self.use_steerable_backbone:
            from .steerable import (
                SteerableEquivariantBackbone,
                SteerableToHiddenMerge,
                hidden_dim_to_irreps,
            )

            self.irreps = hidden_dim_to_irreps(hidden_dim)
            if self.irreps.dim != hidden_dim:
                raise RuntimeError(
                    f"irreps dimension mismatch: {self.irreps.dim} != {hidden_dim}"
                )
            self.steerable_embed = nn.Linear(hidden_dim, self.irreps.dim)
            self.steerable_spatial = SteerableEquivariantBackbone(
                self.irreps,
                num_layers=num_layers,
                lmax=self.steerable_lmax,
                dropout=dropout,
            )
            self.steerable_merge = SteerableToHiddenMerge(self.irreps, hidden_dim)
            self.vel_lift = None
            self.so3_spatial = None
            self.invariant_merge = None
            self.spatial = None
        elif self.use_so3_backbone:
            self.irreps = None
            self.steerable_embed = None
            self.steerable_spatial = None
            self.steerable_merge = None
            self.vel_lift = EquivariantVelocityLift(self.d_v)
            self.so3_spatial = SO3EquivariantBackbone(
                hidden_dim, self.d_v, num_layers=num_layers, dropout=dropout
            )
            self.invariant_merge = SO3ToHiddenMerge(
                hidden_dim, self.d_v, hidden_dim
            )
            self.spatial = None
        else:
            self.irreps = None
            self.steerable_embed = None
            self.steerable_spatial = None
            self.steerable_merge = None
            self.vel_lift = None
            self.so3_spatial = None
            self.invariant_merge = None
            self.spatial = MeshGraphNetBackbone(
                hidden_dim, num_layers=num_layers, dropout=dropout
            )
        self.temporal = TemporalAttentionHead(
            hidden_dim,
            t_out=t_out,
            temporal_dim=hidden_dim,
            num_heads=heads,
            num_attn_layers=num_attn_layers,
            dropout=dropout,
        )
        self.temporal_dim = hidden_dim
        self.decoder = DeltaDecoder(
            self.temporal_dim,
            t_out,
            hidden_dim,
            use_siren=use_siren_head,
            siren_omega0=siren_omega0,
            dropout=dropout,
        )

        # Optional bundled weights for smoke tests / baseline submission. Typical size is
        # a few MB; very large checkpoints belong in Git LFS or a release, not the PR diff.
        weights = os.path.join(
            os.path.dirname(__file__), "state_dict.pt"
        )
        if load_bundled_weights and os.path.isfile(weights):
            try:
                ck = torch.load(weights, map_location="cpu", weights_only=False)
            except TypeError:
                ck = torch.load(weights, map_location="cpu")
            state = ck.get("model_state_dict", ck)
            if isinstance(state, dict):
                _load_state_dict_shape_safe(self, state)

    def _velocity_baseline(self, vel_in: torch.Tensor) -> torch.Tensor:
        """(t_in, N, 3) -> (N, 3) fixed prior for residual."""
        if self.residual_baseline == "mean":
            return vel_in.mean(dim=0)
        return vel_in[-1]

    def _forward_single(
        self,
        pos: torch.Tensor,
        vel_in: torch.Tensor,
        airfoil_idx: torch.Tensor,
        pressure_in: Optional[torch.Tensor] = None,
        velocity_mean: Optional[torch.Tensor] = None,
        velocity_std: Optional[torch.Tensor] = None,
        t_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = pos.size(0)
        device = pos.device
        airfoil_idx = _sanitize_airfoil_idx(airfoil_idx, n, device)
        mask = pos.new_zeros(n, 1)
        mask[airfoil_idx] = 1.0
        v0 = _scaled_velocity_zero(velocity_mean, velocity_std, ref=vel_in)

        neighbors, rel_pos, dists = knn_graph(pos, self.k)

        pos_feat = self.pos_enc(pos) if self.pos_enc else pos
        vel_enc = vel_in.clone()
        if v0 is None:
            vel_enc[:, airfoil_idx] = 0.0
        else:
            vel_enc[:, airfoil_idx, :] = v0
        if self.use_relative_velocity_inputs:
            vel_enc = vel_enc - vel_enc[-1].unsqueeze(0)
        vel_flat = vel_enc.permute(1, 0, 2).reshape(n, -1)
        parts = [pos_feat, vel_flat]
        if self.use_pressure:
            if pressure_in is None:
                pfeat = pos.new_zeros(n, self.t_in)
            else:
                pfeat = pressure_in.permute(1, 0).reshape(n, self.t_in)
            parts.append(pfeat)
        parts.append(mask)
        if self.use_t_embedding and self.t_mlp is not None:
            t_vec = _coerce_t_vector(
                t_s if t_s is not None else pos.new_zeros(self.t_dim),
                self.t_dim,
                ref=pos,
            )
            te = self.t_mlp(t_vec)
            parts.append(te.unsqueeze(0).expand(n, -1))
        if self.use_boundary_features:
            bfeat = airfoil_boundary_features(
                pos,
                airfoil_idx,
                max_airfoil_samples=self.max_airfoil_samples,
                chunk_size=self.boundary_feature_chunk,
            )
            parts.append(bfeat)
        # Boundary math may run in fp32 internally then cast back; align all strips to
        # ``pos.dtype`` so ``torch.cat`` never mixes bf16/fp32 (e.g. AMP + float32 pressure).
        feat_dtype = pos.dtype
        x = torch.cat([p.to(dtype=feat_dtype) for p in parts], dim=-1)
        x = self.node_enc(x)
        if self.use_steerable_backbone:
            x = self.steerable_embed(x)
            x = self.steerable_spatial(x, neighbors, rel_pos, dists)
            x = self.steerable_merge(x)
        elif self.use_so3_backbone:
            u0 = self._velocity_baseline(vel_enc)
            v0 = self.vel_lift(u0)
            s, v = self.so3_spatial(x, v0, neighbors, rel_pos, dists)
            x = self.invariant_merge(s, v)
        else:
            x = self.spatial(x, neighbors, rel_pos, dists)
        x = self.temporal(x)
        delta = self.decoder(x)
        baseline = self._velocity_baseline(vel_enc)
        x = delta + baseline.unsqueeze(1)
        x_full = x.permute(1, 0, 2)
        apply_hard_no_slip(x_full, airfoil_idx, scaled_physical_zero=v0)
        return x_full

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
        velocity_mean: Optional[torch.Tensor] = None,
        velocity_std: Optional[torch.Tensor] = None,
        *,
        wall_distance: Optional[torch.Tensor] = None,
        surface_frame: Optional[torch.Tensor] = None,
        knn_indices: Optional[torch.Tensor] = None,
        pressure_in: Optional[torch.Tensor] = None,
        return_knn_indices: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.use_t_embedding:
            _ = t
        _ = (wall_distance, surface_frame, knn_indices)
        b = velocity_in.size(0)
        outs = []
        for bi in range(b):
            n_pts = pos[bi].shape[0]
            p_batch = pressure_in[bi] if pressure_in is not None else None
            vm = velocity_mean[bi] if velocity_mean is not None else None
            vs = velocity_std[bi] if velocity_std is not None else None
            t_batch = t[bi] if self.use_t_embedding else None
            out = self._forward_single(
                pos[bi],
                velocity_in[bi],
                _sanitize_airfoil_idx(idcs_airfoil[bi], n_pts, pos.device),
                p_batch,
                velocity_mean=vm,
                velocity_std=vs,
                t_s=t_batch,
            )
            outs.append(out)
        stacked = torch.stack(outs, dim=0)
        if return_knn_indices:
            return stacked, None
        return stacked
