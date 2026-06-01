from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .components import (
    HAS_TORCH_CLUSTER,
    FourierTimeEmbedding,
    MNOBlock,
    _make_mlp,
    build_airfoil_mask,
    compute_surface_frame,
    compute_wall_distance,
    torch_cluster_knn_graph,
)


class SpatiotemporalMNO(nn.Module):
    """Spatial MNO encoder + latent temporal forecaster + spatial decoder.

    The model factorises the problem into:
    1. per-timestep spatial encoding on the point cloud,
    2. per-point temporal forecasting in latent space, and
    3. future spatial refinement + decoding back to velocity residuals.

    This keeps the MNO blocks responsible for geometry-aware spatial mixing
    while replacing the baseline's flattened-history treatment of time with an
    explicit latent sequence model.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        num_modes: int = 256,
        num_heads: int = 8,
        num_blocks: int = 4,
        k: int = 16,
        num_t_in: int = 5,
        num_t_out: int = 5,
        output_channels: int = 3,
        knn_query_chunk_size: int = 1024,
        graph_query_chunk_size: int = 2048,
        use_torch_cluster_knn: bool = True,
        temporal_layers: int = 2,
        activation_checkpointing: bool = False,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.k = k
        self.num_t_in = num_t_in
        self.num_t_out = num_t_out
        self.output_channels = output_channels
        self.knn_query_chunk_size = knn_query_chunk_size
        self.use_torch_cluster_knn = use_torch_cluster_knn and HAS_TORCH_CLUSTER
        self.activation_checkpointing = activation_checkpointing

        num_time_freqs = 16
        self.time_embedding = FourierTimeEmbedding(num_freqs=num_time_freqs)
        time_feat_dim = self.time_embedding.out_dim

        # Current frame velocity + geometry/context features.
        encoder_in_dim = 3 + 3 + time_feat_dim + 1 + 1 + 9
        self.frame_encoder = _make_mlp(
            encoder_in_dim,
            2 * latent_dim,
            latent_dim,
        )
        self.temporal_input_proj = _make_mlp(
            time_feat_dim,
            latent_dim,
            latent_dim,
            num_hidden_layers=1,
        )
        self.future_time_proj = _make_mlp(
            time_feat_dim,
            latent_dim,
            latent_dim,
            num_hidden_layers=1,
        )

        self.encoder_blocks = nn.ModuleList(
            [
                MNOBlock(
                    latent_dim=latent_dim,
                    num_modes=num_modes,
                    num_heads=num_heads,
                    k=k,
                    graph_query_chunk_size=graph_query_chunk_size,
                )
                for _ in range(num_blocks)
            ]
        )
        self.forecast_blocks = nn.ModuleList(
            [
                MNOBlock(
                    latent_dim=latent_dim,
                    num_modes=num_modes,
                    num_heads=num_heads,
                    k=k,
                    graph_query_chunk_size=graph_query_chunk_size,
                )
                for _ in range(num_blocks)
            ]
        )

        self.temporal_input_norm = nn.LayerNorm(latent_dim)
        self.temporal_model = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=max(1, temporal_layers),
            batch_first=True,
        )
        self.temporal_output_norm = nn.LayerNorm(latent_dim)
        self.temporal_predictor = _make_mlp(
            num_t_in * latent_dim,
            2 * latent_dim,
            num_t_out * latent_dim,
        )

        self.decoder = _make_mlp(
            latent_dim,
            2 * latent_dim,
            output_channels,
        )

        final_layer = self.decoder[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    @staticmethod
    def _build_airfoil_mask(
        idcs_airfoil: list[torch.Tensor],
        batch_size: int,
        num_pos: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return build_airfoil_mask(
            idcs_airfoil,
            batch_size,
            num_pos,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _compute_wall_distance(
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
    ) -> torch.Tensor:
        return compute_wall_distance(pos, idcs_airfoil)

    @staticmethod
    def _compute_surface_frame(
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
    ) -> torch.Tensor:
        return compute_surface_frame(pos, idcs_airfoil)

    @staticmethod
    def _pad_neighbors(indices: torch.Tensor, target_k: int) -> torch.Tensor:
        current_k = indices.size(-1)
        if current_k == target_k:
            return indices
        pad = indices[..., -1:].expand(-1, -1, target_k - current_k)
        return torch.cat((indices, pad), dim=-1)

    def _knn_cdist(self, pos: torch.Tensor) -> torch.Tensor:
        batch_size, num_pos, _ = pos.shape
        k_eff = min(self.k, max(1, num_pos - 1))
        all_indices = torch.empty(
            (batch_size, num_pos, k_eff),
            dtype=torch.long,
            device=pos.device,
        )

        for batch_idx in range(batch_size):
            points = pos[batch_idx]
            for start in range(0, num_pos, self.knn_query_chunk_size):
                end = min(start + self.knn_query_chunk_size, num_pos)
                dist = torch.cdist(points[start:end], points)
                row_indices = torch.arange(end - start, device=pos.device)
                col_indices = torch.arange(start, end, device=pos.device)
                dist[row_indices, col_indices] = float("inf")
                all_indices[batch_idx, start:end, :] = torch.topk(
                    dist,
                    k=k_eff,
                    dim=1,
                    largest=False,
                ).indices

        return self._pad_neighbors(all_indices, self.k)

    @torch._dynamo.disable
    def _knn_torch_cluster(self, pos: torch.Tensor) -> torch.Tensor:
        batch_size, num_pos, _ = pos.shape
        k_eff = min(self.k, max(1, num_pos - 1))
        all_indices = torch.empty(
            (batch_size, num_pos, k_eff),
            dtype=torch.long,
            device=pos.device,
        )

        for batch_idx in range(batch_size):
            edge_index = torch_cluster_knn_graph(
                pos[batch_idx],
                k=k_eff,
                loop=False,
            )
            src, dst = edge_index[0], edge_index[1]
            order = torch.argsort(dst)
            all_indices[batch_idx] = src[order].view(num_pos, k_eff)

        return self._pad_neighbors(all_indices, self.k)

    def _build_knn_graph(self, pos: torch.Tensor) -> torch.Tensor:
        if self.use_torch_cluster_knn:
            return self._knn_torch_cluster(pos)
        return self._knn_cdist(pos)

    def _run_spatial_stack(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        knn_indices: torch.Tensor,
        blocks: nn.ModuleList,
    ) -> torch.Tensor:
        for block in blocks:
            if self.activation_checkpointing and self.training:
                x = checkpoint(
                    lambda x_in: block(x_in, pos, knn_indices),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, pos, knn_indices)
        return x

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
        velocity_mean: torch.Tensor | None = None,
        velocity_std: torch.Tensor | None = None,
        return_knn_indices: bool = False,
        wall_distance: torch.Tensor | None = None,
        surface_frame: torch.Tensor | None = None,
        knn_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_t_in, num_pos, _ = velocity_in.shape
        expected_t = self.num_t_in + self.num_t_out

        if num_t_in != self.num_t_in:
            raise ValueError(f"Expected num_t_in={self.num_t_in}, received {num_t_in}.")
        if t.shape != (batch_size, expected_t):
            raise ValueError(
                f"Expected t.shape={(batch_size, expected_t)}, got {tuple(t.shape)}."
            )
        if len(idcs_airfoil) != batch_size:
            raise ValueError(
                "idcs_airfoil must contain one index tensor per batch element."
            )

        time_emb = self.time_embedding(t)
        input_time_emb = time_emb[:, : self.num_t_in, :]
        output_time_emb = time_emb[:, self.num_t_in :, :]

        airfoil_mask = self._build_airfoil_mask(
            idcs_airfoil,
            batch_size,
            num_pos,
            device=pos.device,
            dtype=pos.dtype,
        )

        with torch.no_grad():
            if wall_distance is not None:
                wall_distance_feat = torch.log1p(wall_distance).unsqueeze(-1)
            else:
                wall_distance_feat = self._compute_wall_distance(pos, idcs_airfoil)

            if surface_frame is not None:
                surface_frame_feat = surface_frame
            else:
                surface_frame_feat = self._compute_surface_frame(pos, idcs_airfoil)

            if knn_indices is None:
                knn_indices = self._build_knn_graph(pos)

        encoded_history: list[torch.Tensor] = []
        for step_idx in range(self.num_t_in):
            step_time_feat = input_time_emb[:, step_idx, :].unsqueeze(1).expand(
                -1,
                num_pos,
                -1,
            )
            encoder_input = torch.cat(
                (
                    pos,
                    velocity_in[:, step_idx, :, :],
                    step_time_feat,
                    airfoil_mask,
                    wall_distance_feat,
                    surface_frame_feat,
                ),
                dim=-1,
            )
            x = self.frame_encoder(encoder_input)
            x = self._run_spatial_stack(x, pos, knn_indices, self.encoder_blocks)
            encoded_history.append(x)

        latent_history = torch.stack(encoded_history, dim=1)  # (B, T_in, N, D)
        temporal_bias = self.temporal_input_proj(input_time_emb).unsqueeze(1)
        temporal_input = latent_history.permute(0, 2, 1, 3)  # (B, N, T_in, D)
        temporal_input = self.temporal_input_norm(temporal_input + temporal_bias)
        temporal_input = temporal_input.reshape(
            batch_size * num_pos,
            self.num_t_in,
            self.latent_dim,
        )

        temporal_output, _ = self.temporal_model(temporal_input)
        temporal_output = self.temporal_output_norm(temporal_output)
        temporal_summary = temporal_output.reshape(
            batch_size,
            num_pos,
            self.num_t_in * self.latent_dim,
        )
        future_latent = self.temporal_predictor(temporal_summary).view(
            batch_size,
            num_pos,
            self.num_t_out,
            self.latent_dim,
        ).permute(0, 2, 1, 3)

        last_latent = latent_history[:, -1:, :, :]
        future_latent = future_latent + last_latent
        future_latent = future_latent + self.future_time_proj(output_time_emb).unsqueeze(2)

        residual_frames: list[torch.Tensor] = []
        for step_idx in range(self.num_t_out):
            x = future_latent[:, step_idx, :, :]
            x = self._run_spatial_stack(x, pos, knn_indices, self.forecast_blocks)
            residual_frames.append(self.decoder(x))

        residual = torch.stack(residual_frames, dim=1)
        baseline = velocity_in[:, -1:, :, :].expand(-1, self.num_t_out, -1, -1)
        velocity_out = baseline + residual

        airfoil_mask_bool = airfoil_mask.bool().unsqueeze(1).expand(
            -1,
            self.num_t_out,
            -1,
            self.output_channels,
        )
        if velocity_mean is not None and velocity_std is not None:
            scaled_zero = ((0.0 - velocity_mean) / velocity_std).view(
                batch_size,
                1,
                1,
                self.output_channels,
            )
            scaled_zero = scaled_zero.to(
                device=velocity_out.device,
                dtype=velocity_out.dtype,
            )
            velocity_out = torch.where(airfoil_mask_bool, scaled_zero, velocity_out)
        else:
            velocity_out = velocity_out * (~airfoil_mask_bool).to(dtype=velocity_out.dtype)

        output = velocity_out.contiguous()
        if return_knn_indices:
            return output, knn_indices
        return output
