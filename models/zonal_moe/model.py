import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_scatter import scatter_mean
from .preprocessing import compute_wall_distance, compute_polynomial_baseline


class TemporalEncoder(nn.Module):
    def __init__(self, d_temporal=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.d_temporal = d_temporal
        self.proj = nn.Linear(4, d_temporal)  # 3 velocity + 1 time
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_temporal,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, vel_in, t_vals=None, chunk_size: int = 4096):

        T, N, _ = vel_in.shape
        if t_vals is None:
            t_vals = torch.linspace(0, 1, T, device=vel_in.device)

        t = t_vals.view(T, 1, 1).expand(T, N, 1)
        x = torch.cat([vel_in, t], dim=-1).permute(1, 0, 2)
        x = self.proj(x)  # (N, T, d_temporal)
        outputs = []
        for start in range(0, N, chunk_size):
            chunk = x[start : start + chunk_size]  # (C, T, d_temporal)
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ):
                chunk = self.transformer(chunk)  # (C, T, d_temporal)
            outputs.append(chunk.mean(dim=1))  # (C, d_temporal)

        return torch.cat(outputs, dim=0)  # (N, d_temporal)

class GATLayer(nn.Module):

    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.conv = GATConv(
            in_channels,
            out_channels // heads if concat else out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        return F.elu(x)

class SharedBackbone(nn.Module):

    def __init__(self, in_channels, hidden_channels=128, heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATLayer(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATLayer(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        self.out_channels = hidden_channels * 2  # local + global

    def forward(self, x, edge_index, batch=None):
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        g_global = scatter_mean(x, batch, dim=0)
        g_broadcast = g_global[batch]
        x = torch.cat([x, g_broadcast], dim=-1)
        return x

class RoutingGate(nn.Module):

    def __init__(self, wall_dist_scale=1.0, vorticity_scale=1.0):
        super().__init__()
        self.register_buffer("wall_dist_scale", torch.tensor(wall_dist_scale))
        self.register_buffer("vorticity_scale", torch.tensor(vorticity_scale))

    def set_normalization_stats(self, wall_dist_scale: float, vorticity_scale: float):
        self.wall_dist_scale.fill_(wall_dist_scale)
        self.vorticity_scale.fill_(vorticity_scale)

    def forward(self, wall_dist, vorticity):
        wall_dist_norm = wall_dist / (self.wall_dist_scale + 1e-8)
        vorticity_norm = vorticity / (self.vorticity_scale + 1e-8)
        g_raw = (0.1 - wall_dist_norm) * 20.0 + (vorticity_norm * 5.0)
        g = torch.sigmoid(g_raw)
        return g


class LaminarExpert(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATLayer(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATLayer(hidden_channels, out_channels, heads=1, dropout=dropout, concat=False)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        return x

class TurbulentExpert(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=256, heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.gat1 = GATLayer(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATLayer(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat3 = GATLayer(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat4 = GATLayer(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_index_dense=None):
        ei = edge_index_dense if edge_index_dense is not None else edge_index
        x = self.input_proj(x)
        h = self.gat1(x, ei)
        h = self.gat2(h, ei)
        x = x + h

        h = self.gat3(x, ei)
        h = self.gat4(h, ei)
        x = x + h
        x = self.output_proj(x)
        return x


class ZonalMoE(nn.Module):
    def __init__(
        self,
        d_temporal=64,
        backbone_hidden=128,
        laminar_hidden=64,
        turbulent_hidden=256,
        heads=4,
        dropout=0.1,
        output_timesteps=5,
        output_channels=3,
        wall_dist_scale=1.0,
        vorticity_scale=1.0,
    ):
        super().__init__()

        self.output_timesteps = output_timesteps
        self.output_channels = output_channels

        self.temporal_encoder = TemporalEncoder(
            d_temporal=d_temporal, nhead=heads, num_layers=2, dim_feedforward=128
        )

        backbone_in = d_temporal + 2  
        self.backbone = SharedBackbone(
            in_channels=backbone_in,
            hidden_channels=backbone_hidden,
            heads=heads,
            dropout=dropout,
        )

        expert_in = self.backbone.out_channels
        expert_out = output_timesteps * output_channels  

        self.routing_gate = RoutingGate(
            wall_dist_scale=wall_dist_scale,
            vorticity_scale=vorticity_scale,
        )

        self.laminar_expert = LaminarExpert(
            in_channels=expert_in,
            out_channels=expert_out,
            hidden_channels=laminar_hidden,
            heads=heads,
            dropout=dropout,
        )

        self.turbulent_expert = TurbulentExpert(
            in_channels=expert_in,
            out_channels=expert_out,
            hidden_channels=turbulent_hidden,
            heads=heads,
            dropout=dropout,
        )

    def compute_vorticity_proxy(self, vel_in):
        return (vel_in[-1] - vel_in[0]).norm(dim=-1)

    def forward(
        self,
        vel_in,
        edge_index,
        wall_dist,
        is_airfoil,
        batch=None,
        edge_index_dense=None,
        debug_gate=False,
    ):
        T, N, C = vel_in.shape
        temporal_emb = self.temporal_encoder(vel_in)  # (N, d_temporal)
        vorticity = self.compute_vorticity_proxy(vel_in)  # (N,)
        wall_dist_feat = wall_dist.unsqueeze(-1)
        is_airfoil_feat = is_airfoil.float().unsqueeze(-1)
        node_features = torch.cat([temporal_emb, wall_dist_feat, is_airfoil_feat], dim=-1)
        H = self.backbone(node_features, edge_index, batch)
        g = self.routing_gate(wall_dist, vorticity)
        self._last_g = g.detach()

        #stoopid gate stats printer
        if debug_gate:
            turb_frac = (g > 0.5).float().mean().item()
            print(
                f"  [gate] mean={g.mean():.4f} std={g.std():.4f} turbulent={turb_frac * 100:.1f}%"
            )

        laminar_out = self.laminar_expert(H, edge_index)
        turbulent_out = self.turbulent_expert(H, edge_index, edge_index_dense)
        g_expanded = g.unsqueeze(-1)
        mixed = g_expanded * turbulent_out + (1 - g_expanded) * laminar_out
        residual = mixed.view(N, self.output_timesteps, self.output_channels)
        residual = residual.permute(1, 0, 2)  # (T_out, N, 3)

        return residual

    def predict(
        self,
        vel_in,
        edge_index,
        wall_dist,
        is_airfoil,
        batch=None,
        edge_index_dense=None,
    ):
        residual = self.forward(vel_in, edge_index, wall_dist, is_airfoil, batch, edge_index_dense)
        poly_baseline = compute_polynomial_baseline(vel_in)
        return poly_baseline + residual

    def set_normalization_stats(self, wall_dist_scale: float, vorticity_scale: float):
        self.routing_gate.set_normalization_stats(wall_dist_scale, vorticity_scale)

    def get_routing_stats(self):
        if not hasattr(self, "_last_g") or self._last_g is None:
            return {
                "gate_mean": 0.0,
                "gate_std": 0.0,
                "turbulent_fraction": 0.0,
                "laminar_fraction": 0.0,
            }
        g = self._last_g
        return {
            "gate_mean": g.mean().item(),
            "gate_std": g.std().item(),
            "turbulent_fraction": (g > 0.5).float().mean().item(),
            "laminar_fraction": (g <= 0.5).float().mean().item(),
        }