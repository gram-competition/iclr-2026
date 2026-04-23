from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from models import CANONICAL_MODEL_REGISTRY, get_model_class, normalise_model_name
from src.training.pointcloud_losses import RelativeL2Loss
from src.data import WarpedIFWDataset, build_loader, split_train_val_test
from src.training import NUM_POS, NUM_T_IN, NUM_T_OUT, evaluate, resolve_device, set_seed
from src.utils import apply_parser_defaults_from_config, read_config_defaults_from_cli

# Hyperparameters that must match between training and eval for SpatioTemporalGNN checkpoints.
_STNN_CHECKPOINT_ARG_KEYS: tuple[str, ...] = (
    "latent_dim",
    "num_blocks",
    "num_heads",
    "num_temporal_layers",
    "k",
    "use_so3_backbone",
    "use_steerable_backbone",
    "steerable_lmax",
    "use_pressure",
    "use_relative_velocity_inputs",
)


def _merge_stnn_args_from_checkpoint(
    eval_args: argparse.Namespace, checkpoint: object
) -> argparse.Namespace:
    """Prefer architecture flags stored in training checkpoints (``args`` dict)."""
    merged = copy.copy(eval_args)
    if not isinstance(checkpoint, dict):
        return merged
    saved = checkpoint.get("args")
    if not isinstance(saved, dict):
        return merged
    for key in _STNN_CHECKPOINT_ARG_KEYS:
        if key in saved:
            setattr(merged, key, saved[key])
    return merged


def _build_spatio_temporal_gnn(
    eval_args: argparse.Namespace,
    device: torch.device,
    *,
    load_bundled_weights: bool = True,
):
    """Match ``trainer.train`` construction for ``spatio_temporal_gnn``."""
    use_so3 = bool(getattr(eval_args, "use_so3_backbone", False))
    use_steerable = bool(getattr(eval_args, "use_steerable_backbone", False))
    if use_so3 and use_steerable:
        raise ValueError(
            "Checkpoint / args enable both use_so3_backbone and use_steerable_backbone; "
            "they are mutually exclusive."
        )
    model_cls = get_model_class("spatio_temporal_gnn")
    return model_cls(
        num_t_in=NUM_T_IN,
        num_t_out=NUM_T_OUT,
        hidden_dim=int(eval_args.latent_dim),
        num_layers=int(eval_args.num_blocks),
        heads=int(eval_args.num_heads),
        k=int(eval_args.k),
        num_attn_layers=int(eval_args.num_temporal_layers),
        use_so3_backbone=use_so3,
        use_steerable_backbone=use_steerable,
        steerable_lmax=int(getattr(eval_args, "steerable_lmax", 2)),
        use_pressure=bool(getattr(eval_args, "use_pressure", True)),
        use_relative_velocity_inputs=bool(
            getattr(eval_args, "use_relative_velocity_inputs", False)
        ),
        load_bundled_weights=load_bundled_weights,
    ).to(device)


def parse_eval_args() -> argparse.Namespace:
    config_defaults = read_config_defaults_from_cli()

    parser = argparse.ArgumentParser(
        description="Evaluate a trained MNO checkpoint on train/val/test split."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional YAML config file. CLI arguments override YAML values.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset_huggingface/warped-ifw",
        help="Directory containing .npz files.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        choices=sorted(
            set(CANONICAL_MODEL_REGISTRY)
            | {"stnn", "fmgreco_stnn", "mlp"}
        ),
        default="spatio_temporal_gnn",
        help="Model family to evaluate (this branch: SpatioTemporalGNN only).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor (only used when --num-workers > 0).",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DataLoader workers alive across evaluation.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=NUM_POS,
        help="Number of points per sample during evaluation.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Validation fraction of groups.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.05,
        help="Test fraction of groups.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent channel width for MNO blocks.",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=256,
        help="Number of global attention modes.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help=(
            "For SpatioTemporalGNN: MultiheadAttention heads in the temporal encoder. "
            "For other models: attention head count as defined by that architecture."
        ),
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=12,
        help=(
            "For SpatioTemporalGNN: number of MeshGraphNet message-passing layers. "
            "For MNO-style models: number of blocks."
        ),
    )
    parser.add_argument(
        "--num-temporal-layers",
        type=int,
        default=2,
        help=(
            "SpatioTemporalGNN only: number of TransformerEncoder layers in the temporal head."
        ),
    )
    parser.add_argument(
        "--use-so3-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "SpatioTemporalGNN only: SO(3)-equivariant k-NN backbone instead of MeshGraphNet."
        ),
    )
    parser.add_argument(
        "--use-steerable-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "SpatioTemporalGNN only: e3nn steerable backbone (mutually exclusive with "
            "--use-so3-backbone)."
        ),
    )
    parser.add_argument(
        "--steerable-lmax",
        type=int,
        default=2,
        help="SpatioTemporalGNN + steerable: max spherical-harmonic order on edges.",
    )
    parser.add_argument(
        "--use-pressure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "SpatioTemporalGNN: concatenate standardized input-frame pressure into the node encoder."
        ),
    )
    parser.add_argument(
        "--use-relative-velocity-inputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "SpatioTemporalGNN: encode v_t - v_last (after no-slip zeroing) in the node encoder."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=16,
        help="k in kNN graph construction for local attention.",
    )
    parser.add_argument(
        "--knn-query-chunk-size",
        type=int,
        default=1024,
        help="Chunk size for cdist fallback kNN queries.",
    )
    parser.add_argument(
        "--graph-query-chunk-size",
        type=int,
        default=2048,
        help="Chunk size for local graph attention queries.",
    )
    parser.add_argument(
        "--use-torch-cluster-knn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch_cluster kNN graph backend when available.",
    )
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA automatic mixed precision for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "Execution device selection. Use 'cuda' for GPU jobs so failures do "
            "not silently fall back to CPU."
        ),
    )
    parser.add_argument(
        "--scaler-eps",
        type=float,
        default=1e-6,
        help="Lower bound for velocity standard-deviation during scaling.",
    )
    parser.add_argument(
        "--no-slip-atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for optional no-slip boundary assertions.",
    )
    parser.add_argument(
        "--assert-no-slip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable no-slip assertions during evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="models/fmgreco_stnn/state_dict.pt",
        help="Checkpoint path for evaluation.",
    )

    apply_parser_defaults_from_config(
        parser,
        config_defaults,
        strict_unknown_keys=False,
    )

    return parser.parse_args()


def evaluate_checkpoint(args: argparse.Namespace) -> tuple[float, float, float]:
    set_seed(args.seed)

    device = resolve_device(getattr(args, "device", "auto"))
    use_amp = bool(getattr(args, "use_amp", True)) and device.type == "cuda"

    dataset_dir = Path(args.dataset_dir)
    files = sorted(dataset_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {dataset_dir}.")

    train_files, val_files, test_files = split_train_val_test(
        files,
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
        seed=int(args.seed),
    )
    split_to_files = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }
    selected_files = split_to_files[args.split]

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path.resolve()}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stnn_args = _merge_stnn_args_from_checkpoint(args, checkpoint)
    if (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("args"), dict)
        and any(k in checkpoint["args"] for k in _STNN_CHECKPOINT_ARG_KEYS)
    ):
        print(
            "SpatioTemporalGNN: using architecture hyperparameters from checkpoint['args'] "
            "(where present) so evaluation matches training."
        )

    dataset = WarpedIFWDataset(
        selected_files,
        num_points=int(args.num_points),
        random_crop=False,
        seed=int(args.seed),
        scaler_eps=float(args.scaler_eps),
    )
    loader = build_loader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        prefetch_factor=int(args.prefetch_factor),
        persistent_workers=bool(args.persistent_workers),
    )

    resolved_model_name = normalise_model_name(getattr(args, "model_name", "mlp"))
    model_cls = get_model_class(resolved_model_name)
    if resolved_model_name == "spatio_temporal_gnn":
        model = _build_spatio_temporal_gnn(
            stnn_args, device, load_bundled_weights=False
        )
    elif resolved_model_name in ("gated_egno_mean_res", "transolver_ar"):
        model = model_cls().to(device)
    elif resolved_model_name == "delta_graph":
        model = model_cls(load_weights=False).to(device)
    elif resolved_model_name == "cdf_double_grid_net":
        model = model_cls(load_weights=False).to(device)
    else:
        model = model_cls(
            num_t_in=NUM_T_IN,
            num_t_out=NUM_T_OUT,
            latent_dim=int(args.latent_dim),
            num_modes=int(args.num_modes),
            num_heads=int(args.num_heads),
            num_blocks=int(args.num_blocks),
            k=int(args.k),
            knn_query_chunk_size=int(args.knn_query_chunk_size),
            graph_query_chunk_size=int(args.graph_query_chunk_size),
            use_torch_cluster_knn=bool(args.use_torch_cluster_knn),
        ).to(device)

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=True)

    loss_fn = RelativeL2Loss()

    eval_loss, eval_metric, _ = evaluate(
        model,
        loader,
        loss_fn,
        device,
        no_slip_atol=float(args.no_slip_atol),
        use_amp=use_amp,
        assert_no_slip=bool(args.assert_no_slip),
    )

    print(f"Using device: {device}")
    print(
        f"Evaluated split={args.split} | samples={len(dataset)} | "
        f"points_per_sample={args.num_points}"
    )
    print(f"Model: {resolved_model_name} ({model_cls.__name__})")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eval RL2 (scaled): {eval_loss:.6f}")
    print(f"Eval HINT (unscaled): {eval_metric:.6f}")

    return eval_loss, eval_metric, 0.0


if __name__ == "__main__":
    evaluate_checkpoint(parse_eval_args())
