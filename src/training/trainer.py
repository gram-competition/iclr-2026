from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models import (
    CANONICAL_MODEL_REGISTRY,
    get_model_class,
    normalise_model_name,
)
from src.training.pointcloud_losses import (
    SobolevLoss,
    build_training_components,
    compute_wall_distance_weights,
)
from src.data import (
    WarpedIFWDataset,
    build_loader,
    resolve_overfit_file,
    split_train_val_kfold,
    split_train_val_test,
    unscale_velocity_batch,
)
from src.training.loop import (
    assert_no_slip_boundary,
    autocast_context,
    evaluate,
    forward_model,
    hint_metric,
    move_batch_to_device,
    run_full_test_inference,
)
from src.utils import (
    apply_parser_defaults_from_config,
    read_config_defaults_from_cli,
)
from src.utils.training_history import TrainingEpochRecord, persist_training_artifacts


NUM_T_IN = 5
NUM_T_OUT = 5
NUM_POS = 100000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _torch_build_is_rocm() -> bool:
    return bool(getattr(torch.version, "hip", None))


def _configure_cuda_backends(device: torch.device, *, rank: int) -> None:
    """NVIDIA: TF32 on. ROCm/HIP: TF32 off + math-only SDPA (efficient kernels often error)."""
    if device.type != "cuda":
        return
    if _torch_build_is_rocm():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for name, enable in (
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_math_sdp", True),
        ):
            fn = getattr(torch.backends.cuda, name, None)
            if callable(fn):
                fn(enable)
        if rank == 0:
            print(
                "ROCm/HIP: disabled TF32; SDPA backends = math only "
                "(avoids MultiheadAttention HIP invalid-argument on some wheels)."
            )
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def resolve_model_class(model_name: str):
    normalised = normalise_model_name(model_name)
    return get_model_class(normalised)


def _core_module(model: torch.nn.Module) -> torch.nn.Module:
    m = model.module if isinstance(model, DDP) else model
    return m._orig_mod if hasattr(m, "_orig_mod") else m


def _eval_module(model: torch.nn.Module) -> torch.nn.Module:
    """Rank-0-only eval must not use DDP wrapper: forwards would wait on other ranks (NCCL hang)."""
    return _core_module(model)


def resolve_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device selection: {device_arg}")

    if requested == "cpu":
        return torch.device("cpu")

    cuda_available = torch.cuda.is_available()
    if requested == "cuda":
        if cuda_available:
            return torch.device("cuda")
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
        slurm_hint = f" for SLURM job {slurm_job_id}" if slurm_job_id else ""
        raise RuntimeError(
            "CUDA was explicitly requested but is unavailable"
            f"{slurm_hint}. Check that the allocated node exposes a compatible "
            "GPU driver/runtime for this PyTorch build. "
            f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}."
        )

    return torch.device("cuda" if cuda_available else "cpu")


def _format_cuda_memory_stats(device: torch.device) -> str:
    gib = float(1024 ** 3)
    allocated = torch.cuda.memory_allocated(device) / gib
    reserved = torch.cuda.memory_reserved(device) / gib
    peak_allocated = torch.cuda.max_memory_allocated(device) / gib
    peak_reserved = torch.cuda.max_memory_reserved(device) / gib
    return (
        f"cuda_mem_allocated_gib={allocated:.2f} | "
        f"cuda_mem_reserved_gib={reserved:.2f} | "
        f"cuda_peak_allocated_gib={peak_allocated:.2f} | "
        f"cuda_peak_reserved_gib={peak_reserved:.2f}"
    )


def _get_cuda_memory_stats(device: torch.device) -> dict[str, float]:
    gib = float(1024 ** 3)
    return {
        "cuda_mem_allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "cuda_mem_reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
    }


def _initialise_wandb_run(
    args: argparse.Namespace,
    *,
    run_name: str,
    run_dir: Path,
    resolved_model_name: str,
    device: torch.device,
) -> object | None:
    if not bool(getattr(args, "use_wandb", False)):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested but wandb is not installed. "
            "Install it with 'pip install wandb' or disable --use-wandb."
        ) from exc

    project_name = str(getattr(args, "wandb_project", "") or "iclr_develop").strip()
    wandb_run_name = str(getattr(args, "wandb_run_name", "") or run_name).strip()
    wandb_dir = str(getattr(args, "wandb_dir", "") or run_dir).strip()
    wandb_mode = str(getattr(args, "wandb_mode", "") or "online").strip()

    config = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            config[key] = value
        else:
            config[key] = str(value)
    config["resolved_model_name"] = resolved_model_name
    config["run_dir"] = str(run_dir)
    config["device"] = str(device)

    return wandb.init(
        project=project_name,
        name=wandb_run_name,
        dir=wandb_dir,
        mode=wandb_mode,
        config=config,
        tags=[resolved_model_name, str(device)],
    )


def _wandb_log_artifact(
    wandb_run: object,
    path: Path,
    *,
    artifact_name: str,
    artifact_type: str,
) -> None:
    import wandb

    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(str(path))
    wandb_run.log_artifact(artifact)


def _model_state_dict_for_checkpoint(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    m = model.module if isinstance(model, DDP) else model
    if hasattr(m, "_orig_mod"):
        return m._orig_mod.state_dict()
    return m.state_dict()


def _load_model_state_from_checkpoint(
    model: torch.nn.Module,
    checkpoint: object,
) -> dict[str, object]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint
    else:
        state_dict = checkpoint
        metadata = {}

    target_model = model.module if isinstance(model, DDP) else model
    target_model = target_model._orig_mod if hasattr(target_model, "_orig_mod") else target_model
    target_model.load_state_dict(state_dict)
    return metadata if isinstance(metadata, dict) else {}


def _save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_metric: float,
    args: argparse.Namespace,
) -> None:
    payload = {
        "model_state_dict": _model_state_dict_for_checkpoint(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_val_metric": best_val_metric,
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(payload, path)


def train(args: argparse.Namespace) -> None:
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    device_arg = str(getattr(args, "device", "auto")).strip().lower()
    distributed = world_size_env > 1 and device_arg != "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed:
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError(
                "Multi-GPU training requires torchrun (LOCAL_RANK missing). Example: "
                "torchrun --standalone --nproc_per_node=8 main.py ... --device cuda"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requested but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda", local_rank)
    else:
        if world_size_env > 1:
            raise RuntimeError(
                f"WORLD_SIZE={world_size_env} but device is cpu; use --device cuda for multi-GPU."
            )
        rank = 0
        world_size = 1
        device = resolve_device(getattr(args, "device", "auto"))

    set_seed(args.seed)

    use_amp = bool(getattr(args, "use_amp", True)) and device.type == "cuda"
    if device.type == "cuda" and bool(getattr(args, "cudnn_benchmark", True)):
        torch.backends.cudnn.benchmark = True
    _configure_cuda_backends(device, rank=rank)

    dataset_dir = Path(args.dataset_dir)
    files = sorted(dataset_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {dataset_dir}.")

    overfit_single = args.overfit_single or bool(args.overfit_file)
    cv_fold = getattr(args, "cv_fold", None)
    num_cv_folds = int(getattr(args, "num_cv_folds", 8))
    if overfit_single and cv_fold is not None:
        raise ValueError("Cannot use --cv-fold together with overfit / --overfit-file.")
    train_random_crop = args.random_subsample
    val_fraction = float(getattr(args, "val_fraction", 0.05))
    test_fraction = float(getattr(args, "test_fraction", 0.05))
    test_batch_size = int(getattr(args, "test_batch_size", 1))
    prefetch_factor = int(getattr(args, "prefetch_factor", 2))
    persistent_workers = bool(getattr(args, "persistent_workers", True))
    assert_no_slip_train = bool(getattr(args, "assert_no_slip_train", False))
    assert_no_slip_val = bool(getattr(args, "assert_no_slip_val", False))
    assert_no_slip_test = bool(getattr(args, "assert_no_slip_test", False))
    precompute_knn = bool(getattr(args, "precompute_knn", False))
    knn_cache_dir_value = getattr(args, "knn_cache_dir", "")
    knn_cache_dir = Path(knn_cache_dir_value) if knn_cache_dir_value else None
    if precompute_knn and knn_cache_dir is None:
        scratch_root = Path(os.environ.get("MCMLSCRATCH", "/dss/mcmlscratch/08/di97hih"))
        knn_cache_dir = scratch_root / "outputs" / "knn_cache"
    knn_include_neighbors = bool(getattr(args, "knn_include_neighbors", False))
    knn_overflow_points = max(0, int(getattr(args, "knn_overflow_points", 0)))
    if knn_include_neighbors and args.batch_size > 1 and knn_overflow_points <= 0:
        raise ValueError(
            "--knn-include-neighbors can expand each crop to a different point "
            "count unless --knn-overflow-points is set. Disable "
            "--knn-include-neighbors, use --batch-size 1, or set a fixed "
            "--knn-overflow-points budget."
        )

    if overfit_single:
        target_file = resolve_overfit_file(
            files,
            overfit_file=args.overfit_file,
            overfit_index=args.overfit_index,
        )
        repeats = max(1, args.overfit_repeats)
        train_files = [target_file for _ in range(repeats)]
        val_files = [target_file]
        test_files = [target_file]
        train_random_crop = False
        if rank == 0:
            print(
                "Overfit mode enabled | "
                f"file={target_file.name} | repeats_per_epoch={repeats} | "
                "random_crop=False"
            )
    elif cv_fold is not None:
        train_files, val_files, test_files = split_train_val_kfold(
            files,
            num_folds=num_cv_folds,
            fold_index=int(cv_fold),
            seed=args.seed,
        )
        if rank == 0:
            print(
                f"K-fold split | num_folds={num_cv_folds} | val_fold={int(cv_fold)} | "
                f"train_groups via round-robin | no held-out test split (test_files=0)"
            )
    else:
        train_files, val_files, test_files = split_train_val_test(
            files,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=args.seed,
        )

    train_dataset = WarpedIFWDataset(
        train_files,
        num_points=args.num_points,
        random_crop=train_random_crop,
        seed=args.seed,
        scaler_eps=args.scaler_eps,
        boundary_point_fraction=float(getattr(args, "boundary_point_fraction", 0.0)),
        precompute_knn=precompute_knn,
        knn_cache_dir=knn_cache_dir,
        knn_k=args.k,
        knn_include_neighbors=knn_include_neighbors,
        knn_overflow_points=knn_overflow_points,
        augment_xy_plane_isometry=bool(
            getattr(args, "augment_xy_plane_isometry", False)
        ),
    )
    val_dataset = WarpedIFWDataset(
        val_files,
        num_points=args.num_points,
        random_crop=False,
        seed=args.seed,
        scaler_eps=args.scaler_eps,
        precompute_knn=precompute_knn,
        knn_cache_dir=knn_cache_dir,
        knn_k=args.k,
    )

    train_sampler: DistributedSampler | None = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        sampler=train_sampler,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    boundary_frac = float(getattr(args, "boundary_point_fraction", 0.0))
    if rank == 0:
        print(f"Using device: {device}")
        if distributed:
            print(
                f"DistributedDataParallel | world_size={world_size} global_rank={rank} "
                f"local_rank={local_rank}"
            )
        print(
            f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, "
            f"Test samples: {len(test_files)}, Train/Val points/sample: {args.num_points}, "
            f"Test points/sample: {NUM_POS}"
        )
        if boundary_frac > 0.0:
            print(
                f"Boundary-biased sampling: {boundary_frac:.0%} of point budget reserved "
                f"for near-airfoil points (training only)"
            )
        if precompute_knn:
            print(
                f"Precomputed kNN lookup: enabled | cache_dir={knn_cache_dir} | "
                f"k={args.k} | include_neighbors={knn_include_neighbors} | "
                f"overflow_points={knn_overflow_points}"
            )
        if getattr(args, "augment_xy_plane_isometry", False):
            print("XY-plane isometry augmentation: enabled (pos + velocity as vectors).")

    resolved_model_name = normalise_model_name(getattr(args, "model_name", "mlp"))
    model_cls = resolve_model_class(resolved_model_name)
    model_kwargs = dict(
        num_t_in=NUM_T_IN,
        num_t_out=NUM_T_OUT,
        latent_dim=args.latent_dim,
        num_modes=args.num_modes,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        k=args.k,
        knn_query_chunk_size=args.knn_query_chunk_size,
        graph_query_chunk_size=args.graph_query_chunk_size,
        use_torch_cluster_knn=args.use_torch_cluster_knn,
    )
    if resolved_model_name == "spatiotemporal_mno":
        model_kwargs["activation_checkpointing"] = bool(
            getattr(args, "activation_checkpointing", False)
        )
    if resolved_model_name in ("gated_egno_mean_res", "transolver_ar"):
        # gEGNO (PR #10) & TransolverAR (PR #21): no-arg __init__, fixed hyperparameters.
        model = model_cls().to(device)
    elif resolved_model_name == "delta_graph":
        # PR #25: weights loaded from training checkpoint, not the submission bundle.
        model = model_cls(load_weights=False).to(device)
    elif resolved_model_name == "cdf_double_grid_net":
        # PR #17: optional HF weights; trainer uses checkpoint / random init.
        model = model_cls(load_weights=False).to(device)
    elif resolved_model_name == "spatio_temporal_gnn":
        # Map MNO-style CLI names to STNN: latent_dim -> width, num_blocks -> depth.
        use_so3 = bool(getattr(args, "use_so3_backbone", False))
        use_steerable = bool(getattr(args, "use_steerable_backbone", False))
        if use_so3 and use_steerable:
            raise ValueError(
                "Cannot enable both --use-so3-backbone and --use-steerable-backbone."
            )
        model = model_cls(
            num_t_in=NUM_T_IN,
            num_t_out=NUM_T_OUT,
            hidden_dim=int(args.latent_dim),
            num_layers=int(args.num_blocks),
            heads=int(args.num_heads),
            k=int(args.k),
            num_attn_layers=int(args.num_temporal_layers),
            use_so3_backbone=use_so3,
            use_steerable_backbone=use_steerable,
            steerable_lmax=int(getattr(args, "steerable_lmax", 2)),
            use_pressure=bool(getattr(args, "use_pressure", True)),
            use_relative_velocity_inputs=bool(
                getattr(args, "use_relative_velocity_inputs", False)
            ),
            load_bundled_weights=bool(getattr(args, "load_bundled_weights", True)),
        ).to(device)
    else:
        model = model_cls(**model_kwargs).to(device)
    if rank == 0:
        print(f"Model: {resolved_model_name} ({model_cls.__name__})")
        if resolved_model_name == "spatio_temporal_gnn":
            st = _core_module(model)
            if getattr(st, "use_steerable_backbone", False):
                print(
                    "SpatioTemporalGNN: steerable e3nn backbone | "
                    f"lmax={getattr(st, 'steerable_lmax', '?')} | irreps={getattr(st, 'irreps', None)}"
                )
            elif getattr(st, "use_so3_backbone", False):
                print("SpatioTemporalGNN: SO(3) hand-equivariant backbone (equivariant.py)")
            if getattr(st, "use_pressure", False):
                print(
                    "SpatioTemporalGNN: pressure node features enabled "
                    f"(relative Δv inputs={getattr(st, 'use_relative_velocity_inputs', False)})"
                )
        if resolved_model_name == "spatiotemporal_mno":
            print(
                "Activation checkpointing: "
                f"{bool(getattr(args, 'activation_checkpointing', False))}"
            )

    if getattr(args, "compile_model", False):
        model = torch.compile(model)
        if rank == 0:
            print("Model compiled with torch.compile")

    if rank == 0:
        if hasattr(model, "use_torch_cluster_knn"):
            knn_backend = "torch_cluster" if model.use_torch_cluster_knn else "cdist_fallback"
            print(f"kNN backend: {knn_backend}")
        else:
            print("kNN backend: n/a (model does not expose use_torch_cluster_knn)")

    if (
        rank == 0
        and args.use_torch_cluster_knn
        and hasattr(model, "use_torch_cluster_knn")
        and not model.use_torch_cluster_knn
    ):
        print(
            "Warning: torch_cluster was requested but is unavailable. "
            "Training will use a slower cdist-based kNN fallback."
        )

    resume_from_value = getattr(args, "resume_from", "")
    resume_from_path: Path | None = None
    resume_checkpoint: dict[str, object] = {}
    if resume_from_value:
        resume_from_path = Path(resume_from_value)
        if not resume_from_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_from_path.resolve()}"
            )
        resume_payload = torch.load(resume_from_path, map_location=device, weights_only=False)
        resume_checkpoint = _load_model_state_from_checkpoint(model, resume_payload)
        if rank == 0:
            print(f"Loaded initial weights from checkpoint: {resume_from_path}")

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    loss_fn_name = str(getattr(args, "loss_fn", "rl2")).lower()
    sobolev_grad_weight = float(getattr(args, "sobolev_grad_weight", 0.1))
    use_sobolev = loss_fn_name == "sobolev"
    requested_lr_scheduler = str(getattr(args, "lr_scheduler", "auto")).lower()
    if requested_lr_scheduler == "auto":
        effective_lr_scheduler = "fixed" if overfit_single else "one-cycle"
    else:
        effective_lr_scheduler = requested_lr_scheduler
    training = build_training_components(
        model,
        steps_per_epoch=max(
            1,
            math.ceil(
                len(train_loader)
                / max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
            ),
        ),
        max_lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        lr_scheduler_name=effective_lr_scheduler,
        loss_fn_name=loss_fn_name,
        sobolev_grad_weight=sobolev_grad_weight,
    )
    if rank == 0:
        print(f"Loss function: {loss_fn_name}" + (f" (grad_weight={sobolev_grad_weight})" if use_sobolev else ""))
        print(f"LR scheduler: {effective_lr_scheduler}")
    wall_distance_loss_alpha = float(getattr(args, "wall_distance_loss_alpha", 0.0))
    divergence_loss_lambda = float(getattr(args, "divergence_loss_lambda", 0.0))
    steerable_temporal_grad_weight = float(
        getattr(args, "steerable_temporal_grad_weight", 0.0)
    )
    train_loss_on_velocity_delta = bool(
        getattr(args, "train_loss_on_velocity_delta", False)
    )
    use_wall_weights = wall_distance_loss_alpha > 0.0
    if rank == 0:
        if use_wall_weights:
            print(f"Wall-distance weighted loss: alpha={wall_distance_loss_alpha}")
        if divergence_loss_lambda > 0.0:
            print(
                f"k-NN flux divergence penalty (train only, unscaled velocity): "
                f"lambda={divergence_loss_lambda}"
            )
        if steerable_temporal_grad_weight > 0.0:
            print(
                "Steerable temporal-gradient auxiliary loss (train only): "
                f"weight={steerable_temporal_grad_weight}"
            )
        if train_loss_on_velocity_delta:
            print(
                "Training loss uses velocity residuals w.r.t. last input frame "
                "(scaled space); val/test still report full-field RL2."
            )
    optimizer = training.optimizer
    scheduler = training.scheduler
    loss_fn = training.loss_fn

    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    runs_dir = Path(getattr(args, "runs_dir", "outputs/runs"))
    if not runs_dir.is_absolute():
        runs_dir = project_root / runs_dir
    default_run_name = f"{checkpoint_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    history_run_name = getattr(args, "history_run_name", "") or default_run_name
    run_dir = runs_dir / history_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    run_checkpoint_path = run_dir / "checkpoints" / "best.pt"
    run_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = run_dir / "checkpoints" / "latest.pt"

    run_config_path = run_dir / "config.json"
    if rank == 0:
        with run_config_path.open("w", encoding="utf-8") as file_obj:
            json.dump(vars(args), file_obj, indent=2, sort_keys=True)

    no_slip_atol = float(getattr(args, "no_slip_atol", 0.0))
    gradient_accumulation_steps = max(
        1, int(getattr(args, "gradient_accumulation_steps", 1))
    )
    epoch_history: list[TrainingEpochRecord] = []
    scaler_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(scaler_device, enabled=use_amp)
    start_epoch = 1
    if resume_checkpoint:
        optimizer_state = resume_checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            if rank == 0:
                print("Loaded optimizer state from checkpoint.")
        scheduler_state = resume_checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(scheduler_state)
            if rank == 0:
                print("Loaded scheduler state from checkpoint.")
        scaler_state = resume_checkpoint.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
            if rank == 0:
                print("Loaded AMP scaler state from checkpoint.")
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        if resume_checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(resume_checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and resume_checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng_state_all"])
        if resume_checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(resume_checkpoint["numpy_rng_state"])
        if resume_checkpoint.get("python_rng_state") is not None:
            random.setstate(resume_checkpoint["python_rng_state"])
        if rank == 0:
            print(f"Resume training state: next epoch={start_epoch}")

    if rank == 0:
        print(f"No-slip boundary assertion atol (scaled): {no_slip_atol:.3e}")
        print(
            f"AMP enabled: {use_amp} | workers={args.num_workers} | "
            f"persistent_workers={persistent_workers} | prefetch_factor={prefetch_factor}"
        )
        print(
            f"Batch size: {args.batch_size} | "
            f"gradient_accumulation_steps={gradient_accumulation_steps} | "
            f"effective_batch_size={args.batch_size * gradient_accumulation_steps}"
        )
        print(f"Run directory: {run_dir}")

    wandb_run = None
    if rank == 0:
        wandb_run = _initialise_wandb_run(
            args,
            run_name=history_run_name,
            run_dir=run_dir,
            resolved_model_name=resolved_model_name,
            device=device,
        )

    try:
        best_val_metric = float(resume_checkpoint.get("best_val_metric", float("inf")))
        if resume_from_path is not None and not resume_checkpoint:
            if distributed:
                dist.barrier()
            resumed_val_metric = 0.0
            if rank == 0:
                _, resumed_val_metric, _ = evaluate(
                    _eval_module(model),
                    val_loader,
                    loss_fn,
                    device,
                    no_slip_atol=no_slip_atol,
                    use_amp=use_amp,
                    assert_no_slip=assert_no_slip_val,
                    use_sobolev=use_sobolev,
                    wall_distance_loss_alpha=wall_distance_loss_alpha,
                )
                best_val_metric = resumed_val_metric
                # Seed checkpoint_path so test-time loading always has a valid artifact,
                # even if continuation does not improve on resumed validation metric.
                _save_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=0,
                    best_val_metric=best_val_metric,
                    args=args,
                )
                _save_training_checkpoint(
                    run_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=0,
                    best_val_metric=best_val_metric,
                    args=args,
                )
                print(
                    "Resume mode: initialized best val hint metric from loaded checkpoint "
                    f"to {best_val_metric:.6f}."
                )
                print(f"Seeded checkpoint path with resumed weights: {checkpoint_path}")
                if wandb_run is not None:
                    wandb_run.log({"resume_val_hint_unscaled": resumed_val_metric}, step=0)
            if distributed:
                dist.barrier()

        for epoch in range(start_epoch, training.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            train_loss = 0.0
            train_metric = 0.0
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            for batch_index, batch in enumerate(train_loader, start=1):
                (
                    t,
                    pos,
                    idcs_airfoil,
                    velocity_in,
                    velocity_out,
                    pressure_in,
                    velocity_mean,
                    velocity_std,
                    wall_distance,
                    surface_frame,
                    batch_knn_indices,
                ) = move_batch_to_device(batch, device)

                with autocast_context(device, enabled=use_amp):
                    pred_scaled, knn_indices = forward_model(
                        model,
                        t,
                        pos,
                        idcs_airfoil,
                        velocity_in,
                        velocity_mean,
                        velocity_std,
                        return_knn_indices=use_sobolev,
                        pressure_in=pressure_in,
                        wall_distance=wall_distance,
                        surface_frame=surface_frame,
                        knn_indices=batch_knn_indices,
                    )
                    wall_weights = None
                    if use_wall_weights:
                        with torch.no_grad():
                            wall_weights = compute_wall_distance_weights(
                                pos, idcs_airfoil, alpha=wall_distance_loss_alpha,
                                wall_distance=wall_distance,
                            )
                    pred_loss = pred_scaled
                    tgt_loss = velocity_out
                    if train_loss_on_velocity_delta:
                        baseline = velocity_in[:, -1:, :, :]
                        pred_loss = pred_scaled - baseline
                        tgt_loss = velocity_out - baseline
                    if use_sobolev:
                        direct_loss = loss_fn(
                            pred_loss,
                            tgt_loss,
                            knn_indices=knn_indices,
                            pos=pos,
                            wall_weights=wall_weights,
                        )
                    else:
                        direct_loss = loss_fn(
                            pred_loss, tgt_loss, wall_weights=wall_weights
                        )
                    if divergence_loss_lambda > 0.0:
                        from models.fmgreco_stnn.graph_utils import knn_flux_divergence_loss

                        mcore = _core_module(model)
                        if mcore.__class__.__name__ == "SpatioTemporalGNN":
                            stnn_k = int(getattr(mcore, "k", 12))
                            pred_phys = unscale_velocity_batch(
                                pred_scaled.float(),
                                velocity_mean,
                                velocity_std,
                            )
                            div_sum = torch.zeros(
                                (),
                                device=pred_scaled.device,
                                dtype=torch.float32,
                            )
                            bsz = pos.size(0)
                            for bi in range(bsz):
                                u = pred_phys[bi].mean(dim=0)
                                div_sum = div_sum + knn_flux_divergence_loss(
                                    pos[bi].float(),
                                    u,
                                    stnn_k,
                                ).float()
                            direct_loss = direct_loss + divergence_loss_lambda * (
                                div_sum / max(1, bsz)
                            )
                    if (
                        steerable_temporal_grad_weight > 0.0
                        and NUM_T_OUT > 1
                    ):
                        mcore = _core_module(model)
                        if (
                            mcore.__class__.__name__ == "SpatioTemporalGNN"
                            and getattr(mcore, "use_steerable_backbone", False)
                        ):
                            pd = pred_scaled[:, 1:] - pred_scaled[:, :-1]
                            td = velocity_out[:, 1:] - velocity_out[:, :-1]
                            direct_loss = direct_loss + steerable_temporal_grad_weight * torch.nn.functional.mse_loss(
                                pd, td
                            )
                    loss = direct_loss / gradient_accumulation_steps

                if assert_no_slip_train:
                    assert_no_slip_boundary(
                        pred_scaled,
                        idcs_airfoil,
                        atol=no_slip_atol,
                        context="train",
                        velocity_mean=velocity_mean,
                        velocity_std=velocity_std,
                    )

                should_step = (
                    batch_index % gradient_accumulation_steps == 0
                    or batch_index == len(train_loader)
                )
                sync_ctx = (
                    model.no_sync()
                    if isinstance(model, DDP)
                    and gradient_accumulation_steps > 1
                    and not should_step
                    else nullcontext()
                )
                with sync_ctx:
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if should_step:
                    if use_amp:
                        if args.grad_clip > 0.0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), args.grad_clip
                            )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if args.grad_clip > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), args.grad_clip
                            )
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                train_loss += direct_loss.item()

                with torch.no_grad():
                    pred_unscaled = unscale_velocity_batch(
                        pred_scaled.float(),
                        velocity_mean,
                        velocity_std,
                    )
                    target_unscaled = unscale_velocity_batch(
                        velocity_out.float(),
                        velocity_mean,
                        velocity_std,
                    )
                    train_metric += hint_metric(pred_unscaled, target_unscaled).item()

            nb_local = max(1, len(train_loader))
            stats_t = torch.tensor(
                [train_loss, train_metric, float(nb_local)],
                device=device,
                dtype=torch.float64,
            )
            if distributed:
                dist.all_reduce(stats_t, op=dist.ReduceOp.SUM)
            denom = max(1.0, stats_t[2].item())
            avg_train_loss = (stats_t[0] / denom).item()
            avg_train_metric = (stats_t[1] / denom).item()

            if distributed:
                dist.barrier()
            avg_val_loss = 0.0
            avg_val_metric = 0.0
            if rank == 0:
                avg_val_loss, avg_val_metric, _ = evaluate(
                    _eval_module(model),
                    val_loader,
                    loss_fn,
                    device,
                    no_slip_atol=no_slip_atol,
                    use_amp=use_amp,
                    assert_no_slip=assert_no_slip_val,
                    use_sobolev=use_sobolev,
                    wall_distance_loss_alpha=wall_distance_loss_alpha,
                )
            if distributed:
                val_t = torch.tensor(
                    [avg_val_loss, avg_val_metric],
                    device=device,
                    dtype=torch.float64,
                )
                dist.broadcast(val_t, src=0)
                avg_val_loss, avg_val_metric = val_t[0].item(), val_t[1].item()
                dist.barrier()

            current_lr = scheduler.get_last_lr()[0]
            cuda_memory_stats: dict[str, float | None]
            if device.type == "cuda":
                cuda_memory_stats = _get_cuda_memory_stats(device)
            else:
                cuda_memory_stats = {
                    "cuda_mem_allocated_gib": None,
                    "cuda_mem_reserved_gib": None,
                    "cuda_peak_allocated_gib": None,
                    "cuda_peak_reserved_gib": None,
                }

            if rank == 0:
                epoch_history.append(
                    TrainingEpochRecord(
                        epoch=epoch,
                        train_loss=avg_train_loss,
                        val_loss=avg_val_loss,
                        train_hint_metric=avg_train_metric,
                        val_hint_metric=avg_val_metric,
                        lr=current_lr,
                        cuda_mem_allocated_gib=cuda_memory_stats["cuda_mem_allocated_gib"],
                        cuda_mem_reserved_gib=cuda_memory_stats["cuda_mem_reserved_gib"],
                        cuda_peak_allocated_gib=cuda_memory_stats["cuda_peak_allocated_gib"],
                        cuda_peak_reserved_gib=cuda_memory_stats["cuda_peak_reserved_gib"],
                    )
                )

            if avg_val_metric < best_val_metric:
                best_val_metric = avg_val_metric
                if rank == 0:
                    _save_training_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_val_metric=best_val_metric,
                        args=args,
                    )
                    _save_training_checkpoint(
                        run_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_val_metric=best_val_metric,
                        args=args,
                    )
            if rank == 0:
                _save_training_checkpoint(
                    latest_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_val_metric=best_val_metric,
                    args=args,
                )
            # Rank 0 alone runs checkpoint I/O above; other ranks must not enter the next
            # epoch's training (DDP all-reduces) until rank 0 finishes, or NCCL will hang.
            if distributed:
                dist.barrier()

            if rank == 0:
                print(
                    f"Epoch {epoch:03d}/{training.epochs} | "
                    f"train_rl2_scaled={avg_train_loss:.6f} | "
                    f"train_hint_unscaled={avg_train_metric:.6f} | "
                    f"val_rl2_scaled={avg_val_loss:.6f} | "
                    f"val_hint_unscaled={avg_val_metric:.6f} | "
                    f"lr={current_lr:.6e}"
                )
                if device.type == "cuda":
                    print(f"Epoch {epoch:03d} CUDA memory | {_format_cuda_memory_stats(device)}")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train_rl2_scaled": avg_train_loss,
                            "train_hint_unscaled": avg_train_metric,
                            "val_rl2_scaled": avg_val_loss,
                            "val_hint_unscaled": avg_val_metric,
                            "lr": current_lr,
                            "best_val_hint_unscaled": best_val_metric,
                            **{
                                key: value
                                for key, value in cuda_memory_stats.items()
                                if value is not None
                            },
                        },
                        step=epoch,
                    )

        if rank == 0:
            print(f"Best val hint metric (unscaled): {best_val_metric:.6f}")
            print(f"Saved best checkpoint to: {checkpoint_path}")
            print(f"Saved run checkpoint to: {run_checkpoint_path}")
            print(f"Saved latest training-state checkpoint to: {latest_checkpoint_path}")
            print(f"Saved run config snapshot to: {run_config_path}")

            run_artifacts = persist_training_artifacts(
                epoch_history,
                output_dir=run_dir,
                run_name="run",
            )
            print(f"Saved run history CSV to: {run_artifacts.csv_path}")
            print(f"Saved run history JSON to: {run_artifacts.json_path}")
            if run_artifacts.plot_path is not None:
                print(f"Saved run loss plot to: {run_artifacts.plot_path}")
            else:
                print(f"Run loss plot was skipped: {run_artifacts.plot_error}")

            best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            _load_model_state_from_checkpoint(model, best_checkpoint)
            if test_files:
                test_loss, test_metric, _ = run_full_test_inference(
                    _eval_module(model),
                    test_files,
                    loss_fn,
                    device,
                    num_points=NUM_POS,
                    batch_size=test_batch_size,
                    num_workers=args.num_workers,
                    seed=args.seed,
                    scaler_eps=args.scaler_eps,
                    no_slip_atol=no_slip_atol,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers,
                    use_amp=use_amp,
                    assert_no_slip=assert_no_slip_test,
                    use_sobolev=use_sobolev,
                    wall_distance_loss_alpha=wall_distance_loss_alpha,
                    precompute_knn=precompute_knn,
                    knn_cache_dir=knn_cache_dir,
                    knn_k=args.k,
                )
                print(
                    "Full test inference complete | "
                    f"test_rl2_scaled={test_loss:.6f} | "
                    f"test_hint_unscaled={test_metric:.6f}"
                )
            else:
                print(
                    "Skipping full test inference (no test split). "
                    "Typical for --cv-fold k-fold validation; rely on val metrics."
                )
            if wandb_run is not None:
                if test_files:
                    wandb_run.log(
                        {
                            "test_rl2_scaled": test_loss,
                            "test_hint_unscaled": test_metric,
                        },
                        step=training.epochs + 1,
                    )
                _wandb_log_artifact(
                    wandb_run,
                    run_config_path,
                    artifact_name=f"{history_run_name}-config",
                    artifact_type="config",
                )
                _wandb_log_artifact(
                    wandb_run,
                    run_artifacts.csv_path,
                    artifact_name=f"{history_run_name}-history-csv",
                    artifact_type="history",
                )
                _wandb_log_artifact(
                    wandb_run,
                    run_artifacts.json_path,
                    artifact_name=f"{history_run_name}-history-json",
                    artifact_type="history",
                )
                if run_artifacts.plot_path is not None:
                    _wandb_log_artifact(
                        wandb_run,
                        run_artifacts.plot_path,
                        artifact_name=f"{history_run_name}-loss-plot",
                        artifact_type="plot",
                    )
                if checkpoint_path.exists():
                    _wandb_log_artifact(
                        wandb_run,
                        checkpoint_path,
                        artifact_name=f"{history_run_name}-best-checkpoint",
                        artifact_type="checkpoint",
                    )
                if run_checkpoint_path.exists() and run_checkpoint_path != checkpoint_path:
                    _wandb_log_artifact(
                        wandb_run,
                        run_checkpoint_path,
                        artifact_name=f"{history_run_name}-run-best-checkpoint",
                        artifact_type="checkpoint",
                    )
                if latest_checkpoint_path.exists():
                    _wandb_log_artifact(
                        wandb_run,
                        latest_checkpoint_path,
                        artifact_name=f"{history_run_name}-latest-checkpoint",
                        artifact_type="checkpoint",
                    )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_defaults = read_config_defaults_from_cli(argv)

    parser = argparse.ArgumentParser(
        description="Train a point-cloud CFD model on warped-ifw."
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
        help="Model family to train (this branch: SpatioTemporalGNN only).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Batch size used for final full test inference.",
    )
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
        help="Keep DataLoader workers alive across epochs (requires --num-workers > 0).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=NUM_POS,
        help="Number of spatial points used per sample (<= 100000).",
    )
    parser.add_argument(
        "--random-subsample",
        action="store_true",
        help="Use random dense nearest-neighbor cropping when --num-points < 100000.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Validation fraction of groups (default: 0.05 for a 90/5/5 split).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.05,
        help="Test fraction of groups (default: 0.05 for a 90/5/5 split). Ignored if --cv-fold is set.",
    )
    parser.add_argument(
        "--cv-fold",
        type=int,
        default=None,
        help=(
            "Optional group-level k-fold index in [0, --num-cv-folds). When set, overrides "
            "--val-fraction / --test-fraction: validation is that fold, training is all "
            "others (round-robin assignment after shuffling groups with --seed). There is "
            "no held-out test split; final 'test' inference is skipped. Train 8 folds with "
            "eight jobs: --cv-fold 0 .. --cv-fold 7 (each job can still use 8-GPU DDP)."
        ),
    )
    parser.add_argument(
        "--num-cv-folds",
        type=int,
        default=8,
        help="Number of k-folds when --cv-fold is set (default 8).",
    )
    parser.add_argument(
        "--overfit-single",
        action="store_true",
        help="Train and validate on one single datapoint for memorization checks.",
    )
    parser.add_argument(
        "--overfit-file",
        type=str,
        default="",
        help=(
            "File name, stem, or path of the datapoint used in overfit mode. "
            "When set, overfit mode is enabled automatically."
        ),
    )
    parser.add_argument(
        "--overfit-index",
        type=int,
        default=0,
        help="Dataset index used for overfit mode when --overfit-file is not provided.",
    )
    parser.add_argument(
        "--overfit-repeats",
        type=int,
        default=1,
        help="How many times to repeat the same datapoint in each training epoch.",
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
            "SpatioTemporalGNN only: use SO(3)-equivariant k-NN message passing for "
            "vector channels (velocity lift + parallel/perp edge mixing) instead of MeshGraphNet."
        ),
    )
    parser.add_argument(
        "--use-steerable-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "SpatioTemporalGNN only: e3nn steerable irreps + spherical-harmonic tensor products "
            "on the k-NN graph (requires e3nn). Mutually exclusive with --use-so3-backbone."
        ),
    )
    parser.add_argument(
        "--steerable-lmax",
        type=int,
        default=2,
        help="SpatioTemporalGNN + steerable: max spherical-harmonic order on edges (e.g. 2).",
    )
    parser.add_argument(
        "--steerable-temporal-grad-weight",
        type=float,
        default=0.0,
        help=(
            "SpatioTemporalGNN + steerable (train only): auxiliary MSE between consecutive "
            "predicted vs target velocity frames (emphasizes transient / high-frequency temporal structure)."
        ),
    )
    parser.add_argument(
        "--use-pressure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "SpatioTemporalGNN: concatenate standardized input-frame pressure (t_in scalars per node) "
            "into the node encoder. When the .npz has no pressure fields, uses zeros."
        ),
    )
    parser.add_argument(
        "--use-relative-velocity-inputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "SpatioTemporalGNN: encode v_t - v_last (after no-slip zeroing) instead of absolute v_t, "
            "matching residual prediction around the last observed frame."
        ),
    )
    parser.add_argument(
        "--train-loss-on-velocity-delta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Training only: minimize RL2/Sobolev on (prediction - v_last) vs (target - v_last) in scaled space. "
            "Validation and test still use full-field RL2 for leaderboard-aligned metrics."
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
        "--precompute-knn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Precompute/load fixed full-sample kNN lookup tables before training.",
    )
    parser.add_argument(
        "--knn-cache-dir",
        type=str,
        default="",
        help="Directory used for precomputed kNN lookup tables.",
    )
    parser.add_argument(
        "--knn-include-neighbors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When subsampling with precomputed kNN, include one-hop cached "
            "neighbors outside the crop so local graph context can exceed --num-points."
        ),
    )
    parser.add_argument(
        "--knn-overflow-points",
        type=int,
        default=0,
        help=(
            "Fixed number of outside-crop neighbor points to append when "
            "--knn-include-neighbors is enabled. Use >0 for batch_size > 1."
        ),
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Peak learning rate for AdamW / OneCycleLR.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="rl2",
        choices=["rl2", "sobolev"],
        help="Primary loss function: 'rl2' (Relative L2) or 'sobolev' (RL2 + spatial gradient penalty).",
    )
    parser.add_argument(
        "--sobolev-grad-weight",
        type=float,
        default=0.1,
        help="Spatial-gradient penalty weight when --loss-fn=sobolev.",
    )
    parser.add_argument(
        "--wall-distance-loss-alpha",
        type=float,
        default=0.0,
        help=(
            "Exponent for spatially weighted loss: w_i = 1/(d_i + eps)^alpha. "
            "0.0 disables wall-distance weighting. Typical values: 0.5-1.0."
        ),
    )
    parser.add_argument(
        "--divergence-loss-lambda",
        type=float,
        default=0.0,
        help=(
            "Optional (train only) k-NN flux-divergence penalty for SpatioTemporalGNN, "
            "on **unscaled** predicted velocity. Start small (1e-4..1e-2); "
            "validate the discrete div on smooth fields first."
        ),
    )
    parser.add_argument(
        "--augment-xy-plane-isometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Random x-reflection and z-axis rotation of pos + in/out velocity (vector). "
            "Train dataset only. Applied before per-sample standardization."
        ),
    )
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA automatic mixed precision for faster training on T4.",
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
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the model with torch.compile for faster training.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable activation checkpointing for spatiotemporal_mno spatial blocks "
            "to reduce peak VRAM at the cost of extra compute."
        ),
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable cuDNN benchmark mode for fixed-shape training throughput.",
    )
    parser.add_argument(
        "--scaler-eps",
        type=float,
        default=1e-6,
        help="Lower bound for velocity standard-deviation during scaling.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Number of micro-batches to accumulate before each optimizer step. "
            "Effective batch size = batch_size * gradient_accumulation_steps."
        ),
    )
    parser.add_argument(
        "--no-slip-atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for optional no-slip boundary assertions.",
    )
    parser.add_argument(
        "--assert-no-slip-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable per-batch no-slip boundary assertions during training.",
    )
    parser.add_argument(
        "--assert-no-slip-val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable no-slip assertions during validation.",
    )
    parser.add_argument(
        "--assert-no-slip-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable no-slip assertions during full test inference.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=("auto", "one-cycle", "fixed"),
        default="auto",
        help=(
            "Learning-rate schedule. 'auto' preserves the existing behavior: "
            "OneCycle for normal training and fixed LR for overfit mode. "
            "Set 'fixed' to disable LR scheduling entirely."
        ),
    )
    parser.add_argument(
        "--boundary-point-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of subsampled points reserved for the airfoil region. "
            "0.0 disables boundary-focused sampling. When enabled, the reserved budget "
            "is split between sparse true-surface anchors and an off-wall shell around "
            "the airfoil so the batch captures boundary-layer structure (training only)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--load-bundled-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "SpatioTemporalGNN: load models/fmgreco_stnn/state_dict.pt when present. "
            "Use --no-load-bundled-weights when the bundle does not match the model "
            "(e.g. different latent_dim than the checkpoint)."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Optional checkpoint path to warm-start model weights before training.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="models/fmgreco_stnn/state_dict.pt",
    )
    parser.add_argument(
        "--run-name",
        "--history-run-name",
        dest="history_run_name",
        type=str,
        default="",
        help="Optional run folder name under --runs-dir.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="outputs/runs",
        help="Directory containing per-run folders with checkpoints and artifacts.",
    )
    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable optional Weights & Biases experiment logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="iclr_develop",
        help="W&B project name used when --use-wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="",
        help="Optional W&B run name override. Defaults to --run-name.",
    )
    parser.add_argument(
        "--wandb-dir",
        type=str,
        default="",
        help="Directory for W&B local files. Defaults to the run directory.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
        help="W&B mode. Use offline on clusters without internet access.",
    )

    apply_parser_defaults_from_config(
        parser,
        config_defaults,
        strict_unknown_keys=True,
    )

    return parser.parse_args(argv)
