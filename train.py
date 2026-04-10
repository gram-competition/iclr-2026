"""
Training script for TransolverResidual on the GRaM dataset.

Usage:
    ./venv/bin/python train.py                          # defaults
    ./venv/bin/python train.py --epochs 100 --lr 3e-4
    ./venv/bin/python train.py --hidden_dim 256 --n_layers 8

Checkpoints are saved to runs/<run_name>/weights.pt.
The best checkpoint is also copied to models/transolver_residual/weights.pt
so that main.py always loads the best trained model.

TensorBoard logs are written to runs/<run_name>/tb/.
Launch with:
    ./venv/bin/tensorboard --logdir runs/
"""

import argparse
import os
import shutil
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from models.transolver_residual import TransolverResidual
from utils.dataloader import make_loaders


# ── Loss ─────────────────────────────────────────────────────────────────────

def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Relative L2 loss, averaged over the batch.

    pred, target: (B, 5, N, 3)
    Returns a scalar.
    """
    diff_norm = (pred - target).reshape(pred.shape[0], -1).norm(dim=-1)
    tgt_norm  = target.reshape(target.shape[0], -1).norm(dim=-1)
    return (diff_norm / (tgt_norm + 1e-8)).mean()


def variance_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    velocity_in: torch.Tensor,
    wake_weight: float = 4.0,
) -> torch.Tensor:
    """
    Relative L2 loss weighted by per-point temporal variance from the input
    window. Wake and boundary-layer points (high variance) contribute more
    to the loss; freestream points (low variance) contribute less.

    Rationale: the evaluators' metric is a global average, but submissions
    will differ almost entirely in the wake region. Uniform loss rewards the
    model for nailing the trivially-easy freestream and doesn't penalise
    enough for getting the wake wrong. Variance weighting re-focuses capacity
    on the hard points.

    Args:
        pred, target  : (B, 5, N, 3)
        velocity_in   : (B, 5, N, 3)  — used to compute per-point variance
        wake_weight   : maximum weight multiplier for the highest-variance points
                        (default 4.0 — wake points count 4× more than freestream)

    Returns:
        scalar loss
    """
    B, T_out, N, C = pred.shape

    # Per-point temporal std over the input window → (B, N)
    # std over dim=1 (timesteps), then mean over components
    pt_std = velocity_in.std(dim=1).mean(dim=-1)              # (B, N)

    # Normalise to [1, wake_weight] per sample so the total weight sum is
    # stable regardless of the absolute turbulence level
    pt_min = pt_std.min(dim=1, keepdim=True).values
    pt_max = pt_std.max(dim=1, keepdim=True).values
    weights = 1.0 + (wake_weight - 1.0) * (pt_std - pt_min) / (pt_max - pt_min + 1e-8)
    # weights: (B, N),  range [1, wake_weight]

    # Weighted pointwise squared error, summed over T and C
    # error: (B, T_out, N, C) → squared → sum over T,C → (B, N)
    sq_err     = (pred - target).pow(2).sum(dim=(1, 3))       # (B, N)
    sq_tgt     = target.pow(2).sum(dim=(1, 3))                # (B, N)

    # Weighted relative L2 per sample
    w_err = (weights * sq_err).sum(dim=1)                     # (B,)
    w_tgt = (weights * sq_tgt).sum(dim=1)                     # (B,)

    return (w_err / (w_tgt + 1e-8)).sqrt().mean()


@torch.no_grad()
def per_timestep_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> list:
    """
    Returns a list of 5 scalar relative L2 values, one per output timestep.
    pred, target: (B, 5, N, 3)
    """
    losses = []
    for i in range(pred.shape[1]):
        p = pred[:, i]    # (B, N, 3)
        g = target[:, i]
        diff = (p - g).reshape(p.shape[0], -1).norm(dim=-1)
        gt   = g.reshape(g.shape[0], -1).norm(dim=-1)
        losses.append((diff / (gt + 1e-8)).mean().item())
    return losses


# ── Training / validation passes ─────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, accum_steps, is_train):
    model.train(is_train)
    total_loss = 0.0
    n_batches  = 0
    use_bf16   = device.type == "cuda"

    if is_train:
        optimizer.zero_grad()

    for step, ((velocity_in, pos, idcs_airfoil, t, dist_feats), velocity_out) in enumerate(loader):
        velocity_in  = velocity_in.to(device,  non_blocking=True)
        pos          = pos.to(device,          non_blocking=True)
        t            = t.to(device,            non_blocking=True)
        velocity_out = velocity_out.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                pred = model(t, pos, idcs_airfoil, velocity_in, dist_feats)
                loss = variance_weighted_loss(pred, velocity_out, velocity_in)
                if is_train:
                    loss = loss / accum_steps

        if is_train:
            loss.backward()
            if (step + 1) % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += (loss.item() * accum_steps if is_train else loss.item())
        n_batches  += 1

    # flush any remaining accumulated gradients
    if is_train and (len(loader) % accum_steps != 0):
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def val_detailed(model, loader, device):
    """Returns mean val loss and per-timestep losses."""
    model.eval()
    total_loss = 0.0
    per_ts     = [0.0] * 5
    n_batches  = 0

    for (velocity_in, pos, idcs_airfoil, t, dist_feats), velocity_out in loader:
        velocity_in  = velocity_in.to(device,  non_blocking=True)
        pos          = pos.to(device,          non_blocking=True)
        t            = t.to(device,            non_blocking=True)
        velocity_out = velocity_out.to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            pred = model(t, pos, idcs_airfoil, velocity_in, dist_feats)

        total_loss += relative_l2(pred, velocity_out).item()
        ts_losses   = per_timestep_rel_l2(pred, velocity_out)
        for i, v in enumerate(ts_losses):
            per_ts[i] += v
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, [v / n for v in per_ts]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TransolverResidual on GRaM")

    # Data
    parser.add_argument("--data_dir",        default="gram_data")
    parser.add_argument("--train_fraction",  type=float, default=0.9)
    parser.add_argument("--num_workers",     type=int,   default=4)

    # Model
    parser.add_argument("--n_layers",    type=int,   default=8)
    parser.add_argument("--hidden_dim",  type=int,   default=256)
    parser.add_argument("--n_heads",     type=int,   default=8)
    parser.add_argument("--slice_num",   type=int,   default=32)
    parser.add_argument("--mlp_ratio",   type=int,   default=1)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--poly_degree", type=int,   default=2)

    # Training
    parser.add_argument("--epochs",       type=int,   default=400)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--accum_steps",  type=int,   default=4,
                        help="Gradient accumulation steps (effective batch = accum_steps)")
    parser.add_argument("--seed",         type=int,   default=42)

    # Logging / saving
    parser.add_argument("--run_name", default=None,
                        help="Run name for runs/ directory. Defaults to timestamp.")
    parser.add_argument("--val_every",  type=int, default=5,
                        help="Run full val (with per-timestep breakdown) every N epochs")
    parser.add_argument("--resume", action="store_true",
                        help="Load weights from models/transolver_residual/weights.pt before training."
                             " Use this to continue a previous run. Omit for a fresh start.")
    parser.add_argument("--augment", action="store_true",
                        help="Enable y-flip augmentation on the training set.")

    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = os.path.join("runs", run_name)
    tb_dir   = os.path.join(run_dir, "tb")
    ckpt_dir = run_dir
    os.makedirs(tb_dir,   exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(tb_dir)
    print(f"Run: {run_dir}")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_loader, val_loader = make_loaders(
        data_dir=args.data_dir,
        train_fraction=args.train_fraction,
        batch_size=1,                   # always 1 — grad accumulation handles effective batch
        num_workers=args.num_workers,
        seed=args.seed,
        augment=args.augment,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = TransolverResidual(
        n_layers     = args.n_layers,
        hidden_dim   = args.hidden_dim,
        n_heads      = args.n_heads,
        slice_num    = args.slice_num,
        mlp_ratio    = args.mlp_ratio,
        dropout      = args.dropout,
        poly_degree  = args.poly_degree,
        load_weights = args.resume,
    ).to(device)

    print(f"Parameters: {model.num_params():,}")
    writer.add_text("model/params", str(model.num_params()))
    writer.add_text("model/config", str(vars(args)))

    # ── Optimiser + scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    # BF16 on Ada Lovelace (L40S) — no loss scaling needed, same range as FP32

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_ckpt     = os.path.join(ckpt_dir, "weights_best.pt")
    weights_dst   = os.path.join("models", "transolver_residual", "weights.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = run_epoch(
            model, train_loader, optimizer, device,
            args.accum_steps, is_train=True,
        )
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("lr",         lr_now,     epoch)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:4d}/{args.epochs}  train={train_loss:.4f}  lr={lr_now:.2e}  ({elapsed:.0f}s)")

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_loss, per_ts = val_detailed(model, val_loader, device)
            writer.add_scalar("loss/val", val_loss, epoch)
            for i, v in enumerate(per_ts):
                writer.add_scalar(f"val/t{i+5}", v, epoch)

            ts_str = "  ".join(f"t{i+6}={v:.3f}" for i, v in enumerate(per_ts))
            print(f"           val={val_loss:.4f}  [{ts_str}]")

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_ckpt)
                shutil.copy(best_ckpt, weights_dst)
                print(f"           ✓ new best — saved to {weights_dst}")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if epoch % 20 == 0:
            periodic = os.path.join(ckpt_dir, f"weights_ep{epoch:04d}.pt")
            torch.save(model.state_dict(), periodic)

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Weights at: {weights_dst}")


if __name__ == "__main__":
    main()
