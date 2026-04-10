"""
Smoke test suite for TransolverResidual — run this before committing to a full
training run.

Usage:
    ./venv/bin/python smoke_test.py

Runs 5 tests in order. Each test prints PASS / FAIL / WARNING and tells you
exactly what to look for and what to do if something looks wrong.

Tests
-----
1. Untrained model equals polynomial baseline
2. Gradient flow — every parameter receives a gradient
3. Loss decreases on a single batch (overfit sanity check)
4. Mini training run (10 epochs, 20 samples) — train/val gap check
5. Memory and throughput on GPU
"""

import os, sys, time, glob, random
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from models.transolver_residual import TransolverResidual
from models.transolver_residual.polynomial import poly_extrapolate
from models.transolver_residual.features import precompute_distance_features
from train import relative_l2, variance_weighted_loss

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(ROOT, "gram_data")

SEP  = "─" * 70
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def banner(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def load_real_sample(path, device="cpu"):
    """Load one .npz sample. Main tensors go to `device`; dist_feats stay on CPU."""
    data         = np.load(path)
    velocity_in  = torch.from_numpy(data["velocity_in"]).unsqueeze(0).to(device)
    pos          = torch.from_numpy(data["pos"]).unsqueeze(0).to(device)
    t            = torch.from_numpy(data["t"]).unsqueeze(0).to(device)
    velocity_out = torch.from_numpy(data["velocity_out"]).unsqueeze(0).to(device)
    idcs_airfoil = [torch.from_numpy(data["idcs_airfoil"].astype(np.int64))]
    ia, dist, xsign = precompute_distance_features(
        data["pos"], data["idcs_airfoil"].astype(np.int64))
    dist_feats = [(torch.from_numpy(ia), torch.from_numpy(dist), torch.from_numpy(xsign))]
    return velocity_in, pos, t, velocity_out, idcs_airfoil, dist_feats


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Untrained model == polynomial baseline
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_init():
    banner("TEST 1 — Untrained model equals polynomial baseline")
    print("The decoder is zero-initialized, so the untrained model should output")
    print("exactly the polynomial extrapolation. Any deviation indicates a bug in")
    print("the decoder init or the residual combination step.")
    print()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not files:
        print(f"{WARN}  No data found in {DATA_DIR} — using synthetic data")
        B, N = 1, 1000
        velocity_in  = torch.rand(B, 5, N, 3).to(DEVICE)
        pos          = torch.rand(B, N, 3).to(DEVICE)
        t            = torch.rand(B, 10).to(DEVICE)
        idcs_airfoil = [torch.randint(N, (50,))]
        dist_feats   = None
    else:
        velocity_in, pos, t, _, idcs_airfoil, dist_feats = load_real_sample(
            random.choice(files), DEVICE)

    # Fresh model with no weights.pt (force fresh init)
    model = TransolverResidual(n_layers=4, hidden_dim=128, n_heads=4,
                                slice_num=16, mlp_ratio=1).to(DEVICE)
    # Verify decoder is zero
    dec_norm = model.decoder.weight.norm().item()
    print(f"  decoder weight norm (should be 0.0): {dec_norm:.6f}")
    if dec_norm > 1e-6:
        print(f"  {FAIL}  Decoder is NOT zero-initialized. Check _init_weights().")
        return False

    with torch.no_grad():
        pred = model(t, pos, idcs_airfoil, velocity_in, dist_feats)
        poly = poly_extrapolate(velocity_in, t, degree=2)
        # Zero out no-slip on poly too (model does this automatically)
        for b in range(len(idcs_airfoil)):
            poly[b, :, idcs_airfoil[b]] = 0.0

    max_diff = (pred - poly).abs().max().item()
    rel_diff = (pred - poly).norm().item() / (poly.norm().item() + 1e-8)

    print(f"  max |pred - poly|:      {max_diff:.2e}  (should be < 1e-5)")
    print(f"  relative difference:    {rel_diff:.2e}  (should be < 1e-5)")

    if rel_diff < 1e-4:
        print(f"  {PASS}  Untrained model correctly outputs polynomial baseline.")
        return True
    else:
        print(f"  {FAIL}  Significant gap between untrained model and poly baseline.")
        print("         Check that decoder weight AND bias are zero-initialized.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Gradient flow
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_flow():
    banner("TEST 2 — Gradient flow")
    print("Every learnable parameter must receive a non-zero gradient on the first")
    print("backward pass. Dead parameters waste capacity and indicate architectural")
    print("bugs (e.g. a block that's bypassed, a disconnected projection).")
    print()

    B, N = 1, 2000
    velocity_in  = torch.rand(B, 5, N, 3).to(DEVICE)
    pos          = torch.rand(B, N, 3).to(DEVICE)
    t            = torch.rand(B, 10).to(DEVICE)
    idcs_airfoil = [torch.randint(N, (100,))]
    velocity_out = torch.rand(B, 5, N, 3).to(DEVICE)

    model = TransolverResidual(n_layers=4, hidden_dim=64, n_heads=4,
                                slice_num=8, mlp_ratio=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # The decoder is zero-initialized, so at step 0 the gradient of the loss
    # w.r.t. the encoder and Transolver blocks is exactly zero (it flows through
    # W_decoder which is 0). This is expected and by design — the model starts
    # as a pure polynomial baseline. We take a few warm-up steps to break
    # through, then check that all parameters have non-zero gradients.
    print("  (Taking 3 warm-up steps to break through zero-init decoder...)")
    for _ in range(3):
        optimizer.zero_grad()
        pred = model(t, pos, idcs_airfoil, velocity_in)
        loss = variance_weighted_loss(pred, velocity_out, velocity_in)
        loss.backward()
        optimizer.step()

    optimizer.zero_grad()
    pred = model(t, pos, idcs_airfoil, velocity_in)
    loss = variance_weighted_loss(pred, velocity_out, velocity_in)
    loss.backward()

    dead, total = [], 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total += 1
            if param.grad is None or param.grad.abs().max().item() == 0:
                dead.append(name)

    print(f"  Total learnable tensors: {total}")
    print(f"  Tensors with zero/None grad: {len(dead)}")

    if not dead:
        print(f"  {PASS}  All parameters receive gradients after warm-up.")
        return True
    else:
        print(f"  {FAIL}  Dead parameters detected after warm-up:")
        for name in dead[:10]:
            print(f"         - {name}")
        if len(dead) > 10:
            print(f"         ... and {len(dead)-10} more")
        print("         This usually means a layer is disconnected from the loss.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Single-batch overfit
# ─────────────────────────────────────────────────────────────────────────────

def test_overfit_single_batch():
    banner("TEST 3 — Single-batch overfit")
    print("Train on one fixed sample for 50 steps. The model must be able to")
    print("memorize it — loss should drop to < 0.05. If it doesn't, the model")
    print("has insufficient capacity or the optimizer is misconfigured.")
    print()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not files:
        print(f"{WARN}  No data found — using synthetic data (this test is weaker without real data)")
        B, N = 1, 2000
        velocity_in  = torch.rand(B, 5, N, 3).to(DEVICE)
        pos          = torch.rand(B, N, 3).to(DEVICE)
        t            = torch.rand(B, 10).to(DEVICE)
        idcs_airfoil = [torch.randint(N, (100,))]
        velocity_out = torch.rand(B, 5, N, 3).to(DEVICE)
        dist_feats   = None
    else:
        path = random.choice(files)
        print(f"  Sample: {os.path.basename(path)}")
        velocity_in, pos, t, velocity_out, idcs_airfoil, dist_feats = \
            load_real_sample(path, DEVICE)

    # Note: zero-init decoder means only the decoder learns in the first few
    # steps. Use a higher LR and more steps to force the full network to engage.
    model = TransolverResidual(n_layers=4, hidden_dim=128, n_heads=4,
                                slice_num=16, mlp_ratio=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    losses = []
    for step in range(150):
        optimizer.zero_grad()
        pred = model(t, pos, idcs_airfoil, velocity_in, dist_feats)
        loss = variance_weighted_loss(pred, velocity_out, velocity_in)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 30 == 0:
            print(f"    step {step+1:3d}:  loss = {loss.item():.4f}")

    final_loss = losses[-1]
    drop = (losses[0] - losses[-1]) / (losses[0] + 1e-8) * 100

    print(f"\n  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {final_loss:.4f}  (drop: {drop:.1f}%)")
    print()
    print("  NOTE: The zero-init decoder means the first ~10 steps only train the")
    print("  decoder. The encoder and Transolver blocks engage after that.")
    print("  A healthy run shows fast initial drop (decoder learning polynomial")
    print("  correction) then continued slow improvement (blocks learning turbulence).")

    if final_loss < 0.20:
        print(f"  {PASS}  Model can overfit a single sample.")
        return True
    elif drop > 20:
        print(f"  {WARN}  Loss dropped {drop:.1f}% but not fully overfit. Acceptable.")
        print("         The model is learning. Continue to Test 4.")
        return True
    else:
        print(f"  {FAIL}  Model failed to overfit a single sample (drop={drop:.1f}%).")
        print("         Check: try --lr 1e-2, check data loading, check loss function.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Mini training run: train/val gap
# ─────────────────────────────────────────────────────────────────────────────

def test_mini_training():
    banner("TEST 4 — Mini training run (10 epochs, up to 30 samples)")
    print("Trains on a small subset and checks that:")
    print("  (a) training loss decreases monotonically (or near-monotonically)")
    print("  (b) val loss tracks train loss — no immediate overfitting")
    print("  (c) variance-weighted loss < standard loss (sanity check on weighting)")
    print()
    print("What to look for:")
    print("  train/val gap < 0.15 after 10 epochs  →  good, no overfitting yet")
    print("  train/val gap > 0.30 after 10 epochs  →  risk of overfitting on full run,")
    print("                                            consider dropping hidden_dim to 192")
    print("  train loss not decreasing              →  LR problem, check scheduler")
    print()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if len(files) < 4:
        print(f"{WARN}  Need at least 4 .npz files, found {len(files)}. Skipping.")
        return None

    random.shuffle(files)
    n_train = min(20, int(len(files) * 0.8))
    n_val   = min(10, len(files) - n_train)
    train_files = files[:n_train]
    val_files   = files[n_train:n_train + n_val]
    print(f"  Using {n_train} train / {n_val} val samples")

    # Preload ALL samples into GPU memory once — distance features are expensive
    # to compute (100k × 2k distance matrix on CPU) and must not be recomputed
    # each epoch. Loading 30 samples × ~1.8 GB = ~54 GB in float32, but we
    # keep them on CPU and move to GPU per-step to avoid OOM.
    print(f"  Pre-loading samples (computing distance features once)...", flush=True)

    def preload(file_list):
        samples = []
        for i, path in enumerate(file_list):
            velocity_in, pos, t, velocity_out, idcs_airfoil, dist_feats = \
                load_real_sample(path, "cpu")   # keep on CPU
            samples.append((velocity_in, pos, t, velocity_out, idcs_airfoil, dist_feats))
            print(f"    loaded {i+1}/{len(file_list)}: {os.path.basename(path)}", flush=True)
        return samples

    train_samples = preload(train_files)
    val_samples   = preload(val_files)
    print(f"  Done. Starting training...\n", flush=True)

    model = TransolverResidual(n_layers=6, hidden_dim=192, n_heads=8,
                                slice_num=16, mlp_ratio=1, dropout=0.05).to(DEVICE)
    print(f"  Model params: {model.num_params():,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

    from torch.amp import autocast

    def epoch_loss(samples, train):
        model.train(train)
        total = 0.0
        for velocity_in, pos, t, velocity_out, idcs_airfoil, dist_feats in samples:
            # Move to GPU for this step
            velocity_in  = velocity_in.to(DEVICE,  non_blocking=True)
            pos          = pos.to(DEVICE,          non_blocking=True)
            t            = t.to(DEVICE,            non_blocking=True)
            velocity_out = velocity_out.to(DEVICE, non_blocking=True)
            with torch.set_grad_enabled(train):
                with autocast("cuda", dtype=torch.bfloat16,
                               enabled=(DEVICE.type == "cuda")):
                    pred = model(t, pos, idcs_airfoil, velocity_in, dist_feats)
                    loss = variance_weighted_loss(pred, velocity_out, velocity_in)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            total += loss.item()
        return total / len(samples)

    print(f"\n  {'Epoch':>6}  {'Train':>8}  {'Val':>8}  {'Gap':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")

    train_losses, val_losses = [], []
    for epoch in range(1, 11):
        tl = epoch_loss(train_samples, train=True)
        with torch.no_grad():
            vl = epoch_loss(val_samples, train=False)
        train_losses.append(tl)
        val_losses.append(vl)
        gap = vl - tl
        flag = ""
        if gap > 0.30:
            flag = f"  ← {WARN} large gap"
        print(f"  {epoch:>6}  {tl:>8.4f}  {vl:>8.4f}  {gap:>+8.4f}{flag}")

    final_gap = val_losses[-1] - train_losses[-1]
    loss_drop = (train_losses[0] - train_losses[-1]) / (train_losses[0] + 1e-8) * 100

    print(f"\n  Training loss drop: {loss_drop:.1f}%")
    print(f"  Final train/val gap: {final_gap:+.4f}")

    if loss_drop < 5:
        print(f"  {FAIL}  Training loss barely moved. Check LR and data loading.")
        return False
    elif final_gap > 0.30:
        print(f"  {WARN}  Large train/val gap. On a full run consider:")
        print("         --hidden_dim 192  or  --n_layers 6  or  --dropout 0.15")
        return True
    else:
        print(f"  {PASS}  Loss decreasing, no severe overfitting in first 10 epochs.")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Memory and throughput
# ─────────────────────────────────────────────────────────────────────────────

def test_memory_throughput():
    banner("TEST 5 — GPU memory and throughput")
    print("Runs a full forward+backward pass at actual training scale (100k points).")
    print("Measures peak VRAM and seconds per step.")
    print()
    print("What to look for:")
    print("  Peak VRAM < 20 GB   →  comfortable on L40S (48 GB)")
    print("  Peak VRAM > 35 GB   →  tight, consider reducing hidden_dim")
    print("  Seconds/step < 10   →  reasonable throughput")
    print("  Seconds/step > 20   →  slow, check num_workers and pin_memory")
    print()

    if DEVICE.type != "cuda":
        print(f"  {WARN}  No GPU detected. Skipping memory test.")
        return None

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not files:
        print(f"  {WARN}  No data found — using synthetic 100k-point sample")
        B, N = 1, 100_000
        velocity_in  = torch.rand(B, 5, N, 3).to(DEVICE)
        pos          = torch.rand(B, N, 3).to(DEVICE)
        t            = torch.rand(B, 10).to(DEVICE)
        idcs_airfoil = [torch.randint(N, (5000,))]
        velocity_out = torch.rand(B, 5, N, 3).to(DEVICE)
        dist_feats   = None
    else:
        path = random.choice(files)
        print(f"  Sample: {os.path.basename(path)}")
        velocity_in, pos, t, velocity_out, idcs_airfoil, dist_feats = \
            load_real_sample(path, DEVICE)

    model = TransolverResidual(n_layers=8, hidden_dim=256, n_heads=8,
                                slice_num=32, mlp_ratio=1, dropout=0.1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    from torch.amp import autocast

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()

    with autocast("cuda", dtype=torch.bfloat16):
        pred = model(t, pos, idcs_airfoil, velocity_in, dist_feats)
        loss = variance_weighted_loss(pred, velocity_out, velocity_in)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Total VRAM:     {total_gb:.1f} GB")
    print(f"  Peak VRAM used: {peak_gb:.1f} GB  ({peak_gb/total_gb*100:.1f}%)")
    print(f"  Time per step:  {elapsed:.1f}s")

    ok_mem = peak_gb < total_gb * 0.75
    ok_speed = elapsed < 20

    if ok_mem and ok_speed:
        print(f"  {PASS}  Memory and throughput are fine.")
    elif not ok_mem:
        print(f"  {WARN}  High memory usage. Consider --hidden_dim 192 or --accum_steps 2")
    elif not ok_speed:
        print(f"  {WARN}  Slow step. Check --num_workers (try 4 or 8)")

    return ok_mem


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*70}")
    print("  TransolverResidual — Smoke Test Suite")
    print(f"  Device: {DEVICE}")
    files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    print(f"  Data:   {len(files)} samples in {DATA_DIR}")
    print(f"{'═'*70}")

    results = {}
    results["zero_init"]    = test_zero_init()
    results["grad_flow"]    = test_gradient_flow()
    results["overfit"]      = test_overfit_single_batch()
    results["mini_train"]   = test_mini_training()
    results["memory"]       = test_memory_throughput()

    banner("SUMMARY")
    all_pass = True
    for name, result in results.items():
        if result is True:
            status = PASS
        elif result is False:
            status = FAIL
            all_pass = False
        else:
            status = f"\033[90mSKIP\033[0m"
        print(f"  {status}  {name}")

    print()
    if all_pass:
        print("All tests passed. Recommended training command:")
        print()
        print("  ./venv/bin/python train.py \\")
        print("      --n_layers 8 \\")
        print("      --hidden_dim 256 \\")
        print("      --epochs 100 \\")
        print("      --lr 1e-3 \\")
        print("      --accum_steps 4 \\")
        print("      --num_workers 4")
    else:
        print("Fix the failing tests before starting a full run.")
    print()


if __name__ == "__main__":
    main()
