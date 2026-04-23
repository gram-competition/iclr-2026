# Submission #22 — Technical brief (aligned with shipped code)

Use this text to keep the **GitHub PR description** consistent with what reviewers can verify in the diff. Prefer **greppable** claims over marketing language.

## 1. Three pillars (where to look in code)

| Pillar | What to grep / read |
|--------|---------------------|
| **Hard no-slip** | `apply_hard_no_slip` in `gnn_base.py` — in-place zeroing of predicted velocity on `idcs_airfoil` for tensor shape `(T_out, N, 3)`. Inputs also zero airfoil velocity before encoding and before the residual baseline. |
| **Pressure-informed nodes** | `load_pressure_inputs_from_npz`, `pressure_in` in `src/data/dataset.py`; `use_pressure`, `pressure_in` in `SpatioTemporalGNN.forward` / `_forward_single` in `gnn_base.py`. Uses the **first `t_in` frames** of `pressure` or `pressure_in` from each `.npz`, per-sample standardized. |
| **Residual / Δ-focused training** | Decoder still predicts a **delta** added to a **baseline** (last input frame or mean). **`train_loss_on_velocity_delta`** (trainer + `train_config.yaml`) trains the primary loss on **`(pred − v_last)` vs `(target − v_last)`** in **scaled** space; val/test remain **full-field** RL2 unless you change evaluation. **`use_relative_velocity_inputs`** feeds **`v_t − v_last`** (after wall zeroing) into the encoder. |

## 2. Backbone wording (be precise)

- **Default** (`train_config.yaml`): `use_so3_backbone: false`, `use_steerable_backbone: false` → spatial stack is a **deep residual k-NN MeshGraphNet-style encoder** (`num_blocks` layers), **not** an SO(3) backbone.
- **Optional geometry priors**: enable exactly one of `use_so3_backbone` or `use_steerable_backbone` for the hand-built SO(3) path or the e3nn steerable path.

Do **not** title the PR “SO(3)-equivariant …” unless the **same** config you report numbers with has the matching flag set to `true`.

## 3. Temporal mixing / hardware

- `TemporalAttentionHead` uses `nn.TransformerEncoder` → `nn.MultiheadAttention`. On PyTorch 2.x this routes through **`torch.nn.functional.scaled_dot_product_attention`**.
- The **actual** kernel (math vs memory-efficient vs vendor-fused) depends on **dtype, head dim, sequence length, and your wheel** (CUDA or ROCm). We do **not** claim a specific trademarked FlashAttention build unless you profile and name the active backend on your cluster.
- **`t_out` is small (e.g. 5)** → attention cost per node is **O(t_out²)** and is usually **not** the training bottleneck; the **spatial** k-NN stack over ~100k points dominates.

## 4. Training defaults (`train_config.yaml`)

- **LR**: `lr: 0.0001` with `lr_scheduler: one-cycle` (do not quote `1e-3` unless you change and re-benchmark).
- **AMP**: `use_amp: true` → bf16 autocast on GPU when the stack supports it.
- **No-slip checks**: `assert_no_slip_train`, `assert_no_slip_val`, `no_slip_atol` — optional strict checks in scaled space.

## 5. Optional factual contrasts (other PRs)

Keep comparisons **descriptive**, not dismissive:

- **vs global / slice-heavy transformers**: this entry keeps a **full 3D point cloud** and **local k-NN** message passing over unstructured neighborhoods rather than compressing the volume into slices only.
- **vs pure equivariant operators**: this fork adds an explicit **competition dataset** path (pressure, hard boundary indices, residual training options) and a **short-horizon temporal** head on per-node features.

## 6. One-paragraph PR summary (copy-ready)

**Submission #22** is a **physics-informed spatiotemporal GNN** on warped 3D point clouds: a **deep k-NN spatial encoder** (optional **SO(3)** or **e3nn steerable** backbones), a **small temporal Transformer** over **`t_out = 5`** frames per node (attention via PyTorch **SDPA/MHA**), **hard no-slip** enforcement via **`apply_hard_no_slip`**, and **optional standardized pressure** over input time. Training can emphasize **transients** with **Δ-velocity (residual) loss** and **relative velocity inputs** while validation reports **full-field** metrics. Default training uses **OneCycleLR** at **1e-4** and **AMP bf16** where available.

## 7. Eligibility / awards

Workshop awards, co-authorship, and deadlines are governed only by **official** GRaM / ICLR materials — not by this brief.

## 8. Running 8× GPU training (DDP, ROCm)

**Entry point:** training is `main.py`, not `models/fmgreco_stnn/train.py`. The trainer expects `torchrun` to set `WORLD_SIZE` / `LOCAL_RANK` and uses **`--device cuda`** (PyTorch still reports `cuda` on ROCm builds).

From the **repository root**:

```bash
torchrun --standalone --nproc_per_node=8 main.py \
  --config train_config.yaml \
  --dataset-dir /path/to/warped-ifw-npz \
  --device cuda
```

**Strict YAML:** `parse_args` uses `strict_unknown_keys=True`. Keys that are **not** valid CLI `dest` names cause **`parser.error("Unknown key(s) in config file: ...")`**. Do **not** paste generic names like `hidden_dim`, `dataset_path`, `scheduler`, `precision`, `strategy`, `enforce_no_slip`, `use_pressure_channels`, or `predict_delta` unless you extend the argparse parser to define them.

| Wrong / invented | Actual key in `train_config.yaml` |
|------------------|-----------------------------------|
| `hidden_dim` | `latent_dim` (mapped to model `hidden_dim`) |
| `num_layers` | `num_blocks` (mapped to spatial depth) |
| `num_attn_layers` | `num_temporal_layers` |
| `use_pressure_channels` | `use_pressure` |
| `predict_delta` / residual loss | `train_loss_on_velocity_delta` (training loss only; model always predicts δ + baseline) |
| `enforce_no_slip` | No config flag — hard mask is always **`apply_hard_no_slip`** in forward; use **`assert_no_slip_*`** + **`no_slip_atol`** for runtime checks |
| `dataset_path` | `dataset_dir` |
| `scheduler: "OneCycleLR"` | `lr_scheduler: one-cycle` |
| `precision: "bf16"` | `use_amp: true` (bf16 autocast on GPU in this trainer) |
| `strategy: "ddp"` | *(omit — launch with `torchrun` only)* |
| `model_name: "fmgreco_stnn"` | Optional alias: `fmgreco_stnn` → `spatio_temporal_gnn` (canonical default in repo YAML is `spatio_temporal_gnn`) |

- **`--standalone`** is appropriate for a single 8-GPU node; multi-node needs a different launcher layout.
- **Effective batch size** ≈ `world_size × batch_size × gradient_accumulation_steps` (see comments in `train_config.yaml`).
- **Backend:** `dist.init_process_group(backend="nccl")` — on ROCm this maps to **RCCL**; ensure the image has a compatible stack.

**Config format:** keys are **flat** at the top level of `train_config.yaml` (there is **no** nested `model:` / `training:` block). Map your checklist to real keys:

| Intent | Actual YAML / CLI |
|--------|-------------------|
| Width / depth | `latent_dim` (→ `hidden_dim`), `num_blocks` (→ spatial layers), `num_temporal_layers` |
| Pressure | `use_pressure: true` |
| Δ-velocity **training** loss | `train_loss_on_velocity_delta: true` (plus model residual head always) |
| Hard no-slip | Implemented in code (`apply_hard_no_slip`); optional checks: `assert_no_slip_train`, `assert_no_slip_val`, `no_slip_atol` |
| BF16 | `use_amp: true` (bf16 autocast on GPU in this trainer) |
| LR / schedule | `lr: 0.0001`, `lr_scheduler: one-cycle` |

**Checkpoints:** by default the trainer writes under `outputs/runs/<run_name>/checkpoints/` (`best.pt`, `latest.pt`) and updates `checkpoint_path` (default `models/fmgreco_stnn/state_dict.pt` relative to repo root). Override with `--checkpoint-path` / `--run-name` or your PVC path (e.g. `/data/checkpoints/...`) so all ranks agree on rank-0 I/O.

**Logs:** rank 0 prints `cuda_mem_*` and loss lines; OOM → lower `latent_dim`, `num_points`, or `num_blocks`, or increase `gradient_accumulation_steps` / use more GPUs.

**Kubernetes:** request **`amd.com/gpu: "8"`** (or your cluster’s AMD device plugin resource) on the training pod so you do not share GPUs with other jobs.
