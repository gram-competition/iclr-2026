# GRaM Competition @ ICLR 2026 — Transient Airflow Prediction



# Project Environment
- This project relies on a strict Python virtual environment located at `./venv`.
- Do NOT use the system `python` or `pip`.
- Whenever you need to execute Python code or install dependencies, either explicitly use `./venv/bin/python` OR run `source ./venv/bin/activate` first in your bash execution block.
- This environment hold everything that is in `requirements.txt`


## Goal

Predict 5 future velocity field timesteps from 5 input timesteps on irregular 3D point clouds around F1-style airfoil geometries. Submission is a GitHub PR to the competition repo. Deadline: April 22, 2026 (AoE).

The evaluation metric is undisclosed but measures pointwise accuracy (similarity between predicted and ground-truth 3D velocity fields) on a held-out test set.

## Problem Structure

- **Input:** `velocity_in` (B, 5, 100k, 3), `pos` (B, 100k, 3), `t` (B, 10), `idcs_airfoil` list of variable-length tensors
- **Output:** `velocity_out` (B, 5, 100k, 3)
- Points are **fixed per sample** (Eulerian frame). They differ across samples (different meshes per geometry).
- Geometries are **anisotropic warps of the same base airfoil** (Imperial Front Wing), with 1–3 airfoils at varying positions and pitch angles.
- Constant freestream velocity, left-to-right. Consistent y+ regime across all simulations.
- 181 simulations × 5 time windows = 905 samples. All data can be used for training.
- **No-slip boundary condition:** velocity is exactly (0,0,0) on airfoil surface points (`idcs_airfoil`).
- The organizers state the main difficulty is predicting **high-frequency (turbulent) components**; the low-frequency (laminar) part is well-approximated by the input velocity prior.

## Architecture: Transolver + Residual Learning

### Overall Pipeline (forward pass)

```
def __call__(self, t, pos, idcs_airfoil, velocity_in) -> velocity_out:
    1. Normalize pos to [0,1]^3 per sample (center + rescale bounding box). The domain runs 0–2 in x and 0–1.2 in z. Make sure the normalization maps the actual point cloud bounding box to [0,1]³, not some assumed fixed range.
    2. Polynomial extrapolation: per-point quadratic fit on 5 input timesteps → extrapolate 5 output timesteps (no learned params)
    3. Compute per-point input features (see below)
    4. Per-point MLP encoder: features → hidden dim (128)
    5. L Transolver blocks with Physics-Attention (M=32 slices, hidden=128)
    6. Per-point MLP decoder: hidden → 15 (5 timesteps × 3 velocity components)
    7. Output = polynomial_extrapolation + learned_correction
    8. Hard zero on idcs_airfoil points (no-slip enforcement)
    return velocity_out
```

### Per-Point Input Features (~40 channels)

- `pos_normalized` (3): position in [0,1]^3
- `velocity_in` (15): all 5 input velocity snapshots flattened (5×3)
- `poly_residual` (15): difference between actual input velocities and polynomial fit on input window — proxy for local turbulence intensity
- `temporal_mean` (3): mean velocity over 5 input timesteps
- `temporal_variance` (3): variance per component over 5 input timesteps
- `is_airfoil` (1): binary flag from idcs_airfoil
- `dist_to_airfoil` (1): distance to nearest airfoil surface point
- `upstream_dist` (1): signed distance in the freestream (x) direction to nearest airfoil point — encodes upstream/downstream asymmetry
- `t_values` (10): time values from t tensor, broadcast to all points as global conditioning

### Transolver Physics-Attention (core mechanism)

Each Transolver block:
1. Project per-point features (N×C) → slice weights (N×M) via learned linear + softmax
2. Aggregate: weighted combination → M physics-aware tokens (M×C)
3. Standard attention among M tokens (M is small, e.g. 32 — this is cheap)
4. Broadcast tokens back to N points using slice weights
5. Feedforward + residual connections

Complexity: O(N·M·C + M²·C) — linear in N since M is constant.

The slices will learn to separate physical regimes (freestream, boundary layer, wake, inter-airfoil gaps) without explicit supervision.

### Residual Learning Strategy

The polynomial extrapolation handles low-frequency dynamics for free. The Transolver only needs to learn the **correction** — the high-frequency turbulent component that the polynomial gets wrong. This:
- Reduces the learning burden on the network
- Means the model degrades gracefully (polynomial baseline is decent even if network fails)
- Focuses capacity on the wake region where it matters most

### No-Slip Enforcement

After computing the output, zero out all velocity components at `idcs_airfoil` indices. This is exact, costs nothing, and guarantees boundary condition satisfaction.

## Stack

- **PyTorch** end-to-end (the evaluators run a PyTorch test harness — mixing frameworks causes GPU memory conflicts)
- Base code adapted from https://github.com/thuml/Transolver
- Optimizer: Adam + cosine annealing (via `torch.optim` and `torch.optim.lr_scheduler`)
- Loss: relative L2 on velocity_out (train on the correction, not the raw field)

## Submission Format

```
models/
└── transolver_residual/
    ├── __init__.py           # exports TransolverResidual
    ├── model.py              # full model class
    ├── physics_attention.py  # Transolver block with Physics-Attention
    ├── features.py           # per-point feature computation
    ├── polynomial.py         # polynomial extrapolation utilities
    ├── weights.pt            # trained model weights (or download link)
    └── README.md             # training description for proceedings
```

The model class must:
- Be instantiable with no arguments: `model = TransolverResidual()`
- Load weights in `__init__`
- Match the call signature:

```python
def __call__(
    self,
    t: torch.Tensor,           # (B, 10)
    pos: torch.Tensor,         # (B, 100k, 3)
    idcs_airfoil: list[torch.Tensor],  # variable-length, indexing pos
    velocity_in: torch.Tensor  # (B, 5, 100k, 3)
) -> torch.Tensor:             # (B, 5, 100k, 3)
```

## Key Constraints

- **40 GB GPU** — batch size will be small (1–2 for 100k points). May need gradient accumulation.
- **Solo participant, ~14 days** — no time for elaborate hyperparameter search.
- **905 training samples** — risk of overfitting with a large model. Keep model modest (~5–10M params). Use dropout or weight decay.

## Data

- Source: Hugging Face (link on competition page)
- Location on disk: `./data/gram/`
- Format: .npz files
- Loader should yield batches of `(t, pos, idcs_airfoil, velocity_in, velocity_out)`.
- `pos` varies per simulation (not per time window within the same simulation). Time windows from the same simulation share `pos` and `idcs_airfoil`.
Format: `.npz` files loaded via `numpy.load`.
Naming Convention: `{simulation_ID}-{time_window_index}.npz` (e.g., `1021_1-0.npz`).
Each file represents a single, independent sample in the PyTorch Dataset.

| Dictionary Key | Tensor Shape | Role | Description |
| :--- | :--- | :--- | :--- |
| `velocity_in` | `[5, 100000, 3]` | **Model Input** | Velocity field for input timesteps $t_0$ to $t_4$. |
| `pos` | `[100000, 3]` | **Model Input** | Static 3D spatial coordinates of all points. |
| `idcs_airfoil` | `[N]` (Variable) | **Model Input** | Indices of points belonging to the solid airfoil surface. |
| `velocity_out` | `[5, 100000, 3]` | **Target (Label)** | Ground truth velocity for future timesteps $t_5$ to $t_9$. |
| `t` | `[10]` | **** | Time vector. Do not load or process. |
| `pressure` | `[10, 100000]` | **** | Pressure field. Do not load or process. |
</dataset_schema>

<processing_rules>
1. DATALOADER: Dataset size equals the total number of `.npz` files. `__getitem__` must return `(velocity_in, pos, idcs_airfoil), velocity_out`.
2. DATA LEAKAGE PREVENTION: `velocity_out` MUST NEVER be passed to the model's forward pass. It is strictly for MSE loss computation in the training loop.
3. BOUNDARY CONDITION ENFORCEMENT: Inside the model's forward pass, use `idcs_airfoil` to compute the Signed Distance Field (SDF). Before returning the final prediction, the model MUST forcefully mask the output velocities at `idcs_airfoil` to exactly `[0.0, 0.0, 0.0]` to enforce the no-slip boundary condition.

## Development Priorities (in order)

1. Download data, inspect format, write data loader
2. Implement polynomial extrapolation and verify it produces reasonable baselines
3. Implement per-point feature computation
4. Implement Transolver blocks (adapt from official repo)
5. Training loop with relative L2 loss
6. Train, evaluate, iterate on features and hyperparameters
7. Package submission, test the __call__ interface, write README.md
8. Submit PR

## Physical Inductive Biases to Remember

- All geometries are warps of the same base shape — the SDF/distance features encode this variation
- Freestream is constant, left-to-right — upstream points have simpler dynamics than downstream
- Turbulence is generated at the surface and advected into the wake — the wake region (downstream, high temporal variance) is where prediction is hardest
- Points with high temporal variance in the input window are likely turbulent and need the most correction
- The number of airfoils (1–3) and their relative arrangement determines interference patterns in the wake

## Things NOT to Do

- Don't use JAX or any non-PyTorch framework (GPU memory conflicts with evaluator)
- Don't build explicit graphs (too expensive on 100k points with 40GB)
- Don't project to a regular grid / FNO (spectral bias smooths out turbulence — the exact thing we need to predict)
- Don't use autoregressive rollout (the task is single-shot 5→5, not step-by-step)
- Don't overcomplicate the architecture — a working submission beats an unfinished one