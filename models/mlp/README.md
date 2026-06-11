# Airflow Prediction on Warped Airfoil Geometries

**GRaM Competition @ ICLR 2026** — Geometry-grounded Representation Learning and Generative Modeling

[![Competition](https://img.shields.io/badge/GRaM-Competition-blue)](https://gram-competition.github.io)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/gram-competition/warped-ifw)
[![Workshop](https://img.shields.io/badge/ICLR_2026-GRaM_Workshop-green)](https://gram-workshop.github.io)

<p align="center">
  <img src="https://gram-competition.github.io/assets/front_wing.png" width="600"/>
</p>

## Submission Model

The submission model is `MLP`, located in `models/mlp/model.py`. It is instantiated with no arguments — pretrained weights are loaded automatically from `baseline_mno_seed118_base_2_32k_budget_crop_5580430/checkpoints/best.pt` on construction. The competition-required calling convention is:

```python
from models.mlp import MLP

model = MLP()                                          # loads weights automatically
velocity_out = model(t, pos, idcs_airfoil, velocity_in)  # (B, 5, N, 3)
```

`infer.py` at the project root contains a complete, self-contained example that loads `1021_1-0.npz` [Placeholder currently], runs inference using exactly the above signature, and prints per-timestep RL2 and overall improvement over the persistence baseline.

## Challenge

Given a 3D velocity field $\mathbf{u}(t, \mathbf{x})$ of airflow around warped airfoil geometries derived from the [Imperial Front Wing (IFW)](https://data.hpc.imperial.ac.uk/resolve/?doi=6049), predict the velocity field at 5 future timesteps from 5 observed timesteps.

The dataset spans **181 unique geometries** — each composed of one, two, or three differently-sized airfoils at randomly sampled relative positions and pitch angles — with 5 time windows extracted per simulation. Each sample contains **100k spatial points** in 3D with a **no-slip boundary condition** enforced on the airfoil surface.

## Approach

The model is an Encoder → 4× MNO Block → Decoder architecture operating directly on unstructured 3D point clouds. Each MNO Block combines global dimension-shrinkage attention, local kNN graph attention, and micro point-wise attention with FFN residual connections. The encoder receives per-point concatenated features: 3D position, velocity history (5 frames × 3 channels), Fourier time embeddings, an airfoil surface mask, log wall-distance, and a local surface frame. The decoder predicts a residual around the persistence baseline (last observed frame), with the final layer zero-initialized. Airfoil boundary points are analytically forced to zero velocity (no-slip) in normalized space rather than learned. Per-sample velocity normalization is computed from the input frames and applied automatically inside `forward`.

### Encoder Inputs

The encoder MLP receives a concatenation of the following per-point features:

| Feature | Shape per point | Dim | Description |
|---------|----------------|-----|-------------|
| **3D Position** | $(x, y, z)$ | 3 | Raw spatial coordinates of each mesh point |
| **Velocity history** | $(u, v, w) \times T_{in}$ | 15 | Velocity vectors at each of the 5 input timesteps, flattened |
| **Fourier time embedding** | $[\sin(\omega t), \cos(\omega t)]$ | 320 | 16 learnable frequencies × 2 (sin+cos) × 10 timestamps ($T_{in} + T_{out}$), broadcast to all points |
| **Airfoil mask** | $\{0, 1\}$ | 1 | Binary flag: 1 for points on the airfoil surface, 0 for fluid |
| **Wall distance** | $\log(1 + d)$ | 1 | Log-transformed minimum Euclidean distance to the nearest airfoil surface point |
| **Surface frame** | $[\mathbf{n}, \mathbf{t_1}, \mathbf{t_2}]$ | 9 | Local surface coordinate frame: outward normal + two tangent vectors, computed from nearest airfoil geometry |
| | | **349** | **Total encoder input dimension** |

Wall distance and surface frame are precomputed in the dataset (via `cKDTree`) and passed through the pipeline; at inference on unseen geometries they are computed on the fly.

### Loss Functions

| Loss | Description |
|------|-------------|
| **Sobolev** | Relative L2 + kNN-estimated spatial gradient penalty, encouraging smooth predictions in the boundary layer |
| **Wall-distance weighting** | Per-point loss weights $w_i \propto (d_i + \varepsilon)^{-\alpha}$, focusing optimization on near-wall regions where error matters most |

### Training Pipeline

| Feature | Detail |
|---------|--------|
| **Precision** | BFloat16 AMP with TF32 tensor cores |
| **Optimizer** | AdamW with OneCycleLR scheduling |
| **Compilation** | Optional `torch.compile` with dynamo skip for custom CUDA ops |
| **Data loading** | Boundary-biased point sampling, persistent workers, prefetched batches |
| **Wall distance** | Precomputed once in dataset (cKDTree), passed through the full pipeline to avoid redundant computation in both the model forward pass and loss weighting |

## Project Structure

```
├── models/
│   └── mlp/                     # Submission model package
│       ├── model.py             # MNO architecture (Encoder, MNOBlock, Decoder)
│       └── training.py          # Loss functions (RL2, Sobolev, Continuity)
├── src/
│   ├── data/
│   │   └── dataset.py           # Dataset, boundary-biased sampling, collate
│   ├── training/
│   │   ├── trainer.py           # Training orchestration, CLI, checkpointing
│   │   └── loop.py              # Forward pass, evaluation
│   ├── evaluation/              # Inference utilities
│   └── utils/                   # Config loading
├── config/
│   └── baseline.yaml            # Default hyperparameters
├── notebooks/
│   └── main.ipynb               # Training + visualization (Colab-ready)
├── scripts/
│   ├── train.py                 # CLI training entry point
│   └── evaluate.py              # CLI evaluation entry point
└── tests/                       # Unit tests
```

## Quick Start

**Run inference on an example sample:**
```bash
python infer.py
```

**Local training:**
```bash
python scripts/train.py --config config/baseline.yaml \
    --dataset-dir dataset_huggingface/warped-ifw \
    --epochs 200 --batch-size 8 --loss-fn sobolev
```

**Google Colab:** Open `notebooks/main.ipynb` — handles dataset download, environment setup, training, and visualization end-to-end.

## Results Visualization

The notebook includes diagnostic cells for:

- **Error vs. distance from boundary** — scatter plot revealing whether the model struggles near the wall or in the freestream
- **Full-resolution side-view cut** — interpolated velocity magnitude on an x-z plane through the airfoil, showing ground truth, prediction, and signed delta across all predicted timesteps

## Dataset

Downloaded from [HuggingFace](https://huggingface.co/datasets/gram-competition/warped-ifw). Each `.npz` sample contains:

| Field | Shape | Description |
|-------|-------|-------------|
| `t` | `(10,)` | Timestamps (5 input + 5 output) |
| `pos` | `(100k, 3)` | 3D point positions |
| `idcs_airfoil` | `(~8k–20k,)` | Indices of airfoil surface points |
| `velocity_in` | `(5, 100k, 3)` | Input velocity field (5 frames) |
| `velocity_out` | `(5, 100k, 3)` | Target velocity field (5 frames) |

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- `scipy`, `numpy`, `matplotlib`, `pyyaml`
- `torch-cluster` (optional, faster kNN; falls back to cdist)
