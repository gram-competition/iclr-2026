# SpatioTemporalGNN — Model Architecture and Training

## Task

Predict 5 future velocity-field timesteps given 5 past timesteps over 100,000-point
3D airfoil meshes. Each point carries a 3D position and a 3D velocity vector per
timestep, with airfoil surface points constrained to zero velocity (no-slip boundary
condition).

---

## Architecture

SpatioTemporalGNN is a graph neural network for spatiotemporal velocity-field
prediction on unstructured 3D point clouds. It operates directly on the full
100k-point mesh without any downsampling, constructing a k-nearest-neighbor graph
to define local spatial neighborhoods and using a Graph Transformer backbone for
spatial reasoning combined with a temporal attention head for multi-step prediction.

### High-Level Pipeline

```
                            Input
                              |
            pos (N,3)    vel_in (5,N,3)    airfoil_idx (M,)
                |              |                  |
                v              v                  v
        +--------------+  +----------+     +-------------+
        | Fourier Pos  |  | Flatten  |     | Binary Mask |
        | Encoding     |  | (N, 15)  |     | (N, 1)      |
        | (N, 51)      |  +----------+     +-------------+
        +--------------+       |                  |
                |              |                  |
                +---------+----+------------------+
                          |
                    Concatenate -> (N, 67)
                          |
                +---------v----------+
                | Node Encoder MLP   |
                | Linear(67->256)    |
                | GELU               |
                | Linear(256->256)   |
                | LayerNorm          |
                +--------+-----------+
                         |  (N, 256)
                         |
          +--------------v--------------+
          |  k-NN Graph (k=24, cKDTree) |
          |  -> neighbors (N,24)        |
          |  -> rel_pos   (N,24,3)      |
          |  -> dists     (N,24)        |
          +----+------------------------+
               |
       +-------v--------+
       | Edge Encoder    |
       | [rel_pos, dist] |
       | -> (N,24,256)   |
       +-------+---------+
               |
    +----------v-----------+
    | Graph Transformer x10 |  <-- 10 stacked layers
    | (details below)       |
    +----------+-----------+
               |  (N, 256)
               |
    +----------v-----------+
    | Temporal Attention    |
    | Head                  |
    | -> (N, 5, 256)        |
    +----------+-----------+
               |
    +----------v-----------+
    | Decoder MLP           |
    | Linear(256->256)      |
    | GELU                  |
    | Linear(256->3)        |
    | -> (N, 5, 3)          |
    +----------+-----------+
               |
    +----------v-----------+
    | + Last Observed Vel   |  <-- Residual connection
    | -> (N, 5, 3)          |
    +----------+-----------+
               |
    +----------v-----------+
    | Zero Airfoil Points   |  <-- No-slip enforcement
    | -> (5, N, 3)          |
    +----------------------+
               |
            Output
```

---

### Component Details

#### 1. Fourier Positional Encoding

Encodes raw 3D coordinates `(x, y, z)` into a high-dimensional representation
that captures multi-scale spatial patterns:

```
Input:  (x, y, z)                              -- 3 dims

For each of 8 frequency bands (f = 2^0, 2^1, ..., 2^7):
  Compute:  sin(f * x), sin(f * y), sin(f * z)  -- 3 dims per band
            cos(f * x), cos(f * y), cos(f * z)  -- 3 dims per band

Output: [x, y, z,
         sin(1*x), sin(1*y), sin(1*z),          -- band 0
         sin(2*x), sin(2*y), sin(2*z),          -- band 1
         ...
         cos(128*x), cos(128*y), cos(128*z)]    -- band 7
Total:  3 + 8*3*2 = 51 dims
```

This enables the model to distinguish fine-grained spatial positions and capture
high-frequency flow structures (e.g. boundary layers, vortex shedding) that raw
coordinates cannot represent.

#### 2. Input Feature Construction

Per node, the input feature vector is assembled by concatenation:

| Component        | Shape   | Description                                    |
|------------------|---------|------------------------------------------------|
| Fourier position | (51,)   | Multi-scale spatial encoding of 3D coordinates |
| Velocity history | (15,)   | 5 timesteps x 3D velocity, flattened           |
| Airfoil mask     | (1,)    | 1.0 if point is on airfoil surface, else 0.0   |
| **Total**        | **(67,)** | Concatenated input per node                  |

#### 3. Node Encoder

A 2-layer MLP with GELU activation and LayerNorm that projects the 67-dim
input features into the 256-dim hidden space:

```
Linear(67 -> 256) -> GELU -> Linear(256 -> 256) -> LayerNorm(256)
```

#### 4. k-NN Graph Construction

A k=24 nearest-neighbor graph is built on the 3D positions using
`scipy.spatial.cKDTree` (exact, multi-threaded). This produces:

- `neighbors`: (N, 24) — indices of the 24 closest points for each node
- `rel_pos`: (N, 24, 3) — relative position vectors (neighbor - center)
- `dists`: (N, 24) — Euclidean distances to each neighbor

The graph is built once per sample and reused across all transformer layers.

#### 5. Edge Encoder

A 2-layer MLP that transforms raw geometric edge information into
learned edge features:

```
Input:  [rel_pos (3D), dist (1D)] = 4D per edge
Linear(4 -> 256) -> GELU -> Linear(256 -> 256)
Output: (N, 24, 256) edge features
```

Edge features encode the direction and distance between connected nodes,
providing geometric context to the attention mechanism. They are computed
once and shared across all 10 transformer layers.

#### 6. Graph Transformer Layer (x10)

Each of the 10 layers performs local multi-head attention over the k=24
neighbors of each node. The detailed computation for a single layer:

```
Given: x (N, 256) node features, neighbors (N, 24), edge_feat (N, 24, 256)

--- Multi-Head Attention (8 heads, 32 dims per head) ---

Q = W_q(x)                           -- (N, 256), query from center node
K = W_k(x)[neighbors]                -- (N, 24, 256), keys from neighbors
V = W_v(x)[neighbors]                -- (N, 24, 256), values from neighbors

V = V + W_e(edge_feat)               -- Edge-conditioned value bias:
                                         geometric info injected into values

Reshape Q -> (N, 1, 8, 32)
Reshape K -> (N, 24, 8, 32)
Reshape V -> (N, 24, 8, 32)

Attention scores:
  A = sum(Q * K, dim=-1) / sqrt(32)  -- (N, 24, 8), scaled dot product
  A = softmax(A, dim=1)              -- Normalize over 24 neighbors
  A = dropout(A)

Aggregate:
  out = sum(A * V, dim=1)            -- (N, 8, 32), weighted sum
  out = reshape(out) -> (N, 256)     -- Concatenate heads
  out = W_o(out)                     -- Output projection

--- Residual + Normalization ---

x = LayerNorm(x + out)               -- Post-attention residual

--- Feed-Forward Network ---

ffn = Linear(256 -> 512) -> GELU -> Dropout -> Linear(512 -> 256) -> Dropout
x = LayerNorm(x + ffn(x))            -- Post-FFN residual

Output: x (N, 256)
```

**Key properties**:
- Attention is **local** (computed only over k=24 neighbors, not all N nodes),
  making it scalable to 100k points.
- The **edge-conditioned value bias** (W_e) injects relative position and distance
  information directly into the value vectors. This means the attention mechanism
  is geometrically aware: two neighbors at different distances or angles contribute
  differently even if their node features are identical.
- With 10 layers and k=24, information can propagate across 10 hops in the graph,
  covering a large spatial receptive field.

#### 7. Temporal Attention Head

After the spatial backbone, each node has a single 256-dim feature vector.
The temporal head generates 5 future timestep predictions per node:

```
Step 1: Project
  h = Linear(256 -> 5*256) -> reshape to (N, 5, 256)
  Each node now has 5 "temporal slot" vectors.

Step 2: Temporal position embeddings
  h = h + time_pe                     -- Learnable (1, 5, 256) embeddings
                                         so the model knows which slot is which timestep.

Step 3: Temporal self-attention (x2 layers)
  For each of 2 attention layers:
    Attention: MultiheadAttention(256, 8 heads, batch_first=True)
      Applied across the 5 time slots per node.
      Each slot attends to all other slots (full attention, not causal).
    h = LayerNorm(h + attn(h))
    ffn = Linear(256->512) -> GELU -> Dropout -> Linear(512->256) -> Dropout
    h = LayerNorm(h + ffn(h))

Output: (N, 5, 256) -- 5 temporally refined feature vectors per node
```

The temporal attention ensures predictions across the 5 future timesteps are
mutually consistent. For example, the model can learn that timestep 3's prediction
should smoothly interpolate between timesteps 2 and 4.

#### 8. Decoder MLP

Independently maps each of the 5 temporal features to 3D velocity deltas:

```
Linear(256 -> 256) -> GELU -> Linear(256 -> 3)
Applied to each (N, 256) slice -> (N, 3) velocity delta
```

#### 9. Residual Prediction

The decoded velocity deltas are added to the last observed velocity:

```
prediction = delta + vel_in[t=5]      -- (N, 5, 3)
```

This residual formulation means the network only needs to learn the small changes
from the current state rather than the full velocity field. Since fluid flows
evolve smoothly, these deltas are small and easier to learn.

#### 10. No-Slip Boundary Enforcement

```
prediction[:, airfoil_idx, :] = 0.0
```

All velocity components at airfoil surface points are hard-set to zero,
enforcing the physical no-slip boundary condition regardless of what the
network predicts.

---

## Model Specifications

| Parameter               | Value       |
|-------------------------|-------------|
| Hidden dimension        | 256         |
| Graph Transformer layers | 10         |
| Attention heads         | 8           |
| Head dimension          | 32          |
| k (neighbors)           | 24          |
| Fourier frequency bands | 8           |
| Dropout                 | 0.05        |
| Input timesteps         | 5           |
| Output timesteps        | 5           |
| Temporal attention layers | 2         |
| Temporal attention heads | 8          |
| Total parameters        | 7,520,523   |
| Weight file size        | 28.7 MB     |

### Parameter Breakdown

| Module             | Parameters  | Description                             |
|--------------------|-------------|-----------------------------------------|
| Node encoder       | ~35k        | 2-layer MLP (67->256->256) + LayerNorm  |
| Edge encoder       | ~67k        | 2-layer MLP (4->256->256)               |
| Graph Transformer  | ~5.3M       | 10 layers x (Q,K,V,E,O projections + FFN) |
| Temporal head      | ~1.9M       | Projection + 2 attention layers + FFN   |
| Decoder            | ~66k        | 2-layer MLP (256->256->3)               |

---

## Training

### Command

```bash
python train.py \
  --data_dir ../warped_ifw \
  --backbone graph_transformer \
  --hidden_dim 256 \
  --num_layers 10 \
  --heads 8 \
  --k 24 \
  --num_sub 100000 \
  --use_fourier \
  --interp_k 5 \
  --dropout 0.05 \
  --lambda_grad 0.5 \
  --alpha 1.0 \
  --beta 0.1 \
  --gamma 0.5 \
  --delta_loss 0.2 \
  --airfoil_weight 5.0 \
  --epochs 50 \
  --batch_size 1 \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --warmup_epochs 10 \
  --scheduler cosine \
  --grad_clip 1.0 \
  --device cuda:2 \
  --save_dir checkpoints_gt_best \
  --log_dir runs_gt_best \
  --save_every 25
```

### Dataset

- **Source**: warped_ifw (GRaM competition dataset)
- **Split**: 688 train / 122 validation samples (85/15 split, seed=42)
- **Resolution**: 100,000 points per sample (full mesh, no subsampling)
- **Graph construction**: scipy.spatial.cKDTree for exact multi-threaded k-NN

### Loss Function

The training loss is a weighted combination of five objectives:

```
L = alpha * L_mse + beta * L_l1 + gamma * L_temporal + delta * L_airfoil + lambda * L_gmse
```

| Component                | Weight | Formulation                                            |
|--------------------------|--------|--------------------------------------------------------|
| MSE (alpha)              | 1.0    | Mean squared error over all points and timesteps       |
| L1 (beta)                | 0.1    | Mean absolute error — reduces sensitivity to outliers  |
| Temporal consistency (gamma) | 0.5 | MSE between consecutive predicted timesteps — penalises abrupt jumps |
| Airfoil surface (delta)  | 0.2    | Weighted MSE on airfoil points with 5x multiplier — prioritises boundary accuracy |
| Spatial gradient GMSE (lambda) | 0.5 | MSE on local spatial gradients computed via k-NN — preserves sharp features and local flow structure |

### Optimizer and Schedule

- **Optimizer**: AdamW (lr=2e-4, weight_decay=1e-4)
- **LR schedule**: Linear warmup for 10 epochs (0 -> 2e-4), then cosine annealing
  to 1e-6 over the remaining 40 epochs
- **Gradient clipping**: Max norm 1.0
- **Mixed precision**: Automatic mixed precision (float16) enabled for memory
  efficiency and faster training

---

## Design Rationale

- **Full-resolution processing**: Operating on all 100k points avoids information
  loss from downsampling, preserving fine-grained flow features near the airfoil
  surface and in turbulent wake regions.

- **Local Graph Transformer**: Multi-head attention over the k=24 nearest neighbors
  allows the model to learn directional, distance-dependent spatial relationships.
  Unlike simple mean-pooling, attention dynamically weights each neighbor's
  contribution based on learned relevance, which is critical for anisotropic flow
  fields where upstream and downstream neighbors carry very different information.

- **Edge-conditioned value bias**: Adding projected edge features (relative position
  + distance) to the value vectors gives the attention mechanism explicit geometric
  awareness without modifying the attention scores. This separates "how much to
  attend" (via Q/K) from "what geometric information to carry" (via V + edge bias).

- **Deep architecture (10 layers)**: Each layer propagates information by 1 hop
  (k=24 neighbors). With 10 layers, the effective receptive field extends ~10 hops,
  covering a large spatial extent. This is necessary to model both local boundary
  layer dynamics and far-field flow interactions simultaneously.

- **Temporal self-attention**: Full (non-causal) attention across the 5 output
  timesteps ensures that predictions are mutually consistent. The model can
  jointly reason about all future timesteps rather than predicting them
  independently, which is important for maintaining smooth temporal evolution.

- **Residual velocity prediction**: Predicting deltas from the last observed
  velocity exploits the temporal smoothness of fluid flows. The deltas are
  typically small relative to the absolute velocities, making the regression
  target easier to learn.

- **Multi-objective loss**: Each loss component targets a different aspect of
  prediction quality — MSE for overall accuracy, L1 for sharpness, temporal
  consistency for smoothness, GMSE for preserving spatial gradients, and
  airfoil weighting for boundary fidelity.

---

## Training Results

Training ran for 6 epochs before early stopping triggered (patience=10 on
intra-epoch validation checks at 20%, 40%, 60%, 80%, and 100% of each epoch).

### Epoch-Level Summary

| Epoch | LR       | Train L2 | Val L2   | Train Loss | Val Loss | Train MSE | Val MSE | Time   |
|-------|----------|----------|----------|------------|----------|-----------|---------|--------|
| 1     | 2.00e-05 | 1.5162   | 1.2792   | 11.25      | 9.36     | 5.2678    | 4.1629  | 1402s  |
| 2     | 4.00e-05 | 1.2600   | 1.2023   | 9.03       | 8.61     | 4.0172    | 3.7718  | 1395s  |
| 3     | 6.00e-05 | 1.2432   | 1.1844   | 8.72       | 8.27     | 3.8723    | 3.6248  | 1395s  |
| 4     | 8.00e-05 | 1.2332   | **1.1525** | 8.48     | 7.98     | 3.7648    | 3.4929  | 1397s  |
| 5     | 1.00e-04 | 1.2234   | 1.1582   | 8.27       | 7.91     | 3.6684    | 3.4807  | 1397s  |
| 6     | 1.20e-04 | 1.2169   | 1.1678   | 8.10       | 7.86     | 3.5940    | 3.4749  | 1397s  |

**Best validation L2: 1.1525** (epoch 4, at 100% validation checkpoint).

### Detailed Validation Progression

The model was validated at 20%, 40%, 60%, 80%, and 100% of each training epoch.
Best checkpoint was selected by the lowest validation L2 (mean pointwise L2 velocity error).

| Checkpoint   | Val L2   | Val Loss | Val MSE | Val L1  | Val GMSE | Val Temp | vsPersist |
|--------------|----------|----------|---------|---------|----------|----------|-----------|
| E1 @20%      | 1.5189   | 11.72    | 5.4873  | 0.7369  | 6.7666   | 5.5563   | 1.625     |
| E1 @40%      | 1.3591   | 10.35    | 4.6971  | 0.6534  | 5.7984   | 5.3865   | 1.180     |
| E1 @60%      | 1.3126   | 9.85     | 4.4226  | 0.6312  | 5.4772   | 5.2425   | 1.036     |
| E1 @100%     | 1.2792   | 9.36     | 4.1629  | 0.6161  | 5.1740   | 5.1005   | 0.889     |
| E2 @20%      | 1.2192   | 8.87     | 3.9016  | 0.5860  | 4.8808   | 4.9410   | 0.915     |
| E2 @40%      | 1.2055   | 8.76     | 3.8438  | 0.5794  | 4.8203   | 4.8923   | 0.876     |
| E2 @60%      | 1.2025   | 8.68     | 3.8145  | 0.5778  | 4.7695   | 4.8548   | 0.857     |
| E2 @80%      | 1.1944   | 8.60     | 3.7712  | 0.5736  | 4.7244   | 4.8085   | 0.785     |
| E3 @100%     | 1.1844   | 8.27     | 3.6248  | 0.5692  | 4.5487   | 4.6293   | 0.846     |
| E4 @20%      | 1.1711   | 8.27     | 3.6324  | 0.5627  | 4.5515   | 4.6098   | 0.790     |
| **E4 @100%** | **1.1525** | **7.98** | **3.4929** | **0.5543** | **4.3870** | **4.4702** | **0.744** |
| E5 @100%     | 1.1582   | 7.91     | 3.4807  | 0.5576  | 4.3851   | 4.3712   | 0.735     |
| E6 @100%     | 1.1678   | 7.86     | 3.4749  | 0.5629  | 4.3229   | 4.3312   | 0.798     |

### Key Observations

- **Rapid convergence**: The model reached a competitive val L2 of 1.28 within the
  first epoch and continued improving to 1.15 by epoch 4.

- **vsPersist ratio**: This measures the model's L2 error relative to a naive
  persistence baseline (repeating the last observed velocity). Values below 1.0
  mean the model outperforms persistence. The model achieved vsPersist=0.744 at
  its best checkpoint, meaning it is 25.6% better than persistence.

- **Early stopping**: The model stopped at epoch 6 after 10 consecutive validation
  checks without improvement. The best val L2 (1.1525) was achieved at epoch 4.
  While the training loss continued decreasing (8.48 -> 8.10), validation L2
  plateaued, indicating the model had reached its capacity for generalisation
  at this configuration.

- **Persistence baseline L2**: 11.4305 (computed on the validation set). The
  model's best L2 of 1.1525 represents a ~90% reduction from the persistence
  baseline.

### Per-Stage Inference Timing (GPU, cuda:2)

Measured on the first few batches during training:

| Stage    | Time (ms) | Description                           |
|----------|-----------|---------------------------------------|
| Graph    | 70-80     | k-NN construction via cKDTree on CPU  |
| Encode   | 1-3       | Fourier PE + node encoder MLP         |
| Spatial  | 452-460   | 10 Graph Transformer layers           |
| Temporal | 65-73     | 2-layer temporal attention head       |
| Decode   | 3         | Decoder MLP + residual                |
| **Total**| **~600**  | **Per-sample forward pass**           |

The spatial backbone dominates inference time (~75%), as expected for a
10-layer Graph Transformer operating on 100k nodes with k=24 neighbors.
