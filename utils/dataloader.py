"""
Dataset and DataLoader for the GRaM transient airflow dataset.

Each .npz file is one sample: one simulation × one 5-step time window.
Files are named {sim_id}-{window}.npz, e.g. "1021_1-0.npz".

__getitem__ returns:
    inputs  = (velocity_in, pos, idcs_airfoil, t, dist_feats, knn_feat)
    target  = velocity_out

dist_feats is a tuple of three (N,) float32 tensors:
    (is_airfoil, dist_to_airfoil, upstream_dist)
Precomputed once per simulation — pos and idcs_airfoil are fixed per simulation.

knn_feat is either None (if use_local_feats=False) or an (N, k) int64 tensor
of k-nearest-neighbour indices, loaded from the .knncache.npz sidecar file.

Augmentation (training only, controlled by augment=True):
    Y-flip: with 50% probability, negate pos[:,1] and velocity[:,:,1].
    This reflects the geometry about the y=0 plane, producing a valid mirrored
    configuration. dist_feats are invariant under this transform:
      - is_airfoil   : unchanged (same point indices)
      - dist_to_airfoil : Euclidean distance, invariant under reflection
      - upstream_dist   : x-direction offset, invariant under y-reflection

Shapes:
    velocity_in  : (5, 100000, 3)   float32 tensor
    pos          : (100000, 3)      float32 tensor
    idcs_airfoil : (N_surf,)        int64 tensor   — variable length per sample
    t            : (10,)            float32 tensor
    velocity_out : (5, 100000, 3)   float32 tensor
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from models.transolver_residual.features import precompute_distance_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "gram_data")


class GRaMDataset(Dataset):
    def __init__(self, data_dir: str = DATA_DIR, augment: bool = False,
                 use_local_feats: bool = False):
        all_npz = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.files          = [f for f in all_npz
                                if ".distcache." not in f and ".knncache." not in f]
        self.augment        = augment
        self.use_local_feats = use_local_feats
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx])

        velocity_in  = torch.from_numpy(data["velocity_in"])                    # (5, 100k, 3)
        pos          = torch.from_numpy(data["pos"])                             # (100k, 3)
        idcs_airfoil = torch.from_numpy(data["idcs_airfoil"].astype(np.int64))
        t            = torch.from_numpy(data["t"])                               # (10,)
        velocity_out = torch.from_numpy(data["velocity_out"])                    # (5, 100k, 3)

        # Distance features are fixed per simulation — cache to disk so they are
        # only computed once ever, not once per epoch per sample.
        cache_path = self.files[idx].replace(".npz", ".distcache.npz")
        if os.path.exists(cache_path):
            cache = np.load(cache_path)
            ia, dist, xsign = cache["ia"], cache["dist"], cache["xsign"]
        else:
            ia, dist, xsign = precompute_distance_features(
                data["pos"],
                data["idcs_airfoil"].astype(np.int64),
            )
            np.savez_compressed(cache_path, ia=ia, dist=dist, xsign=xsign)

        dist_feat = (
            torch.from_numpy(ia),
            torch.from_numpy(dist),
            torch.from_numpy(xsign),
        )   # three (N,) float32 tensors

        # k-NN cache (optional) — only loaded when use_local_feats=True
        knn_feat = None
        if self.use_local_feats:
            knn_path = self.files[idx].replace(".npz", ".knncache.npz")
            if os.path.exists(knn_path):
                knn_feat = torch.from_numpy(
                    np.load(knn_path)["knn_idx"].astype(np.int64)
                )   # (N, k)

        # ── Y-flip augmentation ───────────────────────────────────────────────
        # Applied to training set only (augment=True).
        # Reflects the geometry about y=0: negate pos_y and vy.
        # dist_feats are invariant and do not need modification.
        if self.augment and torch.rand(1).item() < 0.5:
            pos          = pos.clone();          pos[:, 1]             = -pos[:, 1]
            velocity_in  = velocity_in.clone();  velocity_in[:, :, 1]  = -velocity_in[:, :, 1]
            velocity_out = velocity_out.clone(); velocity_out[:, :, 1] = -velocity_out[:, :, 1]

        return (velocity_in, pos, idcs_airfoil, t, dist_feat, knn_feat), velocity_out


def collate_fn(batch):
    """
    Custom collate: stacks everything except idcs_airfoil (variable-length),
    dist_feats, and knn_feats (kept as lists so the model can index by sample).
    """
    inputs, targets = zip(*batch)
    vel_in_list, pos_list, idcs_list, t_list, dist_feat_list, knn_list = zip(*inputs)

    velocity_in  = torch.stack(vel_in_list)   # (B, 5, 100k, 3)
    pos          = torch.stack(pos_list)       # (B, 100k, 3)
    t            = torch.stack(t_list)         # (B, 10)
    velocity_out = torch.stack(targets)        # (B, 5, 100k, 3)
    idcs_airfoil = list(idcs_list)             # list[B] of variable-length tensors
    dist_feats   = list(dist_feat_list)        # list[B] of (ia, dist, xsign) tuples
    knn_feats    = list(knn_list)              # list[B] of (N, k) tensors or None

    return (velocity_in, pos, idcs_airfoil, t, dist_feats, knn_feats), velocity_out


def make_loaders(
    data_dir: str = DATA_DIR,
    train_fraction: float = 0.9,
    batch_size: int = 1,
    num_workers: int = 4,
    seed: int = 42,
    augment: bool = False,
    use_local_feats: bool = False,
):
    """
    Split the dataset into train/val and return DataLoaders.

    The train and val sets use separate GRaMDataset instances so augmentation
    is applied only to training samples, never to validation.

    When train_fraction=1.0 the val loader is empty (returns an empty DataLoader).

    Args:
        data_dir:        Path to the gram_data directory.
        train_fraction:  Fraction of samples used for training (default 0.9).
        batch_size:      Samples per batch.
        num_workers:     DataLoader worker processes.
        seed:            RNG seed for the split.
        augment:         Enable y-flip augmentation on the training set.
        use_local_feats: Load k-NN cache for local neighbourhood features.

    Returns:
        train_loader, val_loader
    """
    # Determine the split indices once, using a clean (no-augment) dataset.
    base    = GRaMDataset(data_dir, augment=False, use_local_feats=False)
    n_total = len(base)
    n_train = int(n_total * train_fraction)
    n_val   = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    indices   = torch.randperm(n_total, generator=generator).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    # Two dataset instances with different augment flags; same underlying files.
    train_dataset = GRaMDataset(data_dir, augment=augment,
                                use_local_feats=use_local_feats)
    val_dataset   = GRaMDataset(data_dir, augment=False,
                                use_local_feats=use_local_feats)

    train_set = Subset(train_dataset, train_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    if n_val > 0:
        val_set    = Subset(val_dataset, val_idx)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    else:
        val_loader = DataLoader([])   # empty loader when train_fraction=1.0

    aug_str = "y_flip" if augment else "none"
    print(f"Dataset: {n_total} samples  →  train {n_train}  /  val {n_val}  [augment={aug_str}]")
    return train_loader, val_loader
