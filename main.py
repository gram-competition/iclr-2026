from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from models import MLP as Model

# The model constructor has to be callable without arguments
model = Model()

# Load test split
t, pos, idcs_airfoil, velocity_in, ground_truth = [], [], [], [], []
for path in glob("warped-ifw-test-split/*.npz"):
    sample = np.load(path)
    t.append(sample["t"])
    pos.append(sample["pos"])
    idcs_airfoil.append(torch.from_numpy(sample["idcs_airfoil"]))
    velocity_in.append(sample["velocity_in"])
    ground_truth.append(sample["velocity_out"])
t = torch.from_numpy(np.stack(t))
pos = torch.from_numpy(np.stack(pos))
velocity_in = torch.from_numpy(np.stack(velocity_in))
ground_truth = torch.from_numpy(np.stack(ground_truth))

# Dimensions of the data
BATCH_SIZE = 95  # number of point clouds in the test split
NUM_T_IN = 5  # number of time points in the input
NUM_T_OUT = 5  # number of time points in the output
NUM_POS = 100000  # number of points in space
assert t.shape == (BATCH_SIZE, NUM_T_IN + NUM_T_OUT)
assert pos.shape == (BATCH_SIZE, NUM_POS, 3)
assert len(idcs_airfoil) == BATCH_SIZE
assert velocity_in.shape == (BATCH_SIZE, NUM_T_IN, NUM_POS, 3)
assert ground_truth.shape == (BATCH_SIZE, NUM_T_OUT, NUM_POS, 3)

# The model has to return batched estimates
cake = [slice(piece[0], piece[-1] + 1) for piece in torch.arange(BATCH_SIZE).split(2)]
velocity_out = []
for piece in tqdm(cake):
    inputs = [input_[piece] for input_ in (t, pos, idcs_airfoil, velocity_in)]
    with torch.no_grad():
        velocity_out.append(model(*inputs))
velocity_out = torch.cat(velocity_out)
assert velocity_out.shape == (BATCH_SIZE, NUM_T_OUT, NUM_POS, 3)

# The final evaluation metric is relative L² error:
numerator = ((ground_truth - velocity_out) ** 2).sum(dim=(3, 2, 1))
denominator = (ground_truth**2).sum(dim=(3, 2, 1))
metric = (numerator / denominator).sqrt()
print(f"{type(model).__name__}: {metric.mean():.4f} +- {metric.std():.4f}")
