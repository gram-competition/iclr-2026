import torch

from models import PTPointNet as Model

# The model constructor has to be callable without arguments
model = Model()
device = next(model.parameters()).device

# Dimensions of the data
BATCH_SIZE = 5  # indicative per-call batch; held-out test split has 95 samples total
NUM_T_IN = 5  # number of time points in the input
NUM_T_OUT = 5  # number of time points in the output
NUM_POS = 100000  # number of points in space

# Dummy data as placeholder for the test split data of the challenge
t = torch.rand((BATCH_SIZE, NUM_T_IN + NUM_T_OUT), device=device)
pos = torch.rand((BATCH_SIZE, NUM_POS, 3), device=device)
idcs_airfoil = [
    torch.randint(NUM_POS, size=(num_idcs,), device=device)
    for num_idcs in torch.randint(3142, 24198, size=(BATCH_SIZE,))
]  # variable across point clouds so we cannot use batch dimension
velocity_in = torch.rand((BATCH_SIZE, NUM_T_IN, NUM_POS, 3), device=device)
ground_truth = torch.rand((BATCH_SIZE, NUM_T_OUT, NUM_POS, 3), device=device)

# The model has to return batched estimates
velocity_out = model(t, pos, idcs_airfoil, velocity_in)
assert velocity_out.shape == (BATCH_SIZE, NUM_T_OUT, NUM_POS, 3)

# The final evaluation metric is secret, the following is a hint:
metric = (velocity_out - ground_truth).norm(dim=3).mean(dim=(1, 2))
print(f"Metric: {metric.mean():.4f} +- {metric.std():.4f}")
