import os

import torch
from torch.nn import Dropout, Identity, LayerNorm, Linear, Module, ReLU


class MLP(Module):
    num_channels = (18, 256, 15)  # very lean layout, just a proof of concept
    dropout_probability = 0.1

    def __init__(self):
        super().__init__()

        self.linears = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()

        for num_channels_in, num_channels_out in zip(
            self.num_channels[:-2], self.num_channels[1:-1]
        ):
            self.linears.append(Linear(num_channels_in, num_channels_out))
            self.norms.append(LayerNorm(num_channels_out))
            self.activations.append(ReLU())

        self.linears.append(Linear(*self.num_channels[-2:]))
        self.norms.append(Identity())
        self.activations.append(Identity())

        self.dropout = Dropout(self.dropout_probability)

        # Load random weights, just for demonstration
        path_state_dict = os.path.join("models", "mlp", "state_dict.pt")
        state_dict = torch.load(path_state_dict)
        self.load_state_dict(state_dict)

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:

        # Prepare and construct the input
        batch_size, num_t_in, num_pos, _ = velocity_in.shape
        velocity_in = velocity_in.transpose(1, 2).reshape(
            batch_size, num_pos, num_t_in * 3
        )

        x = torch.cat(
            (pos, velocity_in), dim=2
        )  # feel free to also use "t" and "idcs_airfoil"

        # Forward pass
        for linear, norm, activation in zip(self.linears, self.norms, self.activations):
            x = activation(norm(linear(self.dropout(x))))

        # Prepare output
        x = x.view(batch_size, num_pos, num_t_in, 3).transpose(1, 2)

        return x
