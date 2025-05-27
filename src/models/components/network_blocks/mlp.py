import torch
from torch import nn


class MLP(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        embedding_dim: int,
        d_dim: int,
        n_layers: int = 2,
        activation: callable = nn.ReLU,
    ) -> None:
        super().__init__()

        layers = []
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d_dim, d_dim))
            layers.append(activation())
        layers.append(nn.Linear(d_dim, embedding_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
