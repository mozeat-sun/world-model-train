"""ValueCritic (PyTorch) — state value estimator V(z_t)."""

import torch
import torch.nn as nn


class ValueCritic(nn.Module):
    """Value function: z_t → V(z_t) ∈ [-1, 1]."""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        return torch.tanh(self.net(z).squeeze(-1))

    def export_inference_weights(self) -> dict:
        return {
            'hidden1.weight': self.net[0].weight.data,
            'hidden1.bias': self.net[0].bias.data,
            'hidden2.weight': self.net[2].weight.data,
            'hidden2.bias': self.net[2].bias.data,
            'output.weight': self.net[4].weight.data,
            'output.bias': self.net[4].bias.data,
        }
