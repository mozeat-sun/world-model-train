"""StockSelectionPolicy (PyTorch) — Planner for imagination-based RL."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StockSelectionPolicy(nn.Module):
    """Policy: z_t + z_future → stock scores → weights."""

    def __init__(self, latent_dim=128, max_symbols=100, predict_horizon=20,
                 hidden=64, temperature=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_symbols = max_symbols
        self.temperature = temperature
        future_dim = latent_dim * predict_horizon  # 2560
        self.agg_proj = nn.Linear(latent_dim + future_dim, latent_dim)
        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
        )
        self.output = nn.Linear(hidden, max_symbols)

    def forward(self, z_current, z_future, exploratory=False, alpha=0.03, epsilon=0.05):
        """
        Args:
            z_current: (B, D) current latent state
            z_future:  (B, H, D) imagined future trajectory
        Returns:
            weights: (B, N), log_probs: (B, N), entropy: scalar
        """
        B = z_current.shape[0]
        z_flat = z_future.reshape(B, -1)  # (B, H*D)
        z_agg = F.relu(self.agg_proj(torch.cat([z_current, z_flat], dim=-1)))
        h = self.hidden(z_agg)
        logits = self.output(h)

        if exploratory:
            noise = torch.distributions.Dirichlet(
                torch.full((B, self.max_symbols), alpha, device=logits.device)
            ).sample()
            logits = (1 - epsilon) * logits + epsilon * noise

        weights = F.softmax(logits / self.temperature, dim=-1)
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        entropy = -(weights * log_probs).sum(-1).mean()
        return weights, log_probs, entropy

    def export_inference_weights(self) -> dict:
        return {
            'agg_proj.weight': self.agg_proj.weight.data,
            'agg_proj.bias': self.agg_proj.bias.data,
            'hidden.0.weight': self.hidden[0].weight.data,
            'hidden.0.bias': self.hidden[0].bias.data,
            'output.weight': self.output.weight.data,
            'output.bias': self.output.bias.data,
        }
