"""MarketDynamics (PyTorch) — learned market state transition."""

import torch
import torch.nn as nn


class MarketDynamics(nn.Module):
    """
    Dynamics: z_t, a_t → z_{t+1} = z_t + Δ.

    Architecture: z_proj + a_proj → Concat → GRU(2层) → Delta proj.
    """

    def __init__(self, latent_dim=128, max_symbols=100, hidden_dim=256, n_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.learnable_scale = nn.Parameter(torch.tensor(0.1))

        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.a_proj = nn.Linear(max_symbols, 64)
        self.gru = nn.GRU(
            hidden_dim + 64, hidden_dim,
            num_layers=n_layers, batch_first=True
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z, action=None, hidden=None):
        """
        Args:
            z:      (B, D) current state
            action: (B, N) or None
            hidden: GRU hidden state
        Returns:
            z_next: (B, D)
            hidden: updated GRU hidden state
        """
        h = self.z_proj(z)  # (B, H)

        if action is not None:
            a_h = self.a_proj(action)  # (B, 64)
            h = torch.cat([h, a_h], dim=-1)  # (B, H+64)
        else:
            zeros = torch.zeros(h.shape[0], 64, device=h.device)
            h = torch.cat([h, zeros], dim=-1)

        h = h.unsqueeze(1)  # (B, 1, H+64)
        gru_out, hidden = self.gru(h, hidden)
        gru_out = gru_out.squeeze(1)  # (B, H)

        delta = self.delta_proj(gru_out)
        gate = torch.tanh(self.learnable_scale)
        return z + gate * delta, hidden

    def rollout(self, z_0, action_seq=None, n_steps=20):
        """Multi-step autoregressive rollout."""
        trajectory = []
        z = z_0
        with torch.no_grad():
            for step in range(n_steps):
                a = action_seq[:, step, :] if action_seq is not None else None
                z, _ = self.forward(z, a)
                trajectory.append(z)
        return torch.stack(trajectory, dim=1)  # (B, H, D)

    def export_inference_weights(self) -> dict:
        """Export weights for C++ inference."""
        return {
            'z_proj.weight': self.z_proj.weight.data,
            'z_proj.bias': self.z_proj.bias.data,
            'a_proj.weight': self.a_proj.weight.data,
            'a_proj.bias': self.a_proj.bias.data,
            'gru.weight_ih_l0': self.gru.weight_ih_l0.data,
            'gru.weight_hh_l0': self.gru.weight_hh_l0.data,
            'gru.bias_ih_l0': self.gru.bias_ih_l0.data,
            'gru.bias_hh_l0': self.gru.bias_hh_l0.data,
            'gru.weight_ih_l1': self.gru.weight_ih_l1.data,
            'gru.weight_hh_l1': self.gru.weight_hh_l1.data,
            'gru.bias_ih_l1': self.gru.bias_ih_l1.data,
            'gru.bias_hh_l1': self.gru.bias_hh_l1.data,
            'delta_proj.0.weight': self.delta_proj[0].weight.data,
            'delta_proj.0.bias': self.delta_proj[0].bias.data,
            'delta_proj.3.weight': self.delta_proj[3].weight.data,
            'delta_proj.3.bias': self.delta_proj[3].bias.data,
            'learnable_scale': self.learnable_scale.data,
        }
