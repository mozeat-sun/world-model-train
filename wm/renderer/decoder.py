"""MarketRenderer/Decoder — training only, discarded after Phase 1."""

import torch
import torch.nn as nn


class MarketRenderer(nn.Module):
    """
    Decoder: z_t (D,) → ô_t (T, N, C) reconstructed OHLCV.
    Mirror of Encoder, only used in Phase 1 VAE training.
    """

    def __init__(self, latent_dim=128, n_channels=11, max_symbols=100,
                 window_len=60, gru_hidden=256):
        super().__init__()
        self.window_len = window_len
        self.max_symbols = max_symbols
        self.n_channels = n_channels

        self.fc_in = nn.Linear(latent_dim, gru_hidden)
        self.gru = nn.GRU(gru_hidden, gru_hidden, num_layers=2, batch_first=True)
        self.output_proj = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, n_channels),
        )

    def forward(self, z):
        """
        Args:
            z: (B, D) latent state
        Returns:
            ô: (B, T, N, C) reconstructed observation
        """
        B = z.shape[0]
        # Expand z to sequence: repeat for each time step
        h = self.fc_in(z)  # (B, H)
        h = h.unsqueeze(1).repeat(1, self.window_len, 1)  # (B, T, H)
        gru_out, _ = self.gru(h)  # (B, T, H)

        # Project to (N, C) per time step
        out = self.output_proj(gru_out)  # (B, T, C)
        out = out.unsqueeze(2).repeat(1, 1, self.max_symbols, 1)  # (B, T, N, C)
        return out
