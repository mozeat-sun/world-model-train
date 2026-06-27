"""MarketEncoder (PyTorch) — VAE training + deterministic inference export."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarketEncoder(nn.Module):
    """
    MVP Encoder: LayerNorm → Per-symbol Linear → GRU → LayerNorm → (mu, logvar).

    Full version adds: 1D Conv, Multi-head Cross-Attention, Mask Embedding.
    """

    def __init__(self, n_channels=11, max_symbols=100, latent_dim=128,
                 gru_hidden=256, gru_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_symbols = max_symbols
        self.n_channels = n_channels

        self.norm_in = nn.LayerNorm(n_channels)
        self.symbol_proj = nn.Linear(n_channels, latent_dim)
        self.gru = nn.GRU(
            latent_dim, gru_hidden,
            num_layers=gru_layers, batch_first=True
        )
        self.norm_out = nn.LayerNorm(gru_hidden)

        # VAE heads (stripped in export)
        self.mu_head = nn.Linear(gru_hidden, latent_dim)
        self.logvar_head = nn.Linear(gru_hidden, latent_dim)

    def forward(self, obs, mask=None, deterministic=False):
        """
        Args:
            obs:  (B, T, N, C) observation window
            mask: (B, T, N) mask values {0,1,2}
            deterministic: if True, return mu only (inference mode)
        Returns:
            z:    (B, D) latent state
            mu:   (B, D) mean
            logvar: (B, D) log variance (None if deterministic)
        """
        B, T, N, C = obs.shape

        # (B, T, N, C) → (B, T, N, D)
        x = self.norm_in(obs)
        x = self.symbol_proj(x)  # (B, T, N, D)

        # Mask-weighted mean pool over symbols → (B, T, D)
        if mask is not None:
            valid_mask = (mask == 0).float().unsqueeze(-1)  # (B, T, N, 1)
            x = (x * valid_mask).sum(dim=2) / (valid_mask.sum(dim=2) + 1e-8)
        else:
            x = x.mean(dim=2)

        # GRU over time → (B, T, H)
        gru_out, _ = self.gru(x)

        # Take last time step → (B, H)
        last = gru_out[:, -1, :]
        last = self.norm_out(last)

        mu = self.mu_head(last)
        if deterministic:
            return mu, mu, None

        logvar = self.logvar_head(last)
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    def export_inference_weights(self) -> dict:
        """Export weights for C++ inference (strips VAE heads)."""
        return {
            'norm_in.weight': self.norm_in.weight.data,
            'norm_in.bias': self.norm_in.bias.data,
            'symbol_proj.weight': self.symbol_proj.weight.data,
            'symbol_proj.bias': self.symbol_proj.bias.data,
            'gru.weight_ih_l0': self.gru.weight_ih_l0.data,
            'gru.weight_hh_l0': self.gru.weight_hh_l0.data,
            'gru.bias_ih_l0': self.gru.bias_ih_l0.data,
            'gru.bias_hh_l0': self.gru.bias_hh_l0.data,
            'gru.weight_ih_l1': self.gru.weight_ih_l1.data,
            'gru.weight_hh_l1': self.gru.weight_hh_l1.data,
            'gru.bias_ih_l1': self.gru.bias_ih_l1.data,
            'gru.bias_hh_l1': self.gru.bias_hh_l1.data,
            'norm_out.weight': self.norm_out.weight.data,
            'norm_out.bias': self.norm_out.bias.data,
            'output_proj.weight': self.mu_head.weight.data,   # mu head → output_proj
            'output_proj.bias': self.mu_head.bias.data,
        }
