"""Loss functions for all training phases."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, beta=0.001, mask=None):
    """
    Phase 1 VAE loss: reconstruction + KL divergence.

    Args:
        recon_x: reconstructed observation (B, T, N, C)
        x:       original observation (B, T, N, C)
        mu:      mean from encoder
        logvar:  log variance from encoder
        beta:    KL weight (annealed from 0)
        mask:    (B, T, N) mask, 0=valid. If None, all elements used.
    Returns:
        total_loss, recon_loss, kl_loss
    """
    if mask is not None:
        valid = (mask == 0).float().unsqueeze(-1)  # (B, T, N, 1)
        n_valid = valid.sum() + 1e-8
        recon_loss = (valid * (recon_x - x) ** 2).sum() / n_valid
    else:
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def dynamics_loss(z_pred, z_true, delta_pred, delta_true, dir_weight=0.1):
    """
    Phase 2 Dynamics loss: MSE + direction alignment.

    Args:
        z_pred:       predicted next states (B, H, D)
        z_true:       true next states (B, H, D)
        delta_pred:   predicted deltas (B, H, D)
        delta_true:   true deltas (B, H, D)
        dir_weight:   weight for direction alignment term
    Returns:
        total_loss
    """
    mse = F.mse_loss(z_pred, z_true, reduction='none').mean(dim=-1)  # (B, H)

    # Direction alignment: 1 - cos(delta_pred, delta_true)
    cos_sim = F.cosine_similarity(
        delta_pred.reshape(-1, delta_pred.shape[-1]),
        delta_true.reshape(-1, delta_true.shape[-1]),
        dim=-1
    ).reshape(delta_pred.shape[:2])  # (B, H)
    dir_loss = (1.0 - cos_sim)

    # Weight changepoint samples higher (|delta| > 0.3)
    delta_norm = delta_true.norm(dim=-1)  # (B, H)
    sample_weight = 1.0 + 2.0 * (delta_norm > 0.3).float()

    total = ((mse + dir_weight * dir_loss) * sample_weight).mean()
    return total


def td_lambda_targets(rewards, values, gamma=0.99, lambda_=0.95):
    """
    Compute TD(λ) targets.

    Args:
        rewards: (B, H) predicted rewards
        values:  (B, H+1) state values (last is terminal bootstrap)
        gamma:   discount factor
        lambda_: TD-lambda decay
    Returns:
        targets: (B, H) TD-lambda targets
    """
    B, H = rewards.shape
    targets = torch.zeros(B, H, device=rewards.device)
    g = values[:, -1]  # terminal bootstrap

    for t in reversed(range(H)):
        g = rewards[:, t] + gamma * ((1 - lambda_) * values[:, t + 1] + lambda_ * g)
        targets[:, t] = g

    return targets


def actor_loss(log_probs, advantages, entropy, eta=0.01):
    """REINFORCE + entropy bonus."""
    pg_loss = -(log_probs * advantages.detach()).mean()
    entropy_bonus = -eta * entropy
    return pg_loss + entropy_bonus
