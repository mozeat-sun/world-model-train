"""Phase 1: VAE pre-training — Encoder + Decoder."""

import torch
import torch.optim as optim
from ..encoder.encoder import MarketEncoder
from ..renderer.decoder import MarketRenderer
from .losses import vae_loss


def beta_schedule(epoch: int, total_epochs: int,
                  beta_start: float = 0.0, beta_end: float = 0.001) -> float:
    """KL annealing schedule."""
    if epoch < 10:
        return beta_start
    elif epoch < 20:
        frac = (epoch - 10) / 10.0
        return beta_start + frac * (beta_end - beta_start)
    else:
        return beta_end


def train_phase1(
    encoder: MarketEncoder,
    decoder: MarketRenderer,
    train_loader,
    val_loader,
    config,
    device='cpu',
):
    """
    Train VAE (Encoder + Decoder) with KL annealing.

    Returns:
        encoder, decoder, metrics dict
    """
    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.lr, weight_decay=config.weight_decay
    )

    metrics_history = {'epoch': [], 'train_loss': [], 'val_loss': [],
                       'recon_loss': [], 'kl_loss': []}

    for epoch in range(config.phase1_epochs):
        beta = beta_schedule(epoch, config.phase1_epochs,
                            config.phase1_beta_start, config.phase1_beta_end)

        # Training
        encoder.train()
        decoder.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0

        for batch in train_loader:
            obs, mask, _ = batch
            obs = obs.to(device)
            mask = mask.to(device)

            # Forward
            z, mu, logvar = encoder(obs, mask, deterministic=False)
            recon = decoder(z)

            # Loss (mask-aware: only compute recon on valid elements)
            loss, recon_l, kl_l = vae_loss(recon, obs, mu, logvar, beta, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_recon_sum += recon_l.item()
            train_kl_sum += kl_l.item()

        # Validation
        encoder.eval()
        decoder.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs, mask, _ = batch
                obs = obs.to(device)
                mask = mask.to(device)
                z, mu, logvar = encoder(obs, mask, deterministic=True)
                recon = decoder(z)
                # Validation: recon loss only (no KL — KL is regularization not data likelihood)
                _, recon_l, _ = vae_loss(recon, obs, mu, logvar, 0.0, mask)
                val_loss_sum += recon_l.item()

        n_batches = len(train_loader)
        metrics_history['epoch'].append(epoch)
        metrics_history['train_loss'].append(train_loss_sum / n_batches)
        metrics_history['val_loss'].append(val_loss_sum / len(val_loader))
        metrics_history['recon_loss'].append(train_recon_sum / n_batches)
        metrics_history['kl_loss'].append(train_kl_sum / n_batches)

        if epoch % 10 == 0 or epoch == config.phase1_epochs - 1:
            print(f"Phase1 Epoch {epoch:3d}/{config.phase1_epochs} "
                  f"β={beta:.4f} "
                  f"L={train_loss_sum/n_batches:.4f} "
                  f"Recon={train_recon_sum/n_batches:.4f} "
                  f"KL={train_kl_sum/n_batches:.4f} "
                  f"Val={val_loss_sum/len(val_loader):.4f}")

    return encoder, decoder, metrics_history
