#!/usr/bin/env python3
"""Complete training pipeline: Phase 1 → Phase 2 → Export."""

import sys
import os
import json
import numpy as np
import torch

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wm.config import ModelConfig, TrainingConfig
from wm.data.normalization import normalize_ohlcv, compute_norm_params, apply_norm
from wm.data.dataset import create_dataloaders
from wm.encoder.encoder import MarketEncoder
from wm.renderer.decoder import MarketRenderer
from wm.simulator.dynamics import MarketDynamics
from wm.training.phase1_vae import train_phase1
from wm.training.phase2_dynamics import train_phase2
from wm.export.pt2wm import export_encoder, export_dynamics, save_checkpoint


def generate_synthetic_data(n_days=500, n_symbols=20):
    """Generate synthetic OHLCV data for smoke testing."""
    np.random.seed(42)
    T, N = n_days, n_symbols

    # Random walk prices
    close = 100.0 * np.exp(np.cumsum(np.random.randn(T, N) * 0.01, axis=0))
    high = close * (1.0 + np.abs(np.random.randn(T, N) * 0.02))
    low = close * (1.0 - np.abs(np.random.randn(T, N) * 0.02))
    volume = np.exp(10 + np.random.randn(T, N))  # ~ lognormal

    return close, high, low, volume


def main():
    print("=" * 60)
    print("World Model Training Pipeline — MVP")
    print("=" * 60)

    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    train_cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {train_cfg.device}")

    # ---- Step 1: Generate synthetic data ----
    print("\n[1/5] Generating synthetic data...")
    close, high, low, volume = generate_synthetic_data(n_days=500, n_symbols=20)
    T, N = close.shape

    # Normalize
    channels = normalize_ohlcv(close, high, low, volume)  # (T, N, 8)
    # Pad to 11 channels (add zeros for macro)
    channels = np.concatenate([
        channels,
        np.zeros((T, N, 3), dtype=np.float32)
    ], axis=-1)  # (T, N, 11)

    mean, std = compute_norm_params(channels)
    channels_norm = apply_norm(channels, mean, std)

    # Simple mask: all valid
    mask = np.zeros((T, N), dtype=np.uint8)

    # Forward returns (21-day)
    forward_ret = np.zeros((T, N), dtype=np.float32)
    for t in range(T - 21):
        forward_ret[t] = close[t + 21] / (close[t] + 1e-8) - 1.0

    print(f"  Data shape: {channels_norm.shape}, mask: {mask.shape}")
    print(f"  Norm mean range: [{mean.min():.3f}, {mean.max():.3f}]")

    # ---- Step 2: Create dataloaders ----
    print("\n[2/5] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        channels_norm, mask, forward_ret,
        window_len=model_cfg.window_len,
        batch_size=train_cfg.phase1_batch_size,
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- Step 3: Phase 1 VAE training ----
    print("\n[3/5] Phase 1: VAE pre-training...")
    encoder = MarketEncoder(
        n_channels=model_cfg.n_channels,
        max_symbols=N,
        latent_dim=model_cfg.latent_dim,
        gru_hidden=model_cfg.gru_hidden,
        gru_layers=model_cfg.gru_layers,
    )
    decoder = MarketRenderer(
        latent_dim=model_cfg.latent_dim,
        n_channels=model_cfg.n_channels,
        max_symbols=N,
        window_len=model_cfg.window_len,
        gru_hidden=model_cfg.gru_hidden,
    )

    # Override epochs for smoke test
    train_cfg.phase1_epochs = 10
    encoder, decoder, metrics1 = train_phase1(
        encoder, decoder, train_loader, val_loader, train_cfg, train_cfg.device
    )

    # ---- Step 4: Phase 2 Dynamics training ----
    print("\n[4/5] Phase 2: Dynamics training...")
    dynamics = MarketDynamics(
        latent_dim=model_cfg.latent_dim,
        max_symbols=N,
        hidden_dim=model_cfg.gru_hidden,
        n_layers=model_cfg.gru_layers,
    )

    train_cfg.phase2_epochs = 10
    dynamics, metrics2 = train_phase2(
        encoder, dynamics, train_loader, val_loader, train_cfg, train_cfg.device
    )

    # ---- Step 5: Export ----
    print("\n[5/5] Exporting weights...")
    export_dir = train_cfg.export_dir
    os.makedirs(export_dir, exist_ok=True)

    encoder_path = os.path.join(export_dir, 'encoder.wm')
    dynamics_path = os.path.join(export_dir, 'dynamics.wm')
    checkpoint_path = os.path.join(export_dir, '..', 'phase2', 'checkpoint.pt')

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    export_encoder(encoder, encoder_path, mean, std)
    export_dynamics(dynamics, dynamics_path)
    save_checkpoint(encoder, dynamics, checkpoint_path)

    # Save norm params
    with open(os.path.join(export_dir, 'norm_params.json'), 'w') as f:
        json.dump({
            'mean': mean.tolist(),
            'std': std.tolist(),
            'n_channels': int(model_cfg.n_channels),
        }, f, indent=2)

    print(f"\nExported to {export_dir}/")
    print("  - encoder.wm")
    print("  - dynamics.wm")
    print("  - norm_params.json")
    print("\nDone! Training pipeline completed successfully.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
