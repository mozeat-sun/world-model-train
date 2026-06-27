"""Phase 2: Dynamics training — Encoder frozen, train Simulator."""

import torch
import torch.optim as optim
from ..encoder.encoder import MarketEncoder
from ..simulator.dynamics import MarketDynamics
from .losses import dynamics_loss


def train_phase2(
    encoder: MarketEncoder,
    dynamics: MarketDynamics,
    train_loader,
    val_loader,
    config,
    device='cpu',
):
    """
    Train Dynamics with teacher forcing.
    Encoder is frozen, only Dynamics is trained.

    Returns:
        dynamics, metrics dict
    """
    encoder.to(device)
    encoder.eval()  # Frozen
    dynamics.to(device)

    optimizer = optim.Adam(dynamics.parameters(), lr=config.lr)

    metrics_history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(config.phase2_epochs):
        dynamics.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            obs, mask, _ = batch
            obs = obs.to(device)
            mask = mask.to(device)
            B, T, N, C = obs.shape

            # Encode full sequence (frozen encoder)
            with torch.no_grad():
                z_seq = []
                for t in range(T):
                    o_t = obs[:, t:t+1, :, :]  # (B, 1, N, C)
                    m_t = mask[:, t:t+1, :]     # (B, 1, N)
                    z_t, _, _ = encoder(o_t, m_t, deterministic=True)
                    z_seq.append(z_t)
                z_seq = torch.stack(z_seq, dim=1)  # (B, T, D)

            # Train Dynamics: predict z_{t+1} from z_t
            z_current = z_seq[:, :-1, :]   # (B, T-1, D)
            z_true_next = z_seq[:, 1:, :]  # (B, T-1, D)

            z_pred_list = []
            z = z_seq[:, 0, :]  # (B, D) — first state
            for t in range(T - 1):
                z, _ = dynamics(z)  # teacher forcing with true state
                z_pred_list.append(z)
            z_pred = torch.stack(z_pred_list, dim=1)  # (B, T-1, D)

            # Compute deltas
            delta_pred = z_pred - z_current
            delta_true = z_true_next - z_current

            loss = dynamics_loss(z_pred, z_true_next, delta_pred, delta_true,
                                config.phase2_dir_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        # Validation
        dynamics.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs, mask, _ = batch
                obs = obs.to(device)
                mask = mask.to(device)
                B, T = obs.shape[0], obs.shape[1]

                z_seq = []
                for t in range(T):
                    z_t, _, _ = encoder(obs[:, t:t+1], mask[:, t:t+1], deterministic=True)
                    z_seq.append(z_t)
                z_seq = torch.stack(z_seq, dim=1)

                z_current = z_seq[:, :-1, :]
                z_true_next = z_seq[:, 1:, :]

                z_pred_list = []
                z = z_seq[:, 0, :]
                for t in range(T - 1):
                    z, _ = dynamics(z)
                    z_pred_list.append(z)
                z_pred = torch.stack(z_pred_list, dim=1)

                delta_pred = z_pred - z_current
                delta_true = z_true_next - z_current
                loss = dynamics_loss(z_pred, z_true_next, delta_pred, delta_true,
                                    config.phase2_dir_weight)
                val_loss_sum += loss.item()

        n_batches = len(train_loader)
        metrics_history['epoch'].append(epoch)
        metrics_history['train_loss'].append(train_loss_sum / n_batches)
        metrics_history['val_loss'].append(val_loss_sum / len(val_loader))

        if epoch % 10 == 0 or epoch == config.phase2_epochs - 1:
            print(f"Phase2 Epoch {epoch:3d}/{config.phase2_epochs} "
                  f"L={train_loss_sum/n_batches:.6f} "
                  f"Val={val_loss_sum/len(val_loader):.6f}")

    return dynamics, metrics_history
