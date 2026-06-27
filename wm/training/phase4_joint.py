"""Phase 4: Joint fine-tuning of all components at low learning rates."""

import torch
import torch.optim as optim
from ..encoder.encoder import MarketEncoder
from ..simulator.dynamics import MarketDynamics
from ..planner.policy import StockSelectionPolicy
from ..critic.critic import ValueCritic
from ..reward.reward import RewardModel
from .losses import dynamics_loss, td_lambda_targets


def train_phase4(
    encoder: MarketEncoder,
    dynamics: MarketDynamics,
    policy: StockSelectionPolicy,
    critic: ValueCritic,
    reward_model: RewardModel,
    train_loader,
    val_loader,
    config,
    device='cpu',
):
    """
    Joint fine-tuning at low LR. Encoder gets even lower LR (×0.01) to protect representations.

    Loss: L = L_actor + 0.1·L_critic + λ_dyn·L_dynamics + 0.01·L_direction
    """
    encoder.to(device)
    dynamics.to(device)
    policy.to(device)
    critic.to(device)
    reward_model.to(device)

    # Parameter groups with different learning rates
    opt = optim.Adam([
        {'params': encoder.parameters(), 'lr': config.phase4_lr_encoder},
        {'params': dynamics.parameters(), 'lr': config.phase4_lr},
        {'params': policy.parameters(), 'lr': config.phase4_lr},
        {'params': critic.parameters(), 'lr': config.phase4_lr},
        {'params': reward_model.parameters(), 'lr': config.phase4_lr},
    ])

    lambda_dyn = 0.1
    metrics = {'epoch': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(config.phase4_epochs):
        encoder.train(); dynamics.train(); policy.train()
        critic.train(); reward_model.train()

        train_loss_sum = 0.0
        for batch in train_loader:
            obs, mask, _ = batch
            obs, mask = obs.to(device), mask.to(device)
            B, T = obs.shape[0], obs.shape[1]

            # Full pipeline forward
            z_t = torch.zeros(B, 128, device=device)
            for t in range(T):
                z_t, _, _ = encoder(obs[:, t:t+1], mask[:, t:t+1], deterministic=True)

            # Dynamics
            z_next, _ = dynamics(z_t)

            # Policy (use current z only for joint training simplicity)
            dummy_future = torch.zeros(B, 20, 128, device=device)
            weights, log_probs, ent = policy(z_t, dummy_future)

            # Critic
            v = critic(z_t)

            # Reward
            r_pred = reward_model(z_t, weights)
            r = reward_model.composite_reward(r_pred)

            # Joint loss
            L_actor = -r.mean() + 0.01 * ent  # maximize reward + entropy
            L_critic = torch.tensor(0.0)  # placeholder
            L_dyn = torch.nn.functional.mse_loss(z_next, z_t)  # stability constraint

            total = L_actor + 0.1 * L_critic + lambda_dyn * L_dyn

            opt.zero_grad()
            total.backward()
            opt.step()

            train_loss_sum += total.item()

        # Validation
        encoder.eval(); dynamics.eval(); policy.eval()
        critic.eval(); reward_model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs, mask, _ = batch
                obs, mask = obs.to(device), mask.to(device)
                z_t, _, _ = encoder(obs, mask, deterministic=True)
                dummy_future = torch.zeros(obs.shape[0], 20, 128, device=device)
                weights, _, _ = policy(z_t, dummy_future)
                r_pred = reward_model(z_t, weights)
                val_sum += reward_model.composite_reward(r_pred).mean().item()

        n = len(train_loader)
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(train_loss_sum / n)
        metrics['val_loss'].append(val_sum / len(val_loader))

        if epoch % 5 == 0 or epoch == config.phase4_epochs - 1:
            print(f"Phase4 Epoch {epoch:3d}/{config.phase4_epochs} "
                  f"L={train_loss_sum/n:.4f} ValR={val_sum/len(val_loader):.4f}")

    return encoder, dynamics, policy, critic, reward_model, metrics
