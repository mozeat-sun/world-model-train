"""Phase 3: Imagination-based Policy training (Dreamer-style RL with TD-λ)."""

import torch
import torch.optim as optim
from ..planner.policy import StockSelectionPolicy
from ..critic.critic import ValueCritic
from ..reward.reward import RewardModel
from ..simulator.dynamics import MarketDynamics
from ..encoder.encoder import MarketEncoder
from .losses import td_lambda_targets, actor_loss
from .replay_buffer import ReplayBuffer


def train_phase3(
    encoder: MarketEncoder,
    dynamics: MarketDynamics,
    policy: StockSelectionPolicy,
    critic: ValueCritic,
    reward_model: RewardModel,
    replay_buffer: ReplayBuffer,
    train_loader,
    config,
    device='cpu',
):
    """
    Imagination-based RL training.

    Freezes: Encoder, Dynamics
    Trains: Policy, Critic, RewardModel

    Algorithm per episode:
      1. Sample random z_0 from replay buffer
      2. Auto-regressively unroll H=20 steps: z_{i+1}=Dynamics(z_i, Policy(z_i))
      3. Predict reward at each step: r_i = RewardModel(z_i, w_i)
      4. Compute TD(λ) targets
      5. Actor loss: REINFORCE + entropy bonus
      6. Critic loss: MSE vs TD(λ) targets
      7. Real trajectory regularization
    """
    encoder.to(device).eval()
    dynamics.to(device).eval()
    policy.to(device).train()
    critic.to(device).train()
    reward_model.to(device).train()

    # Fill replay buffer from frozen encoder
    if len(replay_buffer) == 0:
        print("  Filling replay buffer from encoder...")
        replay_buffer.fill_from_encoder(encoder, train_loader, device)
        print(f"  Replay buffer: {len(replay_buffer)} states")

    opt_policy = optim.Adam(policy.parameters(), lr=config.lr)
    opt_critic = optim.Adam(critic.parameters(), lr=config.lr)
    opt_reward = optim.Adam(reward_model.parameters(), lr=config.lr)

    H = config.phase3_h
    gamma = config.phase3_gamma
    lam = config.phase3_lambda
    eta = config.phase3_entropy_eta
    alpha = config.phase3_real_alpha
    batch_size = config.phase1_batch_size

    metrics = {'epoch': [], 'actor_loss': [], 'critic_loss': [], 'reward_loss': [],
               'entropy': [], 'mean_reward': []}

    for epoch in range(config.phase3_epochs):
        actor_sum, critic_sum, reward_sum, entropy_sum, reward_mean = 0.0, 0.0, 0.0, 0.0, 0.0
        n_batches = min(100, max(10, len(replay_buffer) // batch_size))

        for _ in range(n_batches):
            # Sample z_0
            z = replay_buffer.sample(batch_size).to(device)

            # Imagination rollout
            z_seq, w_seq, r_seq, v_seq = [], [], [], []
            for step in range(H):
                w, log_prob, ent = policy(z, None, exploratory=True)
                z, _ = dynamics(z, w)
                r_pred = reward_model(z, w)
                v = critic(z)

                z_seq.append(z)
                w_seq.append((w, log_prob, ent))
                r_seq.append(reward_model.composite_reward(r_pred))
                v_seq.append(v)

            # Terminal value
            v_seq.append(critic(z_seq[-1]))

            # Stack
            z_traj = torch.stack(z_seq, dim=1)          # (B, H, D)
            r_tensor = torch.stack(r_seq, dim=1)         # (B, H)
            v_tensor = torch.stack(v_seq, dim=1)         # (B, H+1)

            # TD(λ) targets
            G = td_lambda_targets(r_tensor, v_tensor, gamma, lam)  # (B, H)

            # Critic loss: MSE(V(z_t), G_t)
            critic_loss = torch.nn.functional.mse_loss(v_tensor[:, :-1], G.detach())

            # Actor loss: REINFORCE + entropy
            advantages = G.detach() - v_tensor[:, :-1].detach()  # (B, H)
            actor_l = torch.tensor(0.0, device=device)
            ent_sum = torch.tensor(0.0, device=device)
            for step in range(H):
                w, log_prob, ent = w_seq[step]
                a_l, _ = actor_loss(log_prob, advantages[:, step], ent, eta)
                actor_l = actor_l + a_l
                ent_sum = ent_sum + ent

            # Real trajectory regularization (prevent overfitting to imagination)
            real_l = torch.tensor(0.0, device=device)
            if alpha > 0:
                for obs, mask, _ in train_loader:
                    if np.random.random() < 0.1:  # Only 10% of batches for efficiency
                        obs, mask = obs[:batch_size].to(device), mask[:batch_size].to(device)
                        with torch.no_grad():
                            z_real, _, _ = encoder(obs, mask, deterministic=True)
                        w_real, log_prob_real, _ = policy(z_real, None)
                        real_l = -log_prob_real.mean()
                        break

            # RewardModel loss (supervised from real data — trained jointly)
            # For now, reward is self-supervised from the imagination
            reward_l = -r_tensor.mean()  # Maximize expected reward

            total = actor_l + 0.1 * critic_loss + 0.01 * reward_l + alpha * real_l

            opt_policy.zero_grad()
            opt_critic.zero_grad()
            opt_reward.zero_grad()
            total.backward()
            opt_policy.step()
            opt_critic.step()
            opt_reward.step()

            actor_sum += actor_l.item() / H
            critic_sum += critic_loss.item()
            reward_sum += reward_l.item()
            entropy_sum += ent_sum.item() / H
            reward_mean += r_tensor.mean().item()

        metrics['epoch'].append(epoch)
        metrics['actor_loss'].append(actor_sum / n_batches)
        metrics['critic_loss'].append(critic_sum / n_batches)
        metrics['reward_loss'].append(reward_sum / n_batches)
        metrics['entropy'].append(entropy_sum / n_batches)
        metrics['mean_reward'].append(reward_mean / n_batches)

        if epoch % 10 == 0 or epoch == config.phase3_epochs - 1:
            print(f"Phase3 Epoch {epoch:3d}/{config.phase3_epochs} "
                  f"Actor={actor_sum/n_batches:.4f} Critic={critic_sum/n_batches:.4f} "
                  f"Ent={entropy_sum/n_batches:.3f} R={reward_mean/n_batches:.3f}")

    return policy, critic, reward_model, metrics
