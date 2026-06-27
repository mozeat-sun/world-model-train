"""Export PyTorch state_dict to .wm binary format."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
from .wm_format import write_wm_weights


def export_encoder(
    encoder,
    output_path: str,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    config: Optional[dict] = None,
):
    """Export MarketEncoder to encoder.wm."""
    weights = encoder.export_inference_weights()
    tensors = [
        ('norm_in.weight',    weights['norm_in.weight'].cpu().numpy()),
        ('norm_in.bias',      weights['norm_in.bias'].cpu().numpy()),
        ('symbol_proj.weight',weights['symbol_proj.weight'].cpu().numpy()),
        ('symbol_proj.bias',  weights['symbol_proj.bias'].cpu().numpy()),
        ('gru_cell0.w_ih',    weights['gru.weight_ih_l0'].cpu().numpy()),
        ('gru_cell0.w_hh',    weights['gru.weight_hh_l0'].cpu().numpy()),
        ('gru_cell0.b_ih',    weights['gru.bias_ih_l0'].cpu().numpy()),
        ('gru_cell0.b_hh',    weights['gru.bias_hh_l0'].cpu().numpy()),
        ('gru_cell1.w_ih',    weights['gru.weight_ih_l1'].cpu().numpy()),
        ('gru_cell1.w_hh',    weights['gru.weight_hh_l1'].cpu().numpy()),
        ('gru_cell1.b_ih',    weights['gru.bias_ih_l1'].cpu().numpy()),
        ('gru_cell1.b_hh',    weights['gru.bias_hh_l1'].cpu().numpy()),
        ('output_proj.weight',weights['output_proj.weight'].cpu().numpy()),
        ('output_proj.bias',  weights['output_proj.bias'].cpu().numpy()),
        ('norm_out.weight',   weights['norm_out.weight'].cpu().numpy()),
        ('norm_out.bias',     weights['norm_out.bias'].cpu().numpy()),
    ]

    cfg = config or {}
    cfg.setdefault('latent_dim', 128)
    cfg.setdefault('n_channels', 0)
    cfg.setdefault('max_symbols', 100)
    cfg.setdefault('window_len', 60)
    cfg.setdefault('n_mask_categories', 3)
    cfg.setdefault('attn_heads', 4)
    cfg.setdefault('gru_layers', 2)
    cfg.setdefault('gru_hidden', 256)

    write_wm_weights(output_path, 'encoder', cfg, tensors, norm_mean, norm_std)


def export_dynamics(
    dynamics,
    output_path: str,
    config: Optional[dict] = None,
):
    """Export MarketDynamics to dynamics.wm."""
    weights = dynamics.export_inference_weights()
    tensors = [
        ('z_proj.weight',     weights['z_proj.weight'].cpu().numpy()),
        ('z_proj.bias',       weights['z_proj.bias'].cpu().numpy()),
        ('a_proj.weight',     weights['a_proj.weight'].cpu().numpy()),
        ('a_proj.bias',       weights['a_proj.bias'].cpu().numpy()),
        ('gru_cell0.w_ih',    weights['gru.weight_ih_l0'].cpu().numpy()),
        ('gru_cell0.w_hh',    weights['gru.weight_hh_l0'].cpu().numpy()),
        ('gru_cell0.b_ih',    weights['gru.bias_ih_l0'].cpu().numpy()),
        ('gru_cell0.b_hh',    weights['gru.bias_hh_l0'].cpu().numpy()),
        ('gru_cell1.w_ih',    weights['gru.weight_ih_l1'].cpu().numpy()),
        ('gru_cell1.w_hh',    weights['gru.weight_hh_l1'].cpu().numpy()),
        ('gru_cell1.b_ih',    weights['gru.bias_ih_l1'].cpu().numpy()),
        ('gru_cell1.b_hh',    weights['gru.bias_hh_l1'].cpu().numpy()),
        ('delta_proj1.weight',weights['delta_proj.0.weight'].cpu().numpy()),
        ('delta_proj1.bias',  weights['delta_proj.0.bias'].cpu().numpy()),
        ('delta_proj2.weight',weights['delta_proj.3.weight'].cpu().numpy()),
        ('delta_proj2.bias',  weights['delta_proj.3.bias'].cpu().numpy()),
        ('learnable_scale',   weights['learnable_scale'].cpu().numpy().reshape(1)),
    ]

    cfg = config or {}
    cfg.setdefault('latent_dim', 128)
    cfg.setdefault('max_symbols', 100)
    cfg.setdefault('hidden_dim', 256)
    cfg.setdefault('n_layers', 2)
    cfg.setdefault('predict_horizon', 20)

    write_wm_weights(output_path, 'dynamics', cfg, tensors)


def export_policy(policy, output_path: str, config: Optional[dict] = None):
    """Export StockSelectionPolicy to planner.wm."""
    weights = policy.export_inference_weights()
    tensors = [
        ('agg_proj.weight', weights['agg_proj.weight'].cpu().numpy()),
        ('agg_proj.bias',   weights['agg_proj.bias'].cpu().numpy()),
        ('hidden1.weight',  weights['hidden.0.weight'].cpu().numpy()),
        ('hidden1.bias',    weights['hidden.0.bias'].cpu().numpy()),
        ('output.weight',   weights['output.weight'].cpu().numpy()),
        ('output.bias',     weights['output.bias'].cpu().numpy()),
    ]
    cfg = config or {}
    cfg.setdefault('latent_dim', 128); cfg.setdefault('max_symbols', 100)
    cfg.setdefault('predict_horizon', 20)
    write_wm_weights(output_path, 'planner', cfg, tensors)


def export_critic(critic, output_path: str, config: Optional[dict] = None):
    """Export ValueCritic to critic.wm."""
    weights = critic.export_inference_weights()
    tensors = [
        ('hidden1.weight', weights['hidden1.weight'].cpu().numpy()),
        ('hidden1.bias',   weights['hidden1.bias'].cpu().numpy()),
        ('hidden2.weight', weights['hidden2.weight'].cpu().numpy()),
        ('hidden2.bias',   weights['hidden2.bias'].cpu().numpy()),
        ('output.weight',  weights['output.weight'].cpu().numpy()),
        ('output.bias',    weights['output.bias'].cpu().numpy()),
    ]
    cfg = config or {}; cfg.setdefault('latent_dim', 128)
    write_wm_weights(output_path, 'critic', cfg, tensors)


def export_reward(reward_model, output_path: str, config: Optional[dict] = None):
    """Export RewardModel to reward.wm."""
    weights = reward_model.export_inference_weights()
    tensors = [
        ('trunk1.weight',        weights['trunk1.weight'].cpu().numpy()),
        ('trunk1.bias',          weights['trunk1.bias'].cpu().numpy()),
        ('trunk2.weight',        weights['trunk2.weight'].cpu().numpy()),
        ('trunk2.bias',          weights['trunk2.bias'].cpu().numpy()),
        ('head_sharpe.weight',   weights['head_sharpe.weight'].cpu().numpy()),
        ('head_sharpe.bias',     weights['head_sharpe.bias'].cpu().numpy()),
        ('head_sortino.weight',  weights['head_sortino.weight'].cpu().numpy()),
        ('head_sortino.bias',    weights['head_sortino.bias'].cpu().numpy()),
        ('head_maxdd.weight',    weights['head_maxdd.weight'].cpu().numpy()),
        ('head_maxdd.bias',      weights['head_maxdd.bias'].cpu().numpy()),
        ('head_turnover.weight', weights['head_turnover.weight'].cpu().numpy()),
        ('head_turnover.bias',   weights['head_turnover.bias'].cpu().numpy()),
        ('head_winrate.weight',  weights['head_winrate.weight'].cpu().numpy()),
        ('head_winrate.bias',    weights['head_winrate.bias'].cpu().numpy()),
    ]
    cfg = config or {}; cfg.setdefault('latent_dim', 128); cfg.setdefault('max_symbols', 100)
    write_wm_weights(output_path, 'reward', cfg, tensors)


def save_checkpoint(encoder, dynamics, policy=None, critic=None, reward=None, path: str = ''):
    """Save full PyTorch checkpoint."""
    ckpt = {'encoder': encoder.state_dict(), 'dynamics': dynamics.state_dict()}
    if policy: ckpt['policy'] = policy.state_dict()
    if critic: ckpt['critic'] = critic.state_dict()
    if reward: ckpt['reward'] = reward.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(encoder, dynamics, path: str, device='cpu'):
    """Load PyTorch checkpoint."""
    ckpt = torch.load(path, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    dynamics.load_state_dict(ckpt['dynamics'])
    return ckpt
