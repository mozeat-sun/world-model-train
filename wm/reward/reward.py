"""RewardModel (PyTorch) — multi-objective reward prediction with anti-hacking."""

import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """5-head reward predictor: Sharpe, Sortino, MaxDD, Turnover, WinRate."""

    def __init__(self, latent_dim=128, max_symbols=100):
        super().__init__()
        input_dim = latent_dim + max_symbols  # z + action = 228
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.head_sharpe   = nn.Linear(64, 1)
        self.head_sortino  = nn.Linear(64, 1)
        self.head_maxdd    = nn.Linear(64, 1)
        self.head_turnover = nn.Linear(64, 1)
        self.head_winrate  = nn.Linear(64, 1)

    def forward(self, z, action=None):
        B = z.shape[0]
        if action is None:
            action = torch.zeros(B, 100, device=z.device)
        x = torch.cat([z, action], dim=-1)
        h = self.trunk(x)
        return {
            'sharpe':   2.0 * torch.tanh(self.head_sharpe(h)).squeeze(-1),
            'sortino':  2.5 * torch.tanh(self.head_sortino(h)).squeeze(-1),
            'maxdd':    torch.sigmoid(self.head_maxdd(h)).squeeze(-1),
            'turnover': torch.sigmoid(self.head_turnover(h)).squeeze(-1),
            'winrate':  torch.sigmoid(self.head_winrate(h)).squeeze(-1),
        }

    def composite_reward(self, preds: dict) -> torch.Tensor:
        """Compute anti-hacking composite reward."""
        s, sr = preds['sharpe'], preds['sortino']
        ws  = 10.0 * torch.clamp(1.0 - torch.abs(s) / 2.0, min=0.0)
        wsr = 5.0  * torch.clamp(1.0 - torch.abs(sr) / 2.5, min=0.0)
        wm  = 2.0  * (1.0 + torch.clamp(0.05 - preds['maxdd'], min=0.0) / 0.05)
        ww  = 0.5  * (1.0 + 3.0 * torch.clamp(0.30 - preds['winrate'], min=0.0) / 0.30)
        return (ws * s + wsr * sr
                - wm * torch.clamp(preds['maxdd'] - 0.15, min=0.0)
                - 0.01 * preds['turnover']
                - ww * torch.clamp(0.50 - preds['winrate'], min=0.0))

    def export_inference_weights(self) -> dict:
        return {
            'trunk1.weight': self.trunk[0].weight.data,
            'trunk1.bias': self.trunk[0].bias.data,
            'trunk2.weight': self.trunk[3].weight.data,
            'trunk2.bias': self.trunk[3].bias.data,
            'head_sharpe.weight': self.head_sharpe.weight.data,
            'head_sharpe.bias': self.head_sharpe.bias.data,
            'head_sortino.weight': self.head_sortino.weight.data,
            'head_sortino.bias': self.head_sortino.bias.data,
            'head_maxdd.weight': self.head_maxdd.weight.data,
            'head_maxdd.bias': self.head_maxdd.bias.data,
            'head_turnover.weight': self.head_turnover.weight.data,
            'head_turnover.bias': self.head_turnover.bias.data,
            'head_winrate.weight': self.head_winrate.weight.data,
            'head_winrate.bias': self.head_winrate.bias.data,
        }
