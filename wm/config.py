"""Global configuration for the world model training pipeline.

Loads canonical constants from spec/spec/constants.json.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Locate spec constants relative to this file (wm/config.py → spec/spec/constants.json)
_SPEC_PATH = Path(__file__).parent.parent / "spec" / "spec" / "constants.json"
try:
    with open(_SPEC_PATH) as f:
        _SPEC = json.load(f)
    _D = _SPEC["dimensions"]
except (FileNotFoundError, KeyError):
    # Fallback for when spec submodule is not available
    _D = {
        "latent_dim": 128, "max_symbols": 100, "n_channels": 11,
        "n_mask_categories": 3, "window_len": 60, "gru_hidden": 256,
        "gru_layers": 2, "attn_heads": 4, "predict_horizon": 20, "top_n": 30,
    }


@dataclass
class ModelConfig:
    """Model architecture dimensions. Defaults loaded from spec/constants.json."""
    latent_dim: int = _D["latent_dim"]
    max_symbols: int = _D["max_symbols"]
    n_channels: int = _D["n_channels"]           # 8 OHLCV + 3 macro
    n_mask_categories: int = _D.get("mask_categories", _D.get("n_mask_categories", 3))
    window_len: int = _D["window_len"]
    gru_hidden: int = _D["gru_hidden"]
    gru_layers: int = _D["gru_layers"]
    attn_heads: int = _D["attn_heads"]
    predict_horizon: int = _D["predict_horizon"]
    top_n: int = _D["top_n"]


@dataclass
class TrainingConfig:
    """Training hyperparameters for all 4 phases."""
    # Data
    data_dir: str = "data/raw"
    macro_path: Optional[str] = "data/raw/macro.csv"

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # Phase 1: VAE
    phase1_epochs: int = 100
    phase1_beta_start: float = 0.0
    phase1_beta_end: float = 0.001
    phase1_batch_size: int = 32

    # Phase 2: Dynamics
    phase2_epochs: int = 80
    phase2_rollout_steps: int = 20
    phase2_dir_weight: float = 0.1
    phase2_batch_size: int = 32

    # Phase 3: Policy (not in MVP)
    phase3_epochs: int = 200
    phase3_h: int = 20
    phase3_gamma: float = 0.99
    phase3_lambda: float = 0.95
    phase3_entropy_eta: float = 0.01
    phase3_real_alpha: float = 0.1

    # Phase 4: Joint fine-tuning
    phase4_epochs: int = 20
    phase4_lr: float = 1e-4
    phase4_lr_encoder: float = 1e-5

    # Hardware
    device: str = "cpu"
    num_workers: int = 2

    # Checkpoint
    checkpoint_dir: str = "data/checkpoints"
    export_dir: str = "data/checkpoints/exported"


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    window_len: int = 60
    forward_horizon: int = 21
    max_symbols: int = 100
    n_channels: int = 8           # OHLCV only (macro added separately)
    train_ratio: float = 0.70
    val_ratio: float = 0.15


# Default instances
model_cfg    = ModelConfig()
train_cfg    = TrainingConfig()
data_cfg     = DataConfig()
