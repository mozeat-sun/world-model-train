# world-model-train

Python training pipeline for the World Model trading system. Trains encoder, dynamics, planner, critic, and reward models, then exports them to `.wm` binary format.

## Prerequisites

- Python >= 3.10
- PyTorch >= 2.0

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Initialize spec submodule
git submodule update --init --recursive

# Run full training pipeline
python scripts/train_all.py
```

## Training Phases

| Phase | Name | Description |
|-------|------|-------------|
| 1 | VAE Pre-training | Self-supervised encoder + decoder with KL annealing |
| 2 | Dynamics | Teacher-forced next-state prediction with direction loss |
| 3 | RL (Dreamer-style) | Imagination-based policy, critic, and reward model training |
| 4 | Joint Fine-tuning | Low-LR joint optimization of all components |

## Output

Training produces in `data/checkpoints/exported/`:
- `encoder.wm` — Encoder weights for C++ inference
- `dynamics.wm` — Dynamics weights for C++ inference
- `norm_params.json` — Input normalization parameters

## Spec

This repo references [world-model-spec](https://github.com/...) as a git submodule for canonical constants and the `.wm` format specification.
