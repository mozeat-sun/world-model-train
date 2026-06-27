"""Replay buffer of z_0 starting states for Phase 3 imagination training."""

import torch
import numpy as np
from typing import Iterator


class ReplayBuffer:
    """
    Stores pre-computed z_0 starting states from frozen Encoder.
    Phase 3 samples random z_0 to start imagination rollouts.
    """

    def __init__(self, capacity: int = 10000, latent_dim: int = 128):
        self.capacity = capacity
        self.buffer = torch.zeros(capacity, latent_dim)
        self.ptr = 0
        self.full = False

    def add(self, z: torch.Tensor):
        """Add batch of z vectors to buffer. z: (B, D)"""
        B = z.shape[0]
        for i in range(B):
            self.buffer[self.ptr] = z[i].detach().cpu()
            self.ptr = (self.ptr + 1) % self.capacity
            if self.ptr == 0:
                self.full = True

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample random batch of z_0 states."""
        n = self.capacity if self.full else self.ptr
        if n == 0:
            return torch.randn(batch_size, self.buffer.shape[1])
        indices = torch.randint(0, n, (batch_size,))
        return self.buffer[indices]

    def fill_from_encoder(self, encoder, dataloader, device='cpu'):
        """Pre-compute z_0 states from frozen encoder over entire dataset."""
        encoder.eval()
        with torch.no_grad():
            for obs, mask, _ in dataloader:
                obs, mask = obs.to(device), mask.to(device)
                z, _, _ = encoder(obs, mask, deterministic=True)
                self.add(z)

    def __len__(self) -> int:
        return self.capacity if self.full else self.ptr

    def __iter__(self) -> Iterator[torch.Tensor]:
        n = len(self)
        for i in range(n):
            yield self.buffer[i:i+1]
