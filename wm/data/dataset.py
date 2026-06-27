"""PyTorch Dataset for world model training."""

import numpy as np
import torch
from torch.utils.data import Dataset


class WorldModelDataset(Dataset):
    """
    Sliding window dataset for world model training.

    Each sample:
        obs_window: (window_len, max_symbols, n_channels)
        mask:       (window_len, max_symbols) uint8 {0,1,2}
        forward_ret:(max_symbols,) 21-day forward return (for diagnostics)
    """

    def __init__(
        self,
        channels: np.ndarray,      # (T, N, C) normalized
        mask: np.ndarray,          # (T, N)
        forward_ret: np.ndarray,   # (T, N) 21-day forward return
        window_len: int = 60,
    ):
        self.channels = channels.astype(np.float32)
        self.mask = mask.astype(np.uint8)
        self.forward_ret = forward_ret.astype(np.float32)
        self.window_len = window_len
        self.n_samples = channels.shape[0] - window_len - 1

        if self.n_samples < 1:
            raise ValueError(
                f"Not enough data: need > {window_len} days, got {channels.shape[0]}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        t_end = idx + self.window_len
        obs = self.channels[idx:t_end]           # (60, N, C)
        m = self.mask[idx:t_end]                 # (60, N)
        ret = self.forward_ret[t_end]            # (N,) 21-day fwd return from t_end
        return (
            torch.from_numpy(obs),
            torch.from_numpy(m).long(),
            torch.from_numpy(ret),
        )


def create_dataloaders(
    channels: np.ndarray,
    mask: np.ndarray,
    forward_ret: np.ndarray,
    window_len: int = 60,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 2,
):
    """Create train/val/test DataLoaders with strict time-order split."""
    T = channels.shape[0]
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    # NOTE: window_len offset for valid samples
    train_ds = WorldModelDataset(
        channels[:train_end], mask[:train_end],
        forward_ret[:train_end], window_len
    )
    val_ds = WorldModelDataset(
        channels[train_end - window_len:val_end],
        mask[train_end - window_len:val_end],
        forward_ret[train_end - window_len:val_end],
        window_len,
    )
    test_ds = WorldModelDataset(
        channels[val_end - window_len:],
        mask[val_end - window_len:],
        forward_ret[val_end - window_len:],
        window_len,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
