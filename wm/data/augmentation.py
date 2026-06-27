"""Data augmentation methods to prevent overfitting to specific market periods."""

import numpy as np
from typing import Tuple


def sector_rotation(channels: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly swap two sector blocks of symbols."""
    T, N, C = channels.shape
    if N < 10: return channels, mask
    mid = N // 2
    split = np.random.randint(2, min(mid - 2, N - mid - 2))
    a_start, b_start = np.random.randint(0, split), np.random.randint(mid, mid + split)
    length = np.random.randint(2, min(5, min(split - a_start, N - b_start)))
    result = channels.copy()
    result_mask = mask.copy()
    result[:, a_start:a_start+length] = channels[:, b_start:b_start+length]
    result[:, b_start:b_start+length] = channels[:, a_start:a_start+length]
    result_mask[:, a_start:a_start+length] = mask[:, b_start:b_start+length]
    result_mask[:, b_start:b_start+length] = mask[:, a_start:a_start+length]
    return result, result_mask


def noise_injection(channels: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Add small Gaussian noise to prevent overfitting to price precision."""
    noise = np.random.randn(*channels.shape).astype(np.float32) * sigma
    return channels + noise


def time_warping(channels: np.ndarray, mask: np.ndarray, max_stretch: float = 0.2
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly stretch or compress the time axis by up to ±20%."""
    T, N, C = channels.shape
    factor = 1.0 + (np.random.random() - 0.5) * 2 * max_stretch
    new_T = max(10, int(T * factor))
    result = np.zeros((new_T, N, C), dtype=np.float32)
    result_mask = np.zeros((new_T, N), dtype=np.uint8)
    for n in range(N):
        for c in range(C):
            result[:, n, c] = np.interp(
                np.linspace(0, T - 1, new_T), np.arange(T), channels[:, n, c])
        result_mask[:, n] = np.interp(
            np.linspace(0, T - 1, new_T), np.arange(T), mask[:, n].astype(float)
        ).round().astype(np.uint8)
    return result, result_mask


def bootstrap_resample(channels: np.ndarray, mask: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly resample symbols with replacement to learn cross-sectional structure."""
    T, N, C = channels.shape
    indices = np.random.choice(N, size=N, replace=True)
    return channels[:, indices, :], mask[:, indices]


def regime_injection(channels: np.ndarray, mask: np.ndarray, segment_len: int = 20
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Insert a random segment from elsewhere in the sequence."""
    T = channels.shape[0]
    if T < segment_len * 3: return channels, mask
    src_start = np.random.randint(0, T - segment_len)
    dst_start = np.random.randint(0, T - segment_len)
    result = channels.copy()
    result_mask = mask.copy()
    result[dst_start:dst_start+segment_len] = channels[src_start:src_start+segment_len]
    result_mask[dst_start:dst_start+segment_len] = mask[src_start:src_start+segment_len]
    return result, result_mask


AUGMENTATIONS = [sector_rotation, noise_injection, time_warping,
                 bootstrap_resample, regime_injection]


def apply_random_augmentation(channels: np.ndarray, mask: np.ndarray, prob: float = 0.5
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a random augmentation with given probability."""
    if np.random.random() > prob:
        return channels, mask
    aug = np.random.choice(AUGMENTATIONS)
    return aug(channels, mask)
