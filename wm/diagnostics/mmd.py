"""Maximum Mean Discrepancy (MMD) for distribution shift detection."""

import torch
import numpy as np


def gaussian_kernel_mmd(x: np.ndarray, y: np.ndarray, sigma: float = None) -> float:
    """
    Compute Gaussian Kernel MMD between two sample sets.

    Args:
        x: (N, D) first sample set
        y: (M, D) second sample set
        sigma: kernel bandwidth (auto if None)
    Returns:
        mmd: scalar MMD value
    """
    if sigma is None:
        # Median pairwise distance heuristic
        xy = np.concatenate([x[:1000], y[:1000]], axis=0)
        dists = np.sum(xy**2, axis=1)[:, None] + np.sum(xy**2, axis=1)[None] - 2 * xy @ xy.T
        sigma = np.median(dists[dists > 0]) ** 0.5
        sigma = max(sigma, 0.1)

    gamma = 1.0 / (2.0 * sigma**2)

    def kernel(a, b):
        a2 = np.sum(a**2, axis=1)[:, None]
        b2 = np.sum(b**2, axis=1)[None]
        dist = a2 + b2 - 2 * a @ b.T
        return np.exp(-gamma * np.clip(dist, 0, None))

    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)

    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return float(max(0.0, mmd))


def detect_distribution_shift(z_recent: np.ndarray, z_train: np.ndarray,
                               threshold: float = 0.15) -> tuple:
    """Detect if recent z distribution has shifted from training distribution."""
    mmd = gaussian_kernel_mmd(z_recent, z_train)
    shifted = mmd > threshold
    return mmd, shifted
