"""8-channel OHLCV normalization — data transforms, not alpha factors."""

import numpy as np
from typing import Tuple


def sma(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average along first axis (time)."""
    T = x.shape[0]
    result = np.zeros_like(x)
    cumsum = np.cumsum(x, axis=0)
    # First valid SMA value at index (window-1)
    result[window - 1] = cumsum[window - 1] / window
    # Subsequent values using sliding window difference
    if T > window:
        result[window:] = (cumsum[window:] - cumsum[:-window]) / window
    # Expanding window for first (window-1) rows
    for t in range(window - 1):
        result[t] = np.mean(x[:t + 1], axis=0)
    return result


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation along first axis (time)."""
    sma_x = sma(x, window)
    sma_x2 = sma(x ** 2, window)
    var = np.maximum(sma_x2 - sma_x ** 2, 0.0)
    return np.sqrt(var)


def normalize_ohlcv(
    close: np.ndarray,       # (T, N)
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """
    Compute 8 normalized OHLCV channels.

    Args:
        close:  (T, N) adjusted close prices
        high:   (T, N)
        low:    (T, N)
        volume: (T, N)

    Returns:
        channels: (T, N, 8) normalized channels
    """
    T, N = close.shape
    log_ret = np.log(close / (np.roll(close, 1, axis=0) + 1e-8))
    log_ret[0] = 0.0

    # Daily return: (close_t - close_{t-1}) / close_{t-1}  (NO future leak)
    daily_ret = close / (np.roll(close, 1, axis=0) + 1e-8) - 1.0
    daily_ret[0] = 0.0

    channels = np.stack([
        # ch0: Short-term price position
        close / (sma(close, 20) + 1e-8) - 1.0,
        # ch1: Mid-term price position
        close / (sma(close, 60) + 1e-8) - 1.0,
        # ch2: Daily return (backward-looking, no future leak)
        daily_ret,
        # ch3: Weekly trend (5-day backward)
        close / (np.roll(close, 5, axis=0) + 1e-8) - 1.0,
        # ch4: Monthly trend (21-day backward)
        close / (np.roll(close, 21, axis=0) + 1e-8) - 1.0,
        # ch5: Annualized volatility
        rolling_std(log_ret, 20) * np.sqrt(252),
        # ch6: Intraday range
        (high - low) / (close + 1e-8),
        # ch7: Relative volume
        volume / (sma(volume, 20) + 1e-8),
    ], axis=-1)  # (T, N, 8)

    # Clip extreme values
    channels[..., 2] = np.clip(channels[..., 2], -0.15, 0.15)
    channels[..., 3] = np.clip(channels[..., 3], -0.30, 0.30)
    channels[..., 4] = np.clip(channels[..., 4], -0.50, 0.50)
    channels[..., 7] = np.clip(channels[..., 7], 0.0, 10.0)

    # Mask roll-wrap garbage: first N rows of N-day roll contain wrapped values
    channels[:5, :, 3] = 0.0    # ch3: first 5 rows invalid (5-day roll)
    channels[:21, :, 4] = 0.0   # ch4: first 21 rows invalid (21-day roll)

    return channels


def compute_norm_params(channels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std for z-score normalization."""
    # (T, N, C) → compute over (T, N) for each C
    C = channels.shape[-1]
    mean = np.zeros(C)
    std = np.ones(C)
    for c in range(C):
        data = channels[..., c].ravel()
        mean[c] = np.mean(data)
        std[c] = np.std(data) + 1e-8
    return mean, std


def apply_norm(channels: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score normalization."""
    return (channels - mean) / std
