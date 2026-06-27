#!/usr/bin/env python3
"""Generate test MarketSnapshot binary data for C++ backtest."""

import struct
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wm.data.normalization import normalize_ohlcv


def generate_snapshots(n_days=200, n_symbols=20, output_path='test_snapshot.bin'):
    """Generate synthetic MarketSnapshot binary data."""
    np.random.seed(42)
    T, N = n_days, n_symbols

    # Random walk prices
    close = 100.0 * np.exp(np.cumsum(np.random.randn(T, N) * 0.01, axis=0))
    high = close * (1.0 + np.abs(np.random.randn(T, N) * 0.02))
    low = close * (1.0 - np.abs(np.random.randn(T, N) * 0.02))
    volume = np.exp(10 + np.random.randn(T, N))

    # 8-channel normalization
    channels = normalize_ohlcv(close, high, low, volume)  # (T, N, 8)
    # Pad to 11 channels
    C = 11
    obs = np.zeros((T, N, C), dtype=np.float32)
    obs[:, :, :8] = channels

    # z-score normalize
    for c in range(C):
        data = obs[:, :, c].ravel()
        mean = np.mean(data)
        std = np.std(data) + 1e-8
        obs[:, :, c] = (obs[:, :, c] - mean) / std

    # Mask: all valid
    mask = np.zeros((T, N), dtype=np.uint8)

    # Open prices for next-day fill simulation
    open_prices = np.zeros((T, N), dtype=np.float32)
    open_prices[1:] = close[:-1] * (1.0 + np.random.randn(T-1, N) * 0.005)
    open_prices[0] = close[0]

    # Write binary file
    with open(output_path, 'wb') as f:
        # n_days (uint32)
        f.write(struct.pack('<I', T))
        # Snapshots
        for t in range(T):
            # date[16]
            f.write(f"2024-{((t // 20) + 1) % 12 + 1:02d}-{(t % 28) + 1:02d}".ljust(16, '\0')[:16].encode())
            # obs[N][C]
            f.write(obs[t].tobytes())
            # mask[N]
            f.write(mask[t].tobytes())
            # open_prices[N]
            f.write(open_prices[t].tobytes())

    struct_size = 16 + N*C*4 + N + N*4
    expected = 4 + T * struct_size
    actual = os.path.getsize(output_path)
    print(f"Generated {output_path}: {T} days, {N} symbols, {C} channels")
    print(f"  Size: {actual} bytes (expected {expected})")
    print(f"  Struct size: {struct_size} bytes")
    return output_path


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'test_snapshot.bin'
    generate_snapshots(output_path=out)
