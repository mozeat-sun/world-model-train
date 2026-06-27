"""OHLCVDataPipeline: raw OHLCV → normalized sliding windows."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .normalization import normalize_ohlcv, compute_norm_params, apply_norm
from .mask import generate_mask


class OHLCVDataPipeline:
    """Full data pipeline from raw OHLCV CSV to training-ready PyTorch dataset."""

    def __init__(self, window_len: int = 60, forward_horizon: int = 21,
                 max_symbols: int = 100, n_channels: int = 8):
        self.window_len = window_len
        self.forward_horizon = forward_horizon
        self.max_symbols = max_symbols
        self.n_channels = n_channels

    def load_raw(self, csv_path: str) -> pd.DataFrame:
        """Load raw OHLCV CSV. Expected: date,symbol,open,high,low,close,adj_close,volume."""
        df = pd.read_csv(csv_path, parse_dates=['date'])
        if 'adj_close' not in df.columns:
            df['adj_close'] = df['close']
        return df

    def load_macro(self, csv_path: str) -> pd.DataFrame:
        """Load macro indicators CSV. Expected: date,fed_rate,vix,term_spread."""
        return pd.read_csv(csv_path, parse_dates=['date'])

    def process(self, df: pd.DataFrame, macro_df: Optional[pd.DataFrame] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process raw data into (channels, mask, forward_ret, norm_mean, norm_std).

        Returns:
            channels: (T, N, C) normalized channels
            mask: (T, N) 3-class mask
            forward_ret: (T, N) 21-day forward returns
            norm_mean: (C,) channel means
            norm_std: (C,) channel stds
        """
        dates = sorted(df['date'].unique())
        T = len(dates)

        # Select top symbols by volume
        symbols = (df.groupby('symbol')['volume'].mean()
                   .sort_values(ascending=False).head(self.max_symbols).index.tolist())
        N = min(len(symbols), self.max_symbols)

        # Pivot data
        close = np.zeros((T, N), dtype=np.float64)
        high = np.zeros((T, N), dtype=np.float64)
        low = np.zeros((T, N), dtype=np.float64)
        volume = np.zeros((T, N), dtype=np.float64)

        for i, sym in enumerate(symbols[:N]):
            sym_data = df[df['symbol'] == sym].set_index('date')
            for t, d in enumerate(dates):
                if d in sym_data.index:
                    row = sym_data.loc[d]
                    close[t, i] = row['adj_close'] if 'adj_close' in row else row['close']
                    high[t, i] = row['high']
                    low[t, i] = row['low']
                    volume[t, i] = row['volume']

        # 8-channel normalization
        channels_8 = normalize_ohlcv(close, high, low, volume)  # (T, N, 8)

        # Macro channels (if provided)
        if macro_df is not None:
            macro_df = macro_df.set_index('date')
            macro_channels = np.zeros((T, N, 3), dtype=np.float32)
            for t, d in enumerate(dates):
                if d in macro_df.index:
                    row = macro_df.loc[d]
                    macro_channels[t, :, 0] = row.get('fed_rate', 0)
                    macro_channels[t, :, 1] = row.get('vix', 0)
                    macro_channels[t, :, 2] = row.get('term_spread', 0)
            channels = np.concatenate([channels_8, macro_channels], axis=-1)
        else:
            # Pad with zeros for macro channels
            channels = np.concatenate([
                channels_8,
                np.zeros((T, N, 3), dtype=np.float32)
            ], axis=-1)

        # Normalize
        channels = channels.astype(np.float32)
        norm_mean, norm_std = compute_norm_params(channels)
        channels_norm = apply_norm(channels, norm_mean, norm_std)

        # 3-class mask
        mask = generate_mask(df, symbols[:N])

        # Forward returns (21-day)
        forward_ret = np.zeros((T, N), dtype=np.float32)
        for t in range(T - self.forward_horizon):
            forward_ret[t] = (close[t + self.forward_horizon]
                              / (close[t] + 1e-8) - 1.0).astype(np.float32)

        return channels_norm, mask, forward_ret, norm_mean, norm_std
