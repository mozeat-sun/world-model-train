"""3-class mask generation: VALID(0), SUSPEND(1), DELIST(2)."""

import numpy as np


def generate_mask(df, symbols, volume_col='volume', close_col='close',
                  date_col='date', symbol_col='symbol') -> np.ndarray:
    """
    Generate 3-class mask from OHLCV DataFrame.

    Args:
        df: DataFrame with columns [date, symbol, open, high, low, close, volume]
    Returns:
        mask: (T, N) uint8 array, values {0=VALID, 1=SUSPEND, 2=DELIST}
    """
    T = df[date_col].nunique()
    N = len(symbols)
    mask = np.zeros((T, N), dtype=np.uint8)
    dates = sorted(df[date_col].unique())

    for i, sym in enumerate(symbols):
        sym_data = df[df[symbol_col] == sym].set_index(date_col)
        sym_dates = set(sym_data.index)
        for t, d in enumerate(dates):
            if d not in sym_dates:
                # Check if delisted (past last appearance) or not yet listed
                first_date = sym_data.index.min() if len(sym_data) > 0 else dates[-1]
                last_date = sym_data.index.max() if len(sym_data) > 0 else dates[0]
                if d > last_date:
                    mask[t, i] = 2  # DELIST — past last appearance
                elif d < first_date:
                    mask[t, i] = 2  # DELIST/NOT-LISTED — before first appearance
            else:
                # Check for suspension: zero volume but stock exists
                row = sym_data.loc[d]
                vol = row[volume_col] if hasattr(row, volume_col) else row.get(volume_col, 1)
                if vol == 0 or np.isnan(vol):
                    mask[t, i] = 1  # SUSPEND — no trading today

    return mask


def mask_to_string(mask_val: int) -> str:
    """Convert mask value to human-readable string."""
    return {0: 'VALID', 1: 'SUSPEND', 2: 'DELIST'}.get(mask_val, 'UNKNOWN')
