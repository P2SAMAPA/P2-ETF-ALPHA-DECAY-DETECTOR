"""
signals.py  —  A small library of research signals to monitor.

The detector is signal-agnostic. These five are PROBES with signs fixed in advance from the
standard literature, not claims that they work: they give the monitor signals with a long,
known history, which is what is needed to see how the machinery behaves on real data.
Plug in your own by adding a function that maps (returns, log_prices) -> an (n, T) score
array (higher = expected to outperform), or score picks the engines already archive with
engine_ledger.py.
"""

import numpy as np
import pandas as pd


def _tsum(a, w):
    return pd.DataFrame(a).rolling(w, min_periods=w).sum().values


def mom_21(returns, log_prices):
    """Short-term momentum: the past 21 days' return."""
    return _tsum(returns, 21)


def mom_126_skip(returns, log_prices):
    """Intermediate momentum: days t-126 .. t-22 (skips the most recent month, the usual 12-1 convention scaled down)."""
    return _tsum(returns, 126) - _tsum(returns, 21)


def rev_5(returns, log_prices):
    """Short-term reversal: minus the past 5 days' return."""
    return -_tsum(returns, 5)


def lowvol_63(returns, log_prices):
    """Low-volatility anomaly: minus the 63-day volatility."""
    return -pd.DataFrame(returns).rolling(63, min_periods=63).std().values


def trend_dev(returns, log_prices):
    """Mean reversion to trend: minus the deviation of log price from its 126-day EMA."""
    ema = pd.DataFrame(log_prices).ewm(span=126, adjust=False).mean().values
    x = -(log_prices - ema)
    x[:126] = np.nan
    return x


SIGNALS = {
    "mom_21": (mom_21, "21-day momentum"),
    "mom_126_skip": (mom_126_skip, "6-1 month momentum"),
    "rev_5": (rev_5, "5-day reversal"),
    "lowvol_63": (lowvol_63, "low volatility (63d)"),
    "trend_dev": (trend_dev, "reversion to the 126d trend"),
}
