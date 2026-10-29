"""
decay_model.py  —  Is a signal that worked still working? And how would we know?

Measurement
-----------
For a signal (a cross-sectional score known at the end of day t) the engine
works on NON-OVERLAPPING periods H days apart. Under no skill the per-period
rank correlations are then independent whatever the signal's own persistence
(the returns of disjoint windows are independent; the score path is just a fixed
set of rankings — checked in testing), with sd ~ 1/sqrt(N-1): about 0.17 for 36
ETFs and 0.41 for 7. Per period it records the rank IC, the Fama-MacBeth slope
of realized on z-scored signal (the "forecast error" view), the top-3 net return
vs the universe, and a dollar-neutral rank-weighted long/short book's net return
(the TRADABLE strategy) with turnover-based costs and short borrow.

Detection
---------
The first REFERENCE_YEARS define the signal's alive level (mean, sd). A one-sided
CUSUM then watches for a downward shift in standardized IC. Because the reference
mean is ESTIMATED from a few hundred noisy numbers, its error is the main source
of false alarms, so the alarm threshold is calibrated by simulating exactly that
procedure to a stated false-alarm probability. A "power" check says how likely a
TOTAL loss of the reference IC would be to be caught within the false-alarm
horizon — which is what limits what the detector can promise.

Honest scope: a few weak months are not decay; a persistent shift is. The detector
cannot separate "the signal died" from "the regime it works in left" using returns
alone — it reports regime-conditional IC as evidence, not as a verdict.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import config


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def forward_returns(returns: np.ndarray, horizon: int) -> np.ndarray:
    """fwd[i] = sum of returns over days i+1 .. i+horizon (NaN where not fully observed)."""
    n, T = returns.shape
    cum = np.concatenate([np.zeros((1, T)), np.cumsum(returns, axis=0)], axis=0)
    fwd = np.full((n, T), np.nan)
    if n - horizon > 0:
        fwd[: n - horizon] = cum[horizon + 1: n + 1] - cum[1: n - horizon + 1]
    return fwd


def _ranks(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a, axis=1), axis=1).astype(float)


def row_spearman(S: np.ndarray, R: np.ndarray) -> np.ndarray:
    rs, rr = _ranks(S), _ranks(R)
    rs -= rs.mean(axis=1, keepdims=True)
    rr -= rr.mean(axis=1, keepdims=True)
    den = np.sqrt((rs ** 2).sum(axis=1) * (rr ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rs * rr).sum(axis=1) / den, np.nan)


def non_overlapping_rows(valid: np.ndarray, horizon: int) -> np.ndarray:
    rows, last = [], -10 ** 9
    for i in np.where(valid)[0]:
        if i - last >= horizon:
            rows.append(i)
            last = i
    return np.array(rows, dtype=int)


def topn_weights(S: np.ndarray, n_top: int) -> np.ndarray:
    m, T = S.shape
    W = np.zeros((m, T))
    idx = np.argsort(-S, axis=1)[:, :n_top]
    np.put_along_axis(W, idx, 1.0 / n_top, axis=1)
    return W


def rank_ls_weights(S: np.ndarray) -> np.ndarray:
    """Weights proportional to centered rank, dollar-neutral: long sum +1, short sum -1."""
    m, T = S.shape
    c = _ranks(S) - (T - 1) / 2.0
    pos = np.where(c > 0, c, 0).sum(axis=1, keepdims=True)
    return c / np.where(pos > 0, pos, 1.0)


def book_returns(W: np.ndarray, R: np.ndarray, horizon: int, long_only: bool, cost_bps: float = None,
                 borrow_bps: float = None) -> Dict[str, np.ndarray]:
    """
    Gross and net per-period returns of weight rows W against forward returns R. Long-only books are
    measured ACTIVE vs the equal-weight universe. Cost = sum|w_t - w_{t-1}| * cost/2 (the sibling
    engines' 'fraction of picks replaced x round trip' convention) + borrow on short notional.
    """
    cost_bps = config.TRADING_COST_BPS if cost_bps is None else cost_bps
    borrow_bps = config.SHORT_BORROW_BPS_ANNUAL if borrow_bps is None else borrow_bps
    ew = R.mean(axis=1)
    gross = (W * R).sum(axis=1) - (ew if long_only else 0.0)
    turn = np.zeros(W.shape[0])
    turn[1:] = np.abs(np.diff(W, axis=0)).sum(axis=1)
    cost = turn * cost_bps / 2e4
    if not long_only:
        cost = cost + np.where(W < 0, -W, 0).sum(axis=1) * borrow_bps / 1e4 * horizon / 252.0
    return {"gross": gross, "net": gross - cost, "turnover": turn, "ew": ew}


def measure_signal(scores: np.ndarray, returns: np.ndarray, horizon: int) -> Optional[Dict]:
    """All per-period series for one signal on non-overlapping dates."""
    fwd = forward_returns(returns, horizon)
    valid = np.isfinite(scores).all(axis=1) & np.isfinite(fwd).all(axis=1)
    rows = non_overlapping_rows(valid, horizon)
    if len(rows) < 30:
        return None
    S, R = scores[rows], fwd[rows]
    ic = row_spearman(S, R)
    z = (S - S.mean(axis=1, keepdims=True)) / np.maximum(S.std(axis=1, keepdims=True), 1e-12)
    fmb = (z * (R - R.mean(axis=1, keepdims=True))).sum(axis=1) / np.maximum((z ** 2).sum(axis=1), 1e-12)
    W3, WL = topn_weights(S, config.TOP_N), rank_ls_weights(S)
    top3 = book_returns(W3, R, horizon, long_only=True)
    ls = book_returns(WL, R, horizon, long_only=False)
    return {"rows": rows, "ic": ic, "fmb": fmb, "top3": top3, "ls": ls, "W_ls": WL, "R": R, "ew": R.mean(axis=1)}


# ---------------------------------------------------------------------------
# Reference, CUSUM, calibration, power
# ---------------------------------------------------------------------------
def reference_stats(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    return {"mu0": mu, "sd0": sd, "n": int(len(x)), "t": float(mu / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0}


def cusum_down(x: np.ndarray, mu0: float, sd0: float, k: float) -> np.ndarray:
    """One-sided CUSUM for a DOWNWARD shift: S_t = max(0, S_{t-1} + (mu0 - x_t)/sd0 - k)."""
    S = np.zeros(len(x))
    s = 0.0
    for t in range(len(x)):
        s = max(0.0, s + (mu0 - x[t]) / sd0 - k)
        S[t] = s
    return S


def default_k(horizon: int) -> float:
    """CUSUM slack for a target annualized IR: half the standardized per-period effect, IR / sqrt(periods per year)."""
    return 0.5 * config.TARGET_IR / np.sqrt(config.PERIODS_PER_YEAR[horizon])


_THRESH_CACHE: Dict[Tuple, float] = {}


def calibrate_threshold(R: int, L: int, k: float, target: float = None, reps: int = 6000, seed: int = 0) -> float:
    """
    h such that, for a signal with NO change, P(CUSUM >= h within L periods) = target — simulating the real
    procedure: reference mean and sd ESTIMATED from R periods, then L monitored periods. Standardized units,
    so the answer does not depend on the signal's own IC or the universe size.
    """
    target = config.TARGET_FALSE_ALARM if target is None else target
    key = (R, L, round(k, 4), round(target, 4))
    if key in _THRESH_CACHE:
        return _THRESH_CACHE[key]
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (reps, R + L))
    mu0, sd0 = X[:, :R].mean(axis=1), X[:, :R].std(axis=1, ddof=1)
    Z = (mu0[:, None] - X[:, R:]) / sd0[:, None] - k
    S = np.zeros(reps); mx = np.zeros(reps)
    for t in range(L):
        S = np.maximum(0.0, S + Z[:, t]); mx = np.maximum(mx, S)
    h = float(np.quantile(mx, 1.0 - target))
    _THRESH_CACHE[key] = h
    return h


def power_total_decay(effect: float, k: float, h: float, R: int, L: int, reps: int = 4000, seed: int = 1) -> Dict[str, float]:
    """
    P(alarm within L periods) and median delay when the signal's mean IC falls to ZERO right after a reference of R
    periods, where `effect` = mu0/sd0 (the reference level in sd units). Simulates the whole procedure INCLUDING
    the estimation error in the reference mean and sd (a version that ignored it said 91% where the full
    simulation says 84%).
    """
    rng = np.random.default_rng(seed)
    ref = rng.normal(effect, 1.0, (reps, R))
    mu0, sd0 = ref.mean(axis=1), ref.std(axis=1, ddof=1)
    post = rng.normal(0.0, 1.0, (reps, L))
    Z = (mu0[:, None] - post) / sd0[:, None] - k
    S = np.zeros(reps); first = np.full(reps, np.nan)
    for t in range(L):
        S = np.maximum(0.0, S + Z[:, t])
        newly = np.isnan(first) & (S >= h)
        first[newly] = t + 1
    caught = np.isfinite(first)
    return {"p_detect": float(caught.mean()), "median_delay": float(np.nanmedian(first)) if caught.any() else None}


# ---------------------------------------------------------------------------
# State machine and diagnostics
# ---------------------------------------------------------------------------
ACTIVE, DEGRADING, SUSPENDED = 0, 1, 2
STATE_NAMES = {ACTIVE: "Active", DEGRADING: "Degrading", SUSPENDED: "Suspended"}


def run_state_machine(ic: np.ndarray, R: int, mu0: float, sd0: float, k: float, h: float, rec_window: int,
                      watch: float = None, rec_frac: float = None) -> Dict[str, np.ndarray]:
    """
    Monitor periods R.. . `state[t]` is the state AFTER observing period t (so it governs period t+1).
    Active < watch*h <= Degrading < h <= Suspended. Suspension persists until the trailing `rec_window` mean IC
    is BOTH at least rec_frac * mu0 AND statistically positive (t >= REINSTATE_T), then the CUSUM restarts from 0.
    """
    watch = config.WATCH_FRACTION if watch is None else watch
    rec_frac = config.REINSTATE_FRACTION if rec_frac is None else rec_frac
    n = len(ic)
    state = np.full(n, -1, dtype=int)
    S = np.full(n, np.nan)
    s, suspended = 0.0, False
    alarms, reinstated = [], []
    for t in range(R, n):
        if suspended:
            w = ic[max(R, t - rec_window + 1): t + 1]
            if len(w) >= rec_window and np.nanmean(w) >= rec_frac * mu0 and \
                    np.nanmean(w) / (np.nanstd(w, ddof=1) / np.sqrt(len(w)) + 1e-12) >= config.REINSTATE_T:
                suspended, s = False, 0.0
                reinstated.append(t)
            S[t] = s
            state[t] = SUSPENDED if suspended else ACTIVE
            continue
        s = max(0.0, s + (mu0 - ic[t]) / sd0 - k)
        S[t] = s
        if s >= h:
            suspended = True
            alarms.append(t)
            state[t] = SUSPENDED
        elif s >= watch * h:
            state[t] = DEGRADING
        else:
            state[t] = ACTIVE
    return {"state": state, "S": S, "alarms": np.array(alarms, dtype=int), "reinstated": np.array(reinstated, dtype=int)}


def ewma(x: np.ndarray, half_life: float) -> np.ndarray:
    lam = 1.0 - 0.5 ** (1.0 / max(half_life, 1.0))
    out = np.full(len(x), np.nan)
    m = None
    for t, v in enumerate(x):
        if not np.isfinite(v):
            out[t] = m if m is not None else np.nan
            continue
        m = v if m is None else lam * v + (1 - lam) * m
        out[t] = m
    return out


def trend_slope(x: np.ndarray, window: int) -> Dict[str, Optional[float]]:
    """OLS slope of x on period index over the last `window` periods, with its standard error."""
    y = x[-window:]
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 20:
        return {"slope": None, "se": None, "n": n}
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    slope = float((tc * (y - y.mean())).sum() / (tc ** 2).sum())
    resid = y - y.mean() - slope * tc
    se = float(np.sqrt((resid ** 2).sum() / (n - 2) / (tc ** 2).sum()))
    return {"slope": slope, "se": se, "n": n}


def market_alpha(net: np.ndarray, ew: np.ndarray) -> Dict[str, Optional[float]]:
    """OLS of net on the equal-weight universe return: beta, per-period alpha, alpha t."""
    n = len(net)
    if n < 20:
        return {"beta": None, "alpha": None, "alpha_t": None}
    x = ew - ew.mean()
    sxx = float((x ** 2).sum())
    if sxx <= 0:
        return {"beta": 0.0, "alpha": float(net.mean()), "alpha_t": None}
    beta = float((x * (net - net.mean())).sum() / sxx)
    alpha = float(net.mean() - beta * ew.mean())
    resid = net - alpha - beta * ew
    se = np.sqrt(float((resid ** 2).sum() / max(n - 2, 1)) * (1.0 / n + ew.mean() ** 2 / sxx))
    return {"beta": beta, "alpha": alpha, "alpha_t": float(alpha / se) if se > 0 else None}


def max_drawdown(r: np.ndarray) -> float:
    eq = np.exp(np.cumsum(r))
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def regime_labels(returns: np.ndarray, rows: np.ndarray) -> Dict[str, np.ndarray]:
    """Regimes known at the decision date: high/low volatility, up/down trend of the equal-weight universe."""
    ew = returns.mean(axis=1)
    vol21 = pd.Series(ew).rolling(21, min_periods=21).std()
    med = vol21.rolling(252, min_periods=126).median()
    ret63 = pd.Series(ew).rolling(63, min_periods=63).sum()
    hv = np.where(np.isfinite(vol21.values) & np.isfinite(med.values), (vol21.values > med.values).astype(float), np.nan)
    up = np.where(np.isfinite(ret63.values), (ret63.values > 0).astype(float), np.nan)
    return {"volatility": hv[rows], "trend": up[rows]}


def regime_table(ic: np.ndarray, labels: Dict[str, np.ndarray], R: int, recent: int) -> List[Dict]:
    """Reference vs recent mean IC inside each regime — evidence about whether decay is concentrated in one."""
    out = []
    n = len(ic)
    names = {"volatility": ("low volatility", "high volatility"), "trend": ("down trend", "up trend")}
    for key, lab in labels.items():
        for val in (0.0, 1.0):
            m = lab == val
            ref = ic[:R][m[:R] & np.isfinite(ic[:R])]
            rec = ic[max(R, n - recent):][m[max(R, n - recent):] & np.isfinite(ic[max(R, n - recent):])]
            out.append({"regime": names[key][int(val)], "n_ref": int(len(ref)), "ic_ref": float(ref.mean()) if len(ref) >= 10 else None,
                        "n_recent": int(len(rec)), "ic_recent": float(rec.mean()) if len(rec) >= 10 else None})
    return out


def block_bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, block: int, reps: int, seed: int = 0) -> Dict:
    """Circular block bootstrap of the PAIRED difference (a - b) in mean and in Sharpe-like ratio."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    m = len(a)
    if m < 3 * block:
        return {}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(m / block))
    idx = ((rng.integers(0, m, size=(reps, nb))[:, :, None] + np.arange(block)[None, None, :]) % m).reshape(reps, -1)[:, :m]
    A, B = a[idx], b[idx]
    d_mean = A.mean(axis=1) - B.mean(axis=1)
    sa = A.mean(axis=1) / (A.std(axis=1, ddof=1) + 1e-12)
    sb = B.mean(axis=1) / (B.std(axis=1, ddof=1) + 1e-12)
    d_sh = sa - sb
    return {"d_mean": float(d_mean.mean()), "d_mean_lo": float(np.percentile(d_mean, 5)), "d_mean_hi": float(np.percentile(d_mean, 95)),
            "p_mean_better": float((d_mean > 0).mean()), "d_sharpe": float(d_sh.mean()), "p_sharpe_better": float((d_sh > 0).mean())}
