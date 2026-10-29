"""
trainer.py  —  P2 ALPHA-DECAY-DETECTOR

For each universe, each research signal and each horizon (daily, weekly):
  * measure it on non-overlapping periods: rank IC, Fama-MacBeth slope, top-3 net
    spread, and a tradable dollar-neutral long/short book net of costs;
  * set the reference from the first 3 years; decide whether there is anything
    demonstrated to decay ("Unvalidated" if the reference IC is not significant);
  * run a calibrated CUSUM -> Active / Degrading / Suspended, with a power check
    for how likely a total loss would be to be caught;
  * report evidence for temporary vs structural (regime-conditional IC, cost
    erosion, slope of the IC, alpha vs market) — as evidence, not a verdict;
  * backtest the control layer on the long/short book: ungated vs gated by the
    detector vs a naive trailing filter, with a paired block bootstrap.
It also scores the picks other engines have archived (engine_ledger.py) and ships
the simulation of what the detector can and cannot see (validation.py).
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import decay_model as dm
import signals as sg
import engine_ledger as el
import validation as val
from data_manager import load_master_data, validate_data

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _clean(d):
    if isinstance(d, dict):
        return {str(k): _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, (float, np.floating)):
        return _f(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.bool_,)):
        return bool(d)
    return d


def perf(net: np.ndarray, expo: np.ndarray, ppy: float) -> Dict:
    net = net[np.isfinite(net)]
    if len(net) < 10:
        return {}
    sd = net.std(ddof=1)
    return {"n": int(len(net)), "mean_pct_per_year": float(net.mean() * ppy * 100),
            "sharpe": float(net.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0,
            "max_drawdown": dm.max_drawdown(net), "avg_exposure": float(np.mean(expo))}


def evaluate_monitor(m: Dict, H: int, N: int, regimes: Dict[str, np.ndarray], dates: List[str]) -> Dict:
    ppy = config.PERIODS_PER_YEAR[H]
    R, L = int(config.REFERENCE_YEARS * ppy), int(config.FA_YEARS * ppy)
    ic = m["ic"]
    n = len(ic)
    out = {"horizon": H, "n_periods": int(n), "periods_per_year": ppy, "reference_periods": R}
    if n < R + int(0.5 * ppy):
        out.update({"status": "Insufficient history", "validated": False})
        return out
    ref = dm.reference_stats(ic[:R])
    validated = ref["t"] >= config.VALIDATION_T
    k = dm.default_k(H)
    h = dm.calibrate_threshold(R, L, k)
    rec_w = max(int(config.REINSTATE_YEARS * ppy), 5)
    effect = max(ref["mu0"], 0.0) / ref["sd0"] if ref["sd0"] > 0 else 0.0
    power = dm.power_total_decay(effect, k, h, R, L) if effect > 0 else {"p_detect": 0.0, "median_delay": None}
    # the state machine only means something for a signal with a demonstrated reference level
    sm = dm.run_state_machine(ic, R, ref["mu0"], ref["sd0"], k, h, rec_w) if (validated and ref["mu0"] > 0) else None

    status = "Unvalidated" if not validated else dm.STATE_NAMES[int(sm["state"][-1])]
    # exposure applied to period t = f(state after period t-1); first monitored period fully invested if validated
    e = np.zeros(n)
    if validated and sm is not None:
        e[R] = 1.0
        exp_map = np.array([config.EXPOSURE["Active"], config.EXPOSURE["Degrading"], config.EXPOSURE["Suspended"]])
        e[R + 1:] = exp_map[sm["state"][R:-1]]
    else:
        e[R:] = config.UNVALIDATED_EXPOSURE

    # --- recent vs reference
    rw = max(int(1.0 * ppy), 10)
    rec = ic[-rw:]
    ewm = dm.ewma(ic, config.EWMA_HALF_LIFE_YEARS * ppy)
    slope = dm.trend_slope(ic, int(config.SLOPE_YEARS * ppy))
    ls, t3 = m["ls"], m["top3"]
    w2 = int(config.SLOPE_YEARS * ppy)
    rec_alpha = dm.market_alpha(ls["net"][-w2:], m["ew"][-w2:])
    drag = lambda x: float((x["gross"] - x["net"]).mean() * ppy * 100)
    out.update({
        "status": status, "validated": bool(validated), "underpowered": bool(validated and power["p_detect"] < config.POWER_MIN),
        "reference": {"mu0": ref["mu0"], "sd0": ref["sd0"], "t": ref["t"], "n": ref["n"],
                      "ir_equivalent": float(ref["mu0"] / ref["sd0"] * np.sqrt(ppy)) if ref["sd0"] > 0 else None},
        "power_total_loss": {"p_detect": power["p_detect"], "median_delay_years": (power["median_delay"] / ppy) if power["median_delay"] else None},
        "detector": {"k": k, "h": h, "watch_h": config.WATCH_FRACTION * h, "S_now": _f(sm["S"][-1]) if sm else None,
                     "n_alarms": int(len(sm["alarms"])) if sm else 0, "n_reinstated": int(len(sm["reinstated"])) if sm else 0,
                     "first_alarm": dates[m["rows"][sm["alarms"][0]]] if sm and len(sm["alarms"]) else None,
                     "last_alarm": dates[m["rows"][sm["alarms"][-1]]] if sm and len(sm["alarms"]) else None},
        "recent": {"ic_mean": float(rec.mean()), "ic_se": float(rec.std(ddof=1) / np.sqrt(len(rec))), "ic_ewma": _f(ewm[-1]),
                   "z_vs_reference": float((rec.mean() - ref["mu0"]) / (ref["sd0"] / np.sqrt(len(rec)))),
                   "window_periods": int(rw)},
        "slope_ic_per_year": {"slope": None if slope["slope"] is None else slope["slope"] * ppy,
                              "se": None if slope["se"] is None else slope["se"] * ppy},
        "forecast_slope": {"ref": float(np.nanmean(m["fmb"][:R])), "recent": float(np.nanmean(m["fmb"][-rw:]))},
        "top3_net_pct_per_year": {"ref": float(t3["net"][:R].mean() * ppy * 100), "recent": float(t3["net"][-rw:].mean() * ppy * 100)},
        "ls_net_pct_per_year": {"ref": float(ls["net"][:R].mean() * ppy * 100), "recent": float(ls["net"][-rw:].mean() * ppy * 100)},
        "ls_cost_drag_pct_per_year": {"ref": drag({k_: v[:R] for k_, v in ls.items()}), "recent": drag({k_: v[-rw:] for k_, v in ls.items()})},
        "ls_turnover": {"ref": float(ls["turnover"][:R].mean()), "recent": float(ls["turnover"][-rw:].mean())},
        "ls_alpha_recent": rec_alpha, "ls_max_drawdown": dm.max_drawdown(ls["net"]),
        "regimes": dm.regime_table(ic, regimes, R, int(config.REGIME_RECENT_YEARS * ppy)),
    })

    # --- control-layer backtest on the long/short book, from the end of the reference
    sl = slice(R, n)
    Wl, Rm = m["W_ls"], m["R"]
    ung = dm.book_returns(Wl[sl], Rm[sl], H, False)
    gat = dm.book_returns(Wl[sl] * e[sl, None], Rm[sl], H, False)
    nw = max(int(config.NAIVE_FILTER_YEARS * ppy), 5)
    full_net = ls["net"]
    en = np.array([1.0 if full_net[max(0, t - nw):t].mean() > 0 else 0.0 for t in range(R, n)])
    nai = dm.book_returns(Wl[sl] * en[:, None], Rm[sl], H, False)
    blk = max(int(config.BOOT_BLOCK_YEARS * ppy), 3)
    out["gating"] = {"ungated": perf(ung["net"], np.ones(n - R), ppy), "gated": perf(gat["net"], e[sl], ppy),
                     "naive": perf(nai["net"], en, ppy),
                     "gated_vs_ungated": dm.block_bootstrap_mean_diff(gat["net"], ung["net"], blk, config.BOOT_REPS, seed=1),
                     "naive_vs_ungated": dm.block_bootstrap_mean_diff(nai["net"], ung["net"], blk, config.BOOT_REPS, seed=2),
                     "evaluated_from": dates[m["rows"][R]]}

    # --- series for the dashboard (decimated)
    step = max(int(config.SERIES_STEP * (1 if H == 1 else 0.2)), 1)
    idx = np.arange(0, n, step)
    cum_u, cum_g = np.cumsum(ung["net"]), np.cumsum(gat["net"])
    out["series"] = {"dates": [dates[m["rows"][i]] for i in idx], "ic": [_f(ic[i]) for i in idx], "ic_ewma": [_f(ewm[i]) for i in idx],
                     "cusum": [_f(sm["S"][i]) if sm else None for i in idx], "state": [int(sm["state"][i]) if sm else -1 for i in idx],
                     "cum_ls_ungated": [_f(cum_u[i - R]) if i >= R else None for i in idx],
                     "cum_ls_gated": [_f(cum_g[i - R]) if i >= R else None for i in idx]}
    return out


def analyze_universe(name: str, tickers: List[str], prices_df: pd.DataFrame) -> Optional[Dict]:
    available = [t for t in tickers if t in prices_df.columns]
    if not available:
        return None
    px = prices_df[available]
    px = px[~px.isna().any(axis=1)]
    if len(px) < 700:
        logger.warning(f"Not enough data for {name}")
        return None
    values = px.values
    log_prices = np.log(values)[1:]
    returns = np.diff(np.log(values), axis=0)
    dates = [d.strftime("%Y-%m-%d") for d in px.index[1:]]
    N = len(available)

    monitors, counts = [], {}
    for sname, (fn, desc) in sg.SIGNALS.items():
        scores = fn(returns, log_prices)
        for H in config.HORIZONS:
            m = dm.measure_signal(scores, returns, H)
            if m is None:
                continue
            regimes = dm.regime_labels(returns, m["rows"])
            ev = evaluate_monitor(m, H, N, regimes, dates)
            ev.update({"signal": sname, "description": desc, "universe": name, "n_etfs": N})
            monitors.append(ev)
            counts[ev["status"]] = counts.get(ev["status"], 0) + 1
        logger.info(f"  [{name}] {sname}: done")

    gated = [mo["gating"] for mo in monitors if "gating" in mo]
    diffs = [g["gated"]["mean_pct_per_year"] - g["ungated"]["mean_pct_per_year"] for g in gated if g["gated"] and g["ungated"]]
    summary = {"n_monitors": len(monitors), "status_counts": counts,
               "n_validated": int(sum(1 for mo in monitors if mo.get("validated"))),
               "n_underpowered": int(sum(1 for mo in monitors if mo.get("underpowered"))),
               "gating_mean_diff_pct_per_year": float(np.mean(diffs)) if diffs else None,
               "gating_share_better": float(np.mean([d > 0 for d in diffs])) if diffs else None}
    return {"monitors": monitors, "summary": summary, "n_etfs": N, "n_days": len(dates), "last_date": dates[-1], "first_date": dates[0]}


def run_trainer() -> Dict:
    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {"run_date": run_date, "algorithm": "ALPHA-DECAY-DETECTOR", "universes": {},
               "settings": {"reference_years": config.REFERENCE_YEARS, "fa_years": config.FA_YEARS,
                            "target_false_alarm": config.TARGET_FALSE_ALARM, "target_ir": config.TARGET_IR,
                            "validation_t": config.VALIDATION_T, "watch_fraction": config.WATCH_FRACTION,
                            "exposure": config.EXPOSURE, "horizons": config.HORIZONS, "cost_bps": config.TRADING_COST_BPS,
                            "signals": {k: v[1] for k, v in sg.SIGNALS.items()}}}
    for name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {name}...")
        out = analyze_universe(name, tickers, prices_df)
        if out is None:
            continue
        results["universes"][name] = _clean(out)
        s = out["summary"]
        logger.info(f"  monitors {s['n_monitors']} | statuses {s['status_counts']} | validated {s['n_validated']} | underpowered {s['n_underpowered']}")

    logger.info("\n📒 Scoring picks archived by other engines...")
    try:
        results["engines"] = _clean(el.run_ledger(prices_df, config.HF_TOKEN))
        for nm, r in results["engines"].items():
            logger.info(f"  {nm}: {r['status']} ({r['n_files']} files, {len(r['series'])} series)")
    except Exception as e:
        logger.warning(f"Engine ledger failed: {e}")
        results["engines"] = {}

    logger.info("\n🧪 Running simulation validation...")
    results["validation"] = _clean(val.run_validation(reps=config.SIM_REPS))

    output_path = f"decay_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, separators=(",", ":"), default=str)
    logger.info(f"\n💾 Saved: {output_path}")
    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")
    return results


if __name__ == "__main__":
    run_trainer()
