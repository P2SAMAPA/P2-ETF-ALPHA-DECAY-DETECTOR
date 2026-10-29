"""
validation.py  —  What can the detector see, and what does the control layer do to returns?

Cross-sectional panels are simulated period by period: for period t the signal s
(N ETFs) and the realized returns r satisfy a Spearman correlation of IC_true[t]
(via the bivariate-normal relation rho = 2 sin(pi*IC/6)), with a cross-sectional
return dispersion of SIGMA_R per period. The tradable book is the dollar-neutral
rank-weighted long/short book; costs are a flat COST_BPS per period at full
exposure (an assumption standing in for turnover and borrow).

The whole control layer is exercised exactly as in production: 3-year reference ->
validation gate (reference mean IC must be significantly above zero) -> CUSUM with
the calibrated threshold -> Active / Degrading / Suspended -> exposure 1 / 0.5 / 0
applied to the NEXT period -> reinstatement. Scenarios are expressed in annualized
information ratios (IR ~ IC * sqrt(N * periods per year)) because that is what a
trader recognizes, and it is what detection power depends on.
"""

import numpy as np
from typing import Dict, List

import config
import decay_model as dm

SIGMA_R = 0.02          # weekly cross-sectional dispersion of returns
COST_BPS = 5.0          # per period at full exposure (stand-in for turnover + borrow)
HORIZON = 5
MONITOR_YEARS = 4.0


def ic_for_ir(ir: float, N: int, ppy: float) -> float:
    return ir / np.sqrt(N * ppy)


def simulate_panels(ic_path: np.ndarray, N: int, reps: int, rng) -> Dict[str, np.ndarray]:
    """Per-period exact Spearman IC and rank-weighted long/short gross return for `reps` independent histories."""
    P = len(ic_path)
    rho = 2.0 * np.sin(np.pi * ic_path / 6.0)
    S = rng.normal(size=(reps, P, N))
    E = rng.normal(size=(reps, P, N))
    Rr = rho[None, :, None] * S + np.sqrt(1.0 - rho[None, :, None] ** 2) * E
    rs = np.argsort(np.argsort(S, axis=2), axis=2).astype(float)
    rr = np.argsort(np.argsort(Rr, axis=2), axis=2).astype(float)
    a, b = rs - rs.mean(axis=2, keepdims=True), rr - rr.mean(axis=2, keepdims=True)
    ic = (a * b).sum(axis=2) / np.sqrt((a ** 2).sum(axis=2) * (b ** 2).sum(axis=2))
    c = rs - (N - 1) / 2.0
    pos = np.where(c > 0, c, 0).sum(axis=2, keepdims=True)
    w = c / pos
    gross = (w * SIGMA_R * Rr).sum(axis=2)
    return {"ic": ic, "gross": gross}


def scenario_paths(ir: float, shape: str, N: int, ppy: float, R: int, M: int) -> np.ndarray:
    """True IC path over R reference + M monitored periods."""
    base = ic_for_ir(ir, N, ppy)
    path = np.full(R + M, base)
    if shape == "stable":
        return path
    if shape == "dies":
        path[R:] = 0.0
    elif shape == "inverts":
        path[R:] = -base
    elif shape == "decays":
        path[R:R + int(2 * ppy)] = np.linspace(base, 0.0, int(2 * ppy))
        path[R + int(2 * ppy):] = 0.0
    elif shape == "dips":
        d = int(1 * ppy)
        path[R:R + d] = 0.0                       # a year of nothing, then it comes back
    return path


def run_scenario(ir: float, shape: str, N: int, reps: int, seed: int, horizon: int = HORIZON) -> Dict:
    ppy = config.PERIODS_PER_YEAR[horizon]
    R, L = int(config.REFERENCE_YEARS * ppy), int(config.FA_YEARS * ppy)
    M = int(MONITOR_YEARS * ppy)
    k = dm.default_k(horizon)
    h = dm.calibrate_threshold(R, L, k)
    rng = np.random.default_rng(seed)
    path = scenario_paths(ir, shape, N, ppy, R, M)
    pan = simulate_panels(path, N, reps, rng)
    rec_w = max(int(config.REINSTATE_YEARS * ppy), 5)
    cost = COST_BPS / 1e4
    alarmed, delays, validated, reinst = [], [], [], []
    ung, gat, naive, gate_only = np.zeros(reps), np.zeros(reps), np.zeros(reps), np.zeros(reps)
    for i in range(reps):
        ic, gross = pan["ic"][i], pan["gross"][i]
        ref = dm.reference_stats(ic[:R])
        ok = ref["t"] >= config.VALIDATION_T
        validated.append(ok)
        ung[i] = (gross[R:] - cost).sum()
        gate_only[i] = ung[i] if ok else 0.0          # the validation gate alone: trade iff the reference IC was significant
        e = np.zeros(M)
        if ok:
            sm = dm.run_state_machine(ic, R, ref["mu0"], ref["sd0"], k, h, rec_w)
            st = sm["state"][R:]
            exp_map = np.array([config.EXPOSURE["Active"], config.EXPOSURE["Degrading"], config.EXPOSURE["Suspended"]])
            e[0] = 1.0
            e[1:] = exp_map[st[:-1]]
            al = sm["alarms"]
            alarmed.append(len(al) > 0)
            if len(al):
                delays.append((al[0] - R) / ppy)
            reinst.append(len(sm["reinstated"]) > 0)
        else:
            alarmed.append(False)
            reinst.append(False)
        gat[i] = (e * (gross[R:] - cost)).sum()
        # naive filter: trade next period only if the trailing year's UNGATED net mean was positive
        net_all = gross - cost
        nw = max(int(config.NAIVE_FILTER_YEARS * ppy), 5)
        en = np.array([1.0 if net_all[max(0, t - nw):t].mean() > 0 else 0.0 for t in range(R, R + M)])
        naive[i] = (en * net_all[R:]).sum()
    validated, alarmed = np.array(validated), np.array(alarmed)
    out = {"ir": ir, "shape": shape, "N": N, "reps": reps, "monitor_years": MONITOR_YEARS,
           "p_validated": float(validated.mean()),
           "p_alarm_given_validated": float(alarmed[validated].mean()) if validated.any() else None,
           "median_delay_years": float(np.median(delays)) if delays else None,
           "p_reinstated_given_alarm": float(np.mean(np.array(reinst)[alarmed])) if alarmed.any() else None,
           "net_ungated_pct": float(100 * ung.mean()), "net_gated_pct": float(100 * gat.mean()),
           "net_naive_pct": float(100 * naive.mean()), "net_gate_only_pct": float(100 * gate_only.mean()),
           "gated_minus_ungated_pct": float(100 * (gat - ung).mean()), "p_gated_better": float(np.mean(gat > ung)),
           "naive_minus_ungated_pct": float(100 * (naive - ung).mean())}
    return out


SCENARIOS = [
    ("No skill", 0.0, "stable"), ("Stable skill, IR 1", 1.0, "stable"), ("Stable skill, IR 2", 2.0, "stable"),
    ("Dies suddenly, IR 2 -> 0", 2.0, "dies"), ("Dies suddenly, IR 1 -> 0", 1.0, "dies"),
    ("Inverts, IR 2 -> -2 (crowded out)", 2.0, "inverts"), ("Decays over 2 years, IR 2 -> 0", 2.0, "decays"),
    ("Dips for a year then recovers, IR 2", 2.0, "dips"),
]


def run_validation(reps: int = None, N: int = None, seed: int = 11) -> Dict:
    reps = config.SIM_REPS if reps is None else reps
    N = config.SIM_N_ETFS if N is None else N
    rows = []
    for label, ir, shape in SCENARIOS:
        r = run_scenario(ir, shape, N, reps, seed)
        r["label"] = label
        rows.append(r)
    # power by annualized IR (total loss), daily and weekly, 36 and 7 ETFs
    power = []
    for hz in config.HORIZONS:
        ppy = config.PERIODS_PER_YEAR[hz]
        R, L = int(config.REFERENCE_YEARS * ppy), int(config.FA_YEARS * ppy)
        k = dm.default_k(hz)
        h = dm.calibrate_threshold(R, L, k)
        for n_etf in (36, 7):
            sd = 1.0 / np.sqrt(n_etf - 1)
            for ir in (0.5, 1.0, 1.5, 2.0):
                p = dm.power_total_decay(ic_for_ir(ir, n_etf, ppy) / sd, k, h, R, L)
                power.append({"horizon": hz, "N": n_etf, "ir": ir, "p_detect": p["p_detect"],
                              "median_delay_years": (p["median_delay"] / ppy) if p["median_delay"] else None, "k": k, "h": h})
    return {"scenarios": rows, "power": power, "reps": reps, "N": N, "sigma_r": SIGMA_R, "cost_bps_per_period": COST_BPS,
            "horizon": HORIZON, "target_false_alarm": config.TARGET_FALSE_ALARM}
