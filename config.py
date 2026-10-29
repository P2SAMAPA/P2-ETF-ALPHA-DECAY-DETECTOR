"""
config.py  —  Configuration for P2 ALPHA-DECAY-DETECTOR

The question is not "which ETF goes up" but "is a signal that worked in the
past still working now?" — and, just as importantly, how long it would take to
find out. A rank correlation between a signal and next-period returns is very
noisy (about 0.17 per period for a 36-ETF universe against true values of a few
hundredths), so detecting decay is a sequential change-detection problem whose
speed is limited by arithmetic, not cleverness. Every threshold below was
calibrated by simulation to a stated false-alarm rate, and the repo reports
what the detector can and cannot see (simulated detection delay, false alarms,
and the cost of gating in each kind of decay).
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-etf-alpha-decay-detector-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
}
TOP_N = 3

# ---------------------------------------------------------------------------
# Measurement. Per signal, per universe, per horizon H, on NON-OVERLAPPING dates
# spaced H days apart (so the periods are independent under no skill, whatever
# the signal's own persistence — a permutation argument, checked in testing):
#   ic          cross-sectional Spearman rank correlation of score and the next
#               H-day return              (is the ranking still informative?)
#   fmb_slope   Fama-MacBeth slope of realized return on the z-scored signal
#               (the "forecast error" view: realized per unit of predicted)
#   top3        net return of the top-3 vs the equal-weight universe
#   ls          net return of a dollar-neutral rank-weighted long/short book
#               (the TRADABLE strategy; the signal can rank well while this
#               loses to costs, and the reverse)
# Costs follow the sibling engines: TRADING_COST_BPS round trip per unit of
# notional replaced (turnover x bps/2) plus a short-borrow charge on short
# notional for the long/short book.
# ---------------------------------------------------------------------------
HORIZONS = [1, 5]
PERIODS_PER_YEAR = {1: 252.0, 5: 50.4}
TRADING_COST_BPS = 15
SHORT_BORROW_BPS_ANNUAL = 50

# ---------------------------------------------------------------------------
# Decay detection on the IC series (the primary measurement).
#
# Reference: the first REFERENCE_YEARS of the series define the signal's "alive"
# level (mean mu0, sd sigma0). If the reference mean IC is not significantly
# above zero (t < VALIDATION_T) there is nothing demonstrated to decay and the
# signal is "Unvalidated" — an honest state, not a failure of the detector.
#
# After the reference, a one-sided CUSUM for a DOWNWARD shift runs on standardized
# IC:  S_t = max(0, S_{t-1} + (mu0 - IC_t)/sigma0 - k).
#   * the slack k is tuned to a stated signal strength, TARGET_IR (annualized
#     information ratio whose total loss should be caught): the standardized
#     per-period effect of an annualized IR is ~ IR / sqrt(periods per year) for
#     any universe size, so k = 0.5 * TARGET_IR / sqrt(periods per year) (0.031 daily,
#     0.070 weekly). A fixed k is wrong across horizons: an IC of 0.05 on 1-DAY
#     returns would be an IR near 5, which no real signal has.
#   * The alarm threshold h is CALIBRATED BY SIMULATION so that, with the
#     reference mean and sd ESTIMATED from REFERENCE_YEARS of data, a signal
#     with NO decay raises a false alarm within FA_YEARS with probability
#     TARGET_FALSE_ALARM. (Estimation error in mu0 is the dominant source of
#     false alarms at these signal-to-noise ratios, so it is in the calibration.)
# State machine (decided with information through the previous period):
#   Active     S < WATCH_FRACTION * h
#   Degrading  WATCH_FRACTION * h <= S < h        -> exposure 0.5
#   Suspended  S >= h                              -> exposure 0
#   Reinstated only on statistically real recovery: the trailing REINSTATE_YEARS mean IC
#   must have t >= REINSTATE_T AND be at least REINSTATE_FRACTION * mu0 (the CUSUM then
#   restarts at 0). A first version needed only half of mu0 over one year; with a standard
#   error of ~0.024 that is met by luck about one year in seven, so a signal that died once
#   was reinstated and suspended again. The test is applied to a sliding window at every
#   period, so its false positives ACCUMULATE over a long suspension: with t >= 2 a dead signal
#   was spuriously reinstated at least once in 32% of 6-year suspensions; t >= 3 cuts that to 4%
#   while a genuinely recovered signal (IC back to 0.05) is still reinstated 90% / 95% / 98%
#   within 2 / 3 / 4 years.
# A monitor is also "Underpowered" when, even for a TOTAL loss of its reference
# IC, the detector would catch it within FA_YEARS with probability below
# POWER_MIN.
# ---------------------------------------------------------------------------
REFERENCE_YEARS = 3.0
FA_YEARS = 3.0
TARGET_FALSE_ALARM = 0.10
TARGET_IR = 1.0
WATCH_FRACTION = 0.5
VALIDATION_T = 1.65
POWER_MIN = 0.5
EWMA_HALF_LIFE_YEARS = 0.5
REINSTATE_YEARS = 2.0
REINSTATE_FRACTION = 0.5
REINSTATE_T = 3.0
SLOPE_YEARS = 2.0
EXPOSURE = {"Active": 1.0, "Degrading": 0.5, "Suspended": 0.0}
UNVALIDATED_EXPOSURE = 0.0

# Regime diagnostics (is the decay concentrated in one market regime?). Regimes
# are known at decision time: high/low volatility (EW universe 21-day vol vs its
# trailing 252-day median) and up/down trend (sign of the 63-day EW return).
REGIME_RECENT_YEARS = 2.0

# Gating backtest (does the control layer help?): the long/short book of every
# monitored signal, ungated vs gated by the detector vs gated by a naive trailing
# filter; circular block bootstrap on the paired difference.
NAIVE_FILTER_YEARS = 1.0
BOOT_BLOCK_YEARS = 0.25
BOOT_REPS = 1000

# Simulation validation (re-run every time and shipped in the results).
SIM_REPS = 300
SIM_N_ETFS = 36

# External engines: the daily results JSONs the other engines archive in their
# own HF datasets contain their top picks. The ledger adapter scores those picks
# against realized returns (top picks minus the equal-weight universe, net of
# costs) and reports the series. Add entries as
# {"name": ..., "repo": "P2SAMAPA/<dataset>", "prefix": "<file prefix>"}.
# Missing datasets or schemas are logged and skipped — never fatal.
EXTERNAL_ENGINES = [
    {"name": "graphon-mean-field", "repo": "P2SAMAPA/p2-etf-graphon-mean-field-results", "prefix": "graphon_results_"},
    {"name": "spectral-flow", "repo": "P2SAMAPA/p2-etf-spectral-flow-results", "prefix": "spectral_results_"},
    {"name": "hyperbolic-flows", "repo": "P2SAMAPA/p2-etf-hyperbolic-flows-results", "prefix": "hyperbolic_results_"},
    {"name": "dynamic-ot-barycenter", "repo": "P2SAMAPA/p2-etf-dynamic-ot-barycenter-results", "prefix": "barycenter_results_"},
    {"name": "delay-agent-dynamics", "repo": "P2SAMAPA/p2-etf-delay-agent-dynamics-results", "prefix": "delay_results_"},
    {"name": "morphogenesis", "repo": "P2SAMAPA/p2-etf-morphogenesis-results", "prefix": "morpho_results_"},
]
MAX_ENGINE_FILES = 400
LEDGER_MIN_PERIODS = 12
SERIES_STEP = 5
