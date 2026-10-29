# P2-ETF-ALPHA-DECAY-DETECTOR

Instead of only asking *which ETF will go up*, ask **whether a signal that has
worked is still working** — and, just as important, **how long it would take to
find out**.

Alpha decay (a signal's edge eroding through regime change, crowding, structural
change or rising costs) is hard to detect for a plain arithmetic reason: the
per-period rank correlation between a signal and next-period returns has a
standard deviation of about 0.17 for a 36-ETF universe (0.41 for 7 ETFs) against
true values of a few hundredths. So detecting decay is a *sequential
change-detection* problem whose speed is limited by information, not cleverness.
This repo builds the detector, calibrates every threshold by simulation to a stated
false-alarm rate, and reports honestly what it can and cannot see.

## The three questions

1. **Is the signal still generating useful predictions?** Rolling rank IC with its
   confidence interval, an EWMA, the change versus the reference level (in standard
   errors), and the status.
2. **If it is deteriorating, how quickly?** The slope of the IC over the last two
   years with its standard error, and the CUSUM statistic relative to its threshold.
   (No time-to-zero extrapolation is produced: a sibling engine's extrapolation fired
   on pure noise 17–24% of days, so none is offered here.)
3. **Temporary or structural?** Reported as *evidence*, not a verdict, because returns
   alone cannot settle it: regime-conditional IC (reference vs recent, in high/low
   volatility and up/down trends), cost erosion (gross vs net, turnover), and alpha
   versus the universe. A dip-and-recover is, by construction, indistinguishable from
   a death until it recovers.

## Measurement (per signal, universe and horizon)

On **non-overlapping** periods H days apart (H = 1 and 5):

| measurement | what it answers |
|---|---|
| rank IC | is the ranking still informative? |
| Fama–MacBeth slope of realized on z-scored signal | the "forecast error" view: realized return per unit of predicted |
| top-3 net spread vs the universe | the pick-style product the other engines produce |
| dollar-neutral rank-weighted long/short book, net of costs | the **tradable strategy** — a signal can rank well while this loses to costs, and the reverse |
| alpha vs the equal-weight universe, drawdown, turnover | is the P&L skill or market exposure; how bad was it |

Non-overlapping periods matter: with persistent signals, *overlapping* daily-date ICs
have lag-1 autocorrelation of 0.75 (overstating the evidence), whereas on
non-overlapping periods a very persistent signal (AR(1) scores, φ = 0.98/day) against
i.i.d. returns gave IC autocorrelations of −0.029, −0.008, +0.053 at lags 1–3 and an sd
of 0.1699 against the theoretical 1/√(N−1) = 0.1690. Costs follow the sibling engines
(turnover × 15 bps / 2, plus short borrow on short notional); one of three picks
replaced = 5.000 bps, checked.

## Detection and the control layer

* **Reference.** The first 3 years define the signal's alive level (mean, sd). If the
  reference IC is not significantly above zero (t < 1.65) the signal is **Unvalidated**
  — nothing demonstrated to decay. That is an honest state, not a failure.
* **CUSUM** for a downward shift in standardized IC. The slack is tuned to a stated
  strength, `TARGET_IR = 1` (the per-period standardized effect of an annualized IR is
  ≈ IR/√periods-per-year for *any* universe size): k = 0.031 daily, 0.070 weekly. The
  alarm threshold (h = 46.4 daily, 20.3 weekly) is **calibrated by simulation** so that
  a signal that never decays raises a false alarm within 3 years with probability 10%,
  *including the error in estimating the reference mean* — which is the dominant source
  of false alarms at these signal-to-noise ratios. On fresh no-decay data the realised
  false-alarm rates were 10.9% and 9.9%.
* **States:** Active (CUSUM < ½h, exposure 1), Degrading (exposure ½), Suspended (≥ h,
  exposure 0). **Reinstatement** requires the trailing 2-year mean IC to be both ≥ ½ of
  the reference level *and* statistically real (t ≥ 3). Two first versions were too
  loose: needing only half the reference over one year reinstated a once-dead signal by
  luck about one year in seven, and even t ≥ 2 reinstated a dead signal at least once in
  32% of 6-year suspensions, because the test slides forward every period and its false
  positives accumulate. At t ≥ 3 that is 4%, while a signal that truly recovered (IC back
  to 0.05) is still reinstated 90% / 95% / 98% within 2 / 3 / 4 years.
* **Power check per monitor.** "Underpowered" means that even a total loss of the signal's
  reference IC would be caught within 3 years less than half the time.

## What the detector can and cannot see (simulation, every run)

If a signal dies completely, the chance it is caught within 3 years at a 10% false-alarm
probability depends on its **annualized IR, not on the horizon or the universe size**:

| gross IR of the signal | IR 0.5 | IR 1.0 | IR 1.5 | IR 2.0 |
|---|---|---|---|---|
| caught within 3 years | 22–23% | 41–45% | 62–67% | 80–84% |
| median delay when caught | ~2.0 y | ~1.8–1.9 y | ~1.6–1.7 y | ~1.4–1.5 y |

(Four combinations — daily / weekly × 36 / 7 ETFs — agree to within a few points.) So a
Sharpe-1 signal that dies is *more likely than not to go unnoticed for three years*. That
is an information limit, not something tuning can fix.

**What the control layer does to returns**, by kind of decay (weekly, 36 ETFs, 300
simulated histories per row, % of capital over 4 years after a 3-year reference; costs
a flat 5 bps per period at full exposure, an assumption):

| scenario | passes validation | detector alarms (delay) | always traded | validation gate only | gate + detector | naive trailing-year filter |
|---|---|---|---|---|---|---|
| no skill | 5% | – | −9.4 | −0.5 | −0.2 | −3.1 |
| stable skill, IR 1 | 48% | – | +12.3 | +5.4 | +4.5 | +8.6 |
| stable skill, IR 2 | 95% | 16% (false) | +34.0 | +32.1 | +28.8 | +31.7 |
| dies, IR 2 → 0 | 95% | 93% (1.6 y) | −9.4 | −8.9 | −3.8 | −3.8 |
| dies, IR 1 → 0 | 48% | 79% (1.9 y) | −9.4 | −5.0 | −2.7 | −3.6 |
| inverts, IR 2 → −2 | 95% | 100% (0.8 y) | −52.7 | −50.0 | −8.3 | −5.6 |
| decays over 2 y, IR 2 → 0 | 95% | 85% (2.4 y) | +1.4 | +1.3 | +4.1 | +4.1 |
| dips for 1 y then recovers | 95% | 39% | +23.2 | +21.9 | +14.3 | +19.9 |

How to read it, honestly:

* The **validation gate** (never trade what has not shown skill) does most of the work
  when there is no skill, and **costs** real-but-modest signals (at IR 1 only half are
  validated, so half the real alpha is forgone).
* The **detector** adds most when a signal **inverts** (a crowded trade turning against
  you: about +42 points over the validation gate alone, −50.0 → −8.3), a little when it dies or decays, and gives some back when a signal
  merely dips and recovers or is stable.
* A **plain "trade only if the last year's net return was positive" filter does about as
  well on net returns** — usually better on stable and dip scenarios, level on a sudden
  death. The detector's advantage is a **known, calibrated false-alarm rate and a power
  statement**, not more money. If you only want a cheap safeguard, the naive filter is
  most of it.
* Across false-alarm tolerances of 10% / 25% / 50% and several early-warning levels, the
  average benefit moved only between +6.1 and +6.7 points and the same scenarios won and
  lost, so the calibrated 10% default was kept rather than tuned for appearance.

## Signals monitored

* **Research probes** (`signals.py`): 21-day momentum, 6–1-month momentum, 5-day reversal,
  low volatility, reversion to the 126-day trend — signs fixed in advance from the
  literature. They are *probes with known history*, not claims that they work; most are
  expected to come out Unvalidated.
* **Other engines' archived picks** (`engine_ledger.py`): each engine pushes a daily
  results JSON to its own HF dataset; the adapter scores the picks against realized
  returns (picks minus the equal-weight universe over the pick's horizon, net of costs,
  non-overlapping) and reports the series. Only top picks are archived, so there is no IC.
  Tested on synthetic archives: tolerant of malformed files, exact against a hand
  calculation, costs charged only when picks change. Datasets/schemas that are missing are
  reported and skipped, never fatal. A decay verdict needs ~3 years of history, so for
  newly built engines this is a record, not yet a monitor. Edit `EXTERNAL_ENGINES` in
  `config.py` to add engines.

## Validation status (please read)

Tested on synthetic data only; this sandbox cannot reach `huggingface.co`.

* `row_spearman` identical to scipy (5.6e-17); costs and independence as stated above.
* **End to end on a synthetic universe with a planted momentum effect** (30 ETFs, 3,500
  days): pure noise → all monitors Unvalidated; a signal that dies → validated (IR-equivalent
  ≈ 2.1–2.2, reference t ≈ 3.6–3.8) then Suspended about 1.2–1.5 years after the true death
  (matching the simulated median delay); a signal that inverts → Suspended within about half
  a year. The full run (3 universes, 30 monitors, gating backtests, simulation validation) takes about
  14 seconds of compute, plus the time to fetch the other engines' archives.
* The thresholds, reinstatement bar and economics above were chosen on **simulation**, not
  on real data, so the real-data results are not contaminated by them.

## Limits

* The signals are monitored on the history available, with the first 3 years as reference;
  a signal that had *already* decayed during the reference window looks never-alive.
* Regime-conditional IC cells hold only tens of periods; treat as evidence.
* The simulated economics assume a flat cost per period and independent cross-sections; real
  turnover and regime dependence will differ.
* Crowding itself is not identifiable from price data; "crowding" appears here only as one
  explanation for a sign-inverting decay.
* **About the spec:** the text provided cut off after the first of four measurements (rolling
  rank IC). The accompanying diagram lists rolling alpha, top-three spread, drawdown and
  forecast errors, which are included; if measurements B–D were meant to be something
  specific, they are easy to add in `decay_model.measure_signal`.

This is not financial advice.

## Repo structure

```
config.py          universes, horizons, reference/false-alarm settings, state machine, costs,
                   external-engine list
data_manager.py     master parquet from HF (same as sibling repos)
signals.py          the research-signal probes
decay_model.py      measurement, CUSUM, simulation-calibrated thresholds, power, state machine,
                   regime / cost / alpha diagnostics, paired block bootstrap
engine_ledger.py    scores picks other engines archived on HF
validation.py       decay scenarios and the economics of gating; power by annualized IR
trainer.py          per-universe monitors, gating backtest, engine ledger, validation, JSON
push_results.py     HfApi.upload_file to the results dataset
us_calendar.py       trading-day helpers
streamlit_app.py     Signal health / Does the control layer help? / Engine picks / What the
                   detector can see
.github/workflows/daily_run.yml   weekday cron after the US close (GitHub Actions)
```

## Running (GitHub Actions)

The workflow runs `python trainer.py` with the `HF_TOKEN` secret and pushes
`decay_results_YYYY-MM-DD.json` to `P2SAMAPA/p2-etf-alpha-decay-detector-results`. The
simulation validation re-runs each time and ships in the JSON. For a local dashboard:
`streamlit run streamlit_app.py`.
