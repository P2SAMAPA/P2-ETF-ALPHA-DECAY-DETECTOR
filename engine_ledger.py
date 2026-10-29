"""
engine_ledger.py  —  Score the picks other engines have already archived.

Each engine pushes a daily results JSON to its own HF dataset. Those files contain the engine's top
picks; this adapter turns them into a realized-performance series — the picks' return minus the
equal-weight universe over the pick's horizon, net of costs — so the same decay questions can be asked
of an ENGINE as of a research signal. Only the top picks are archived (not the full ranking), so no IC
is available: the series is the net spread.

Dates use the last trading day on or before the file's run date as the decision point and the NEXT
days' returns as the outcome. Entries are thinned to non-overlapping holding periods. Anything missing
or unrecognized (no dataset, no files, a different schema) is reported and skipped, never fatal.
"""

import json
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

import config


def parse_picks(doc: Dict, universes: Dict[str, List[str]]) -> List[Dict]:
    """Tolerant parser: top_picks[universe] = [ {ticker | symbol, horizon_days | horizon}, ... ]."""
    out = []
    date = doc.get("run_date")
    tp = doc.get("top_picks")
    if not date or not isinstance(tp, dict):
        return out
    for uni, picks in tp.items():
        if not isinstance(picks, list) or not picks:
            continue
        tickers, horizons = [], []
        for p in picks:
            if not isinstance(p, dict):
                continue
            t = p.get("ticker") or p.get("symbol")
            if t:
                tickers.append(str(t))
            h = p.get("horizon_days") or p.get("horizon")
            if h:
                horizons.append(int(h))
        if not tickers:
            continue
        if not horizons:
            sel = (doc.get("selected") or {}).get(uni) or {}
            h = sel.get("horizon") or sel.get("horizon_days")
            horizons = [int(h)] if h else [1]
        meta = (doc.get("universe_meta") or {}).get(uni) or {}
        members = meta.get("tickers") or universes.get(uni)
        out.append({"date": str(date)[:10], "universe": uni, "tickers": tickers,
                    "horizon": int(max(set(horizons), key=horizons.count)), "members": members})
    return out


def realized_series(entries: List[Dict], prices: pd.DataFrame, cost_bps: float = None) -> Dict[str, Dict]:
    """
    Per (universe, horizon): non-overlapping matured entries -> net excess return series.
    excess = mean forward return of the picks - mean forward return of the universe members; cost = fraction of
    picks replaced vs the previous kept entry x cost_bps (the sibling engines' convention).
    """
    cost_bps = config.TRADING_COST_BPS if cost_bps is None else cost_bps
    logp = np.log(prices)
    idx = prices.index
    groups: Dict = {}
    for e in entries:
        groups.setdefault((e["universe"], e["horizon"]), []).append(e)
    out = {}
    for (uni, h), lst in groups.items():
        lst = sorted(lst, key=lambda e: e["date"])
        last_pos, rows = -10 ** 9, []
        for e in lst:
            pos = idx.searchsorted(pd.Timestamp(e["date"]), side="right") - 1
            if pos < 0 or pos - last_pos < h or pos + h >= len(idx):
                continue
            members = [t for t in (e["members"] or []) if t in prices.columns]
            picks = [t for t in e["tickers"] if t in prices.columns]
            if len(members) < 3 or not picks:
                continue
            fwd = logp.iloc[pos + h][members] - logp.iloc[pos][members]
            if fwd.isna().any():
                continue
            excess = float(fwd[picks].mean() - fwd.mean())
            prev = rows[-1]["picks"] if rows else None
            repl = (len(picks) - len(set(picks) & set(prev))) / len(picks) if prev else 0.0
            net = excess - repl * cost_bps / 1e4
            rows.append({"date": e["date"], "exit_date": str(idx[pos + h].date()), "picks": picks, "gross": excess, "net": net})
            last_pos = pos
        if rows:
            out[(uni, h)] = {"universe": uni, "horizon": h, "rows": rows}
    return out


def summarize_series(rows: List[Dict]) -> Dict:
    net = np.array([r["net"] for r in rows], dtype=float)
    n = len(net)
    t = float(net.mean() / (net.std(ddof=1) / np.sqrt(n))) if n >= 3 and net.std(ddof=1) > 0 else None
    return {"n_periods": int(n), "net_mean": float(net.mean()), "net_t": t, "cum_net": float(net.sum()),
            "hit_rate": float((net > 0).mean()), "first_date": rows[0]["date"], "last_date": rows[-1]["date"]}


def fetch_engine_docs(repo: str, prefix: str, token: Optional[str], limit: int) -> List[Dict]:
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi(token=token).list_repo_files(repo, repo_type="dataset")
    names = sorted(f for f in files if f.startswith(prefix) and f.endswith(".json"))[-limit:]
    docs = []
    for f in names:
        try:
            with open(hf_hub_download(repo_id=repo, filename=f, repo_type="dataset", token=token), "r") as fh:
                docs.append(json.load(fh))
        except Exception:
            continue
    return docs


def run_ledger(prices: pd.DataFrame, token: Optional[str]) -> Dict:
    """All configured engines -> {engine: {status, series: {'UNIVERSE|h': summary}}}."""
    report = {}
    for eng in config.EXTERNAL_ENGINES:
        try:
            docs = fetch_engine_docs(eng["repo"], eng["prefix"], token, config.MAX_ENGINE_FILES)
        except Exception as e:
            report[eng["name"]] = {"status": "unavailable: " + type(e).__name__, "series": {}, "n_files": 0}
            continue
        entries = []
        for d in docs:
            entries.extend(parse_picks(d, config.UNIVERSES))
        series = realized_series(entries, prices)
        report[eng["name"]] = {"status": "ok" if series else "no scoreable picks", "n_files": len(docs),
                               "series": {"%s|%d" % (k[0], k[1]): {**summarize_series(v["rows"]),
                                          "recent": [{"date": r["date"], "net": r["net"]} for r in v["rows"][-5:]]}
                                          for k, v in series.items()}}
    return report
