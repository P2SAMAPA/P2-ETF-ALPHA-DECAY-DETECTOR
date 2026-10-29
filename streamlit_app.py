import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import glob
import os
import plotly.graph_objects as go

st.set_page_config(page_title="P2 Alpha-Decay-Detector", page_icon="📉", layout="wide",
                   initial_sidebar_state="collapsed")

PRIMARY = "#0f766e"
INK = "#0f172a"
SUBTLE = "#64748b"
CARD_BORDER = "#e2e8f0"
PAGE_BG = "#f8fafc"
STATUS_COLOR = {"Active": "#16a34a", "Degrading": "#d97706", "Suspended": "#dc2626", "Unvalidated": "#64748b",
                "Insufficient history": "#94a3b8"}

CSS = """
<style>
    .stApp { background-color: __PAGE_BG__; }
    #MainMenu, footer {visibility: hidden;}
    .app-title { font-size: 1.9rem; font-weight: 800; color: __INK__; margin: 0; }
    .app-subtitle { color: __SUBTLE__; font-size: 0.95rem; margin-bottom: 1.1rem; }
    .universe-heading { font-size: 1.15rem; font-weight: 700; color: __INK__; margin: 0 0 0.15rem 0; }
    .kpi-card { background: #ffffff; border: 1px solid __BORDER__; border-radius: 12px; padding: 0.8rem 1rem; text-align: center; }
    .kpi-value { font-size: 1.3rem; font-weight: 800; color: __INK__; }
    .kpi-label { font-size: 0.7rem; color: __SUBTLE__; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.15rem; }
    .note { font-size: 0.82rem; color: __SUBTLE__; margin: 0.4rem 0 0.9rem 0; }
    .banner { background: #ffffff; border: 1px solid __BORDER__; border-left: 6px solid var(--accent);
        border-radius: 10px; padding: 0.8rem 1.1rem; font-size: 0.9rem; color: __INK__; margin: 0.5rem 0 0.9rem 0; }
    div[data-testid="stDataFrame"] { border: 1px solid __BORDER__; border-radius: 10px; overflow: hidden; }
</style>
"""
CSS = (CSS.replace("__PAGE_BG__", PAGE_BG).replace("__INK__", INK).replace("__SUBTLE__", SUBTLE)
       .replace("__BORDER__", CARD_BORDER))
st.markdown(CSS, unsafe_allow_html=True)

RESULTS_REPO = "P2SAMAPA/p2-etf-alpha-decay-detector-results"
STATE_NAMES = {0: "Active", 1: "Degrading", 2: "Suspended"}


def _find_latest_hf_result(repo_id):
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo_id, repo_type="dataset")
        names = sorted(f for f in files if f.startswith("decay_results_") and f.endswith(".json"))
        if not names:
            return None, "No decay_results_*.json files found in " + repo_id + "."
        return names[-1], None
    except Exception as e:
        return None, "Could not list files in " + repo_id + ": " + str(e)


@st.cache_data(show_spinner="Loading results...")
def load_data(_cache_key=""):
    notes = []
    local = glob.glob("decay_results_*.json")
    if local:
        latest = sorted(local)[-1]
        try:
            with open(latest, "r") as f:
                return json.load(f), ["Loaded local file: " + latest]
        except Exception as e:
            notes.append("Found local file " + latest + " but couldn't parse it: " + str(e))
    name, err = _find_latest_hf_result(RESULTS_REPO)
    if err:
        notes.append(err)
    else:
        try:
            url = "https://huggingface.co/datasets/" + RESULTS_REPO + "/resolve/main/" + name
            headers = {}
            token = os.environ.get("HF_TOKEN")
            if token:
                headers["Authorization"] = "Bearer " + token
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json(), ["Loaded from HF dataset: " + name]
            notes.append("Fetching " + name + " returned HTTP " + str(resp.status_code) + ".")
        except Exception as e:
            notes.append("Fetching " + name + " failed: " + str(e))
    return None, notes


def kpi(label, value):
    return ('<div class="kpi-card"><div class="kpi-value">' + value + '</div>'
            '<div class="kpi-label">' + label + '</div></div>')


def fnum(x, fmt="%.2f", na="n/a"):
    return na if x is None else (fmt % x)


def monitor_table(monitors):
    order = {"Suspended": 0, "Degrading": 1, "Active": 2, "Unvalidated": 3, "Insufficient history": 4}
    rows = []
    for m in sorted(monitors, key=lambda m: (order.get(m["status"], 9), m["signal"], m["horizon"])):
        if m["status"] == "Insufficient history":
            continue
        ref, rec, pw, d = m["reference"], m["recent"], m["power_total_loss"], m["detector"]
        flag = m["status"] + (" (underpowered)" if m.get("underpowered") else "")
        rows.append({"Signal": m["signal"], "Horizon (d)": m["horizon"], "Status": flag,
                     "Reference IC": ref["mu0"], "Ref t": ref["t"], "Ref IR-equiv": ref.get("ir_equivalent"),
                     "IC last year": rec["ic_mean"], "Change vs ref (z)": rec["z_vs_reference"],
                     "P(catch total loss)": pw["p_detect"], "CUSUM / threshold": (d["S_now"] / d["h"]) if d.get("S_now") is not None else None,
                     "Alarms": d["n_alarms"] if m["validated"] else None})
    return pd.DataFrame(rows)


def render_monitor_detail(m, key):
    s = m["series"]
    ref = m["reference"]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s["dates"], y=s["ic"], mode="lines", name="IC (per period)", line=dict(color="#cbd5e1", width=1)))
        fig.add_trace(go.Scatter(x=s["dates"], y=s["ic_ewma"], mode="lines", name="EWMA", line=dict(color=PRIMARY, width=2)))
        fig.add_hline(y=ref["mu0"], line_dash="dash", line_color="#334155", annotation_text="reference mean")
        fig.add_hline(y=0, line_color="#94a3b8")
        fig.update_layout(height=290, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
                          yaxis=dict(title="Rank IC", gridcolor="#eef2f7"), xaxis=dict(gridcolor="#eef2f7"), hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True, key="ic_" + key)
    with c2:
        if m["validated"] and any(v is not None for v in s["cusum"]):
            d = m["detector"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["dates"], y=s["cusum"], mode="lines", name="CUSUM", line=dict(color="#b45309", width=1.8)))
            fig.add_hline(y=d["h"], line_dash="dash", line_color="#dc2626", annotation_text="alarm")
            fig.add_hline(y=d["watch_h"], line_dash="dot", line_color="#d97706", annotation_text="watch")
            fig.update_layout(height=290, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
                              yaxis=dict(title="CUSUM (sd units)", gridcolor="#eef2f7"), xaxis=dict(gridcolor="#eef2f7"), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, key="cusum_" + key)
        else:
            st.info("No CUSUM: this signal has no demonstrated reference level to decay from (Unvalidated).")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["dates"], y=[v * 100 if v is not None else None for v in s["cum_ls_ungated"]], mode="lines",
                             name="Long/short book, always traded", line=dict(color="#94a3b8", width=1.8)))
    fig.add_trace(go.Scatter(x=s["dates"], y=[v * 100 if v is not None else None for v in s["cum_ls_gated"]], mode="lines",
                             name="Gated by the detector", line=dict(color=PRIMARY, width=2)))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      yaxis=dict(title="Cumulative net return after the reference (%)", gridcolor="#eef2f7"),
                      xaxis=dict(gridcolor="#eef2f7"), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, key="cum_" + key)


def render_evidence(m):
    sl, la = m["slope_ic_per_year"], m["ls_alpha_recent"]
    rows = [
        {"Measure": "IC trend over the last 2 years (per year)", "Value": fnum(sl["slope"], "%+.4f") + " (se " + fnum(sl["se"], "%.4f") + ")"},
        {"Measure": "Forecast slope (realized per unit of signal), reference -> recent", "Value": fnum(m["forecast_slope"]["ref"], "%+.5f") + " -> " + fnum(m["forecast_slope"]["recent"], "%+.5f")},
        {"Measure": "Top-3 net return, % per year, reference -> recent", "Value": fnum(m["top3_net_pct_per_year"]["ref"], "%+.1f") + " -> " + fnum(m["top3_net_pct_per_year"]["recent"], "%+.1f")},
        {"Measure": "Long/short net return, % per year, reference -> recent", "Value": fnum(m["ls_net_pct_per_year"]["ref"], "%+.1f") + " -> " + fnum(m["ls_net_pct_per_year"]["recent"], "%+.1f")},
        {"Measure": "Cost drag, % per year, reference -> recent", "Value": fnum(m["ls_cost_drag_pct_per_year"]["ref"], "%.1f") + " -> " + fnum(m["ls_cost_drag_pct_per_year"]["recent"], "%.1f")},
        {"Measure": "Turnover per period, reference -> recent", "Value": fnum(m["ls_turnover"]["ref"], "%.2f") + " -> " + fnum(m["ls_turnover"]["recent"], "%.2f")},
        {"Measure": "Long/short alpha vs the universe (last 2 years): beta, alpha t", "Value": fnum(la.get("beta"), "%+.2f") + ", " + fnum(la.get("alpha_t"), "%+.2f")},
        {"Measure": "Maximum drawdown of the long/short book", "Value": fnum(m["ls_max_drawdown"] * 100, "%.1f") + "%"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    reg = pd.DataFrame(m["regimes"])
    if not reg.empty:
        st.dataframe(reg.rename(columns={"regime": "Regime", "n_ref": "n (reference)", "ic_ref": "IC (reference)", "n_recent": "n (recent)",
                                         "ic_recent": "IC (recent)"}).style.format({"IC (reference)": "{:+.4f}", "IC (recent)": "{:+.4f}"}, na_rep="n/a"),
                     use_container_width=True, hide_index=True)
        st.caption("Reference vs recent IC inside each regime. A drop confined to one regime points to a regime change rather than a dead "
                   "signal — but with only tens of periods per cell this is evidence, not a verdict.")


def main():
    if "cache_bust" not in st.session_state:
        st.session_state.cache_bust = 0
    head, refresh = st.columns([5, 1])
    with head:
        st.markdown('<div class="app-title">📉 P2 Alpha-Decay-Detector</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Is a signal that worked still working — and how long would it take to know? '
                    'Every threshold is calibrated by simulation, and the last tab shows what the detector cannot see.</div>',
                    unsafe_allow_html=True)
    with refresh:
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.cache_bust += 1
            st.rerun()

    data, notes = load_data(_cache_key=str(st.session_state.cache_bust))
    if not data:
        st.error("No results found.")
        with st.expander("Why? (click for details)", expanded=True):
            for n in notes or ["No diagnostic information was returned."]:
                st.write("• " + n)
            st.write("Run `python trainer.py` to create a local `decay_results_*.json`, or check the HF results dataset and HF_TOKEN.")
        return

    st.caption("🕒 Results generated: **" + str(data.get("run_date", "Unknown")) + "**")
    if notes:
        st.caption("📡 " + notes[0])
    stt = data.get("settings", {})

    tab1, tab2, tab3, tab4 = st.tabs(["🩺 Signal health", "🎛️ Does the control layer help?", "📒 Engine picks", "🧪 What the detector can see"])

    with tab1:
        st.markdown('<div class="note">Each row is one signal at one horizon on non-overlapping periods. The first '
                    + ("%.0f" % stt.get("reference_years", 3)) + ' years set the reference level; a signal whose reference IC is not '
                    'significantly above zero is Unvalidated (nothing demonstrated to decay). Otherwise a CUSUM calibrated to a '
                    + ("%.0f" % (100 * stt.get("target_false_alarm", 0.1))) + '% false-alarm probability over '
                    + ("%.0f" % stt.get("fa_years", 3)) + ' years sets Active / Degrading / Suspended. "Underpowered" means that even a total '
                    'loss of the signal would be caught within that horizon less than half the time. The signals are research probes with '
                    'signs fixed in advance, not claims that they work.</div>', unsafe_allow_html=True)
        for uni, u in data.get("universes", {}).items():
            sm = u["summary"]
            st.markdown('<div class="universe-heading">' + uni.replace("_", " ").title() + '</div>', unsafe_allow_html=True)
            cnt = sm["status_counts"]
            cols = st.columns(5)
            for c, (lab, val) in zip(cols, [("Monitors", str(sm["n_monitors"])), ("Validated", str(sm["n_validated"])),
                                            ("Underpowered", str(sm["n_underpowered"])), ("Suspended", str(cnt.get("Suspended", 0))),
                                            ("Degrading", str(cnt.get("Degrading", 0)))]):
                with c:
                    st.markdown(kpi(lab, val), unsafe_allow_html=True)
            df = monitor_table(u["monitors"])
            if not df.empty:
                st.dataframe(df.style.format({"Reference IC": "{:+.4f}", "Ref t": "{:+.2f}", "Ref IR-equiv": "{:+.2f}", "IC last year": "{:+.4f}",
                                              "Change vs ref (z)": "{:+.2f}", "P(catch total loss)": "{:.0%}", "CUSUM / threshold": "{:.2f}"}, na_rep="n/a"),
                             use_container_width=True, hide_index=True)
            options = ["%s, %dd" % (m["signal"], m["horizon"]) for m in u["monitors"] if m["status"] != "Insufficient history"]
            if options:
                pick = st.selectbox("Inspect a monitor in " + uni.replace("_", " ").title(), options, key="sel_" + uni)
                m = [x for x in u["monitors"] if "%s, %dd" % (x["signal"], x["horizon"]) == pick][0]
                accent = STATUS_COLOR.get(m["status"], "#64748b")
                st.markdown('<div class="banner" style="--accent: ' + accent + ';"><b>' + m["status"] + '</b>'
                            + (" (underpowered)" if m.get("underpowered") else "") + ' — reference IC ' + fnum(m["reference"]["mu0"], "%+.4f")
                            + ' (t=' + fnum(m["reference"]["t"], "%+.2f") + ', IR-equivalent ' + fnum(m["reference"].get("ir_equivalent"), "%+.2f")
                            + '); last year ' + fnum(m["recent"]["ic_mean"], "%+.4f") + ' +/- ' + fnum(m["recent"]["ic_se"], "%.4f")
                            + '. A total loss would be caught within the false-alarm horizon with probability '
                            + fnum(100 * m["power_total_loss"]["p_detect"], "%.0f") + '%.</div>', unsafe_allow_html=True)
                render_monitor_detail(m, uni + pick)
                with st.expander("Evidence on temporary vs structural, cost erosion and alpha"):
                    render_evidence(m)
            st.markdown("<hr style='margin: 1.4rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="note">The tradable book is a dollar-neutral rank-weighted long/short of each signal, net of costs ('
                    + str(stt.get("cost_bps", 15)) + ' bps round trip on turnover plus short borrow), evaluated from the end of the reference '
                    'period. Gated = exposure 1 / 0.5 / 0 by state (0 for Unvalidated), decided with information through the previous period. '
                    'Naive = trade only if the trailing year\'s net return was positive. "P(better)" is a circular block bootstrap of the paired '
                    'difference in mean net return.</div>', unsafe_allow_html=True)
        rows = []
        for uni, u in data.get("universes", {}).items():
            for m in u["monitors"]:
                g = m.get("gating")
                if not g or not g.get("ungated"):
                    continue
                gb = g.get("gated_vs_ungated", {})
                rows.append({"Universe": uni, "Signal": m["signal"], "Horizon (d)": m["horizon"], "Status": m["status"],
                             "Always traded (%/yr)": g["ungated"]["mean_pct_per_year"], "Gated (%/yr)": g["gated"].get("mean_pct_per_year"),
                             "Naive filter (%/yr)": g["naive"].get("mean_pct_per_year"), "Gated avg exposure": g["gated"].get("avg_exposure"),
                             "Gated Sharpe": g["gated"].get("sharpe"), "Always-traded Sharpe": g["ungated"].get("sharpe"),
                             "P(gated better)": gb.get("p_mean_better")})
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df.style.format({"Always traded (%/yr)": "{:+.1f}", "Gated (%/yr)": "{:+.1f}", "Naive filter (%/yr)": "{:+.1f}",
                                          "Gated avg exposure": "{:.2f}", "Gated Sharpe": "{:+.2f}", "Always-traded Sharpe": "{:+.2f}",
                                          "P(gated better)": "{:.2f}"}, na_rep="n/a"), use_container_width=True, hide_index=True)
            st.caption("Most research signals have no demonstrated alpha, so most are Unvalidated and gated to zero; the gate then mostly "
                       "avoids cost drag. See the last tab for what each part of the control layer is worth in simulation.")

    with tab3:
        st.markdown('<div class="note">Picks the other engines archived in their own HF datasets, scored against realized returns '
                    '(picks minus the equal-weight universe over the pick\'s horizon, net of costs, non-overlapping). Only top picks are '
                    'archived, so there is no IC. A decay verdict needs about three years of history; until then this is a record, not a monitor.</div>',
                    unsafe_allow_html=True)
        eng = data.get("engines", {})
        rows = []
        for name, r in eng.items():
            if not r.get("series"):
                rows.append({"Engine": name, "Series": "-", "Periods scored": 0, "Mean net / period (bps)": None, "t": None, "Since": None,
                             "Status": r.get("status")})
            for key, s in r.get("series", {}).items():
                uni, h = key.split("|")
                rows.append({"Engine": name, "Series": uni + ", " + h + "d", "Periods scored": s["n_periods"],
                             "Mean net / period (bps)": s["net_mean"] * 1e4, "t": s["net_t"], "Since": s["first_date"],
                             "Status": "insufficient history" if s["n_periods"] < 60 else "monitorable"})
        if rows:
            st.dataframe(pd.DataFrame(rows).style.format({"Mean net / period (bps)": "{:+.1f}", "t": "{:+.2f}"}, na_rep="n/a"),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No engines are configured or none returned scoreable picks.")

    with tab4:
        v = data.get("validation") or {}
        st.markdown('<div class="note">Simulated cross-sectional histories (' + str(v.get("N", "?")) + ' ETFs, weekly, ' + str(v.get("reps", "?"))
                    + ' per scenario) go through the exact production pipeline. Skill is expressed as an annualized information ratio '
                    '(IR), which is what detection power actually depends on.</div>', unsafe_allow_html=True)
        pw = v.get("power", [])
        if pw:
            st.markdown("###### If a signal dies completely, how often is it caught within 3 years (10% false-alarm probability)?")
            dfp = pd.DataFrame(pw)
            dfp["Horizon"] = dfp["horizon"].astype(str) + "d, " + dfp["N"].astype(str) + " ETFs"
            piv = dfp.pivot_table(index="Horizon", columns="ir", values="p_detect")
            piv.columns = ["IR " + str(c) for c in piv.columns]
            st.dataframe(piv.style.format("{:.0%}"), use_container_width=True)
            st.caption("The answer depends on the signal's annualized IR, not on the horizon or universe size: a signal with an IR of 1 that "
                       "dies is caught within three years well under half the time, with a median delay of nearly two years when caught. "
                       "That is an information limit — a few hundred noisy rank correlations — not something tuning can fix.")
        sc = v.get("scenarios", [])
        if sc:
            st.markdown("###### What the control layer does to returns, by kind of decay (% of capital over 4 years)")
            rows = [{"Scenario": r["label"], "Passes validation": r["p_validated"], "Detector alarms": r["p_alarm_given_validated"],
                     "Median delay (years)": r["median_delay_years"], "Always traded": r["net_ungated_pct"],
                     "Validation gate only": r.get("net_gate_only_pct"), "Gate + detector": r["net_gated_pct"], "Naive filter": r["net_naive_pct"],
                     "P(gate+detector better)": r["p_gated_better"]} for r in sc]
            st.dataframe(pd.DataFrame(rows).style.format({"Passes validation": "{:.0%}", "Detector alarms": "{:.0%}", "Median delay (years)": "{:.2f}",
                                                          "Always traded": "{:+.1f}", "Validation gate only": "{:+.1f}", "Gate + detector": "{:+.1f}",
                                                          "Naive filter": "{:+.1f}", "P(gate+detector better)": "{:.0%}"}, na_rep="-"),
                         use_container_width=True, hide_index=True)
            st.caption("Reading it: the validation gate (never trade what has not shown skill) does most of the work when there is no skill and "
                       "costs real-but-modest signals; the detector adds most when a signal INVERTS (crowded out), a little when it dies or "
                       "decays, and gives some back when a signal merely dips and recovers or is stable. A plain trailing-year filter does "
                       "about as well on net returns; the detector's advantage is a known false-alarm rate, not more money.")


if __name__ == "__main__":
    main()
