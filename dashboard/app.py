"""
Streamlit process monitoring dashboard for the gas turbine soft sensor.

Reads prediction CSVs from results/:
  results/xgboost_predictions.csv  — XGBoost + MAPIE
  results/pinn_predictions.csv     — PINN + MC Dropout

Falls back to synthetic data if neither file exists, so the dashboard
can be explored without running the full training pipeline first.

Usage:
  pip install streamlit plotly pandas numpy
  streamlit run dashboard/app.py

Alarm logic:
  RED   — predicted value exceeds alarm_hi threshold
  AMBER — predicted value exceeds warn_lo threshold
  Thresholds are adjustable via the sidebar at runtime.

  Debug: if alarms never fire, verify threshold units match mg/m³ and
  that the selected target (CO vs NOx) matches the threshold range.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Turbine Emissions Intelligence",
    page_icon="GT",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLR = {
    "bg":      "#07111f",
    "surface": "#0d1b2a",
    "surface2":"#112338",
    "border":  "#1f3852",
    "accent":  "#23c4a8",
    "blue":    "#4da3ff",
    "amber":   "#f2b84b",
    "red":     "#f06464",
    "purple":  "#9c8cff",
    "text":    "#f2f6fa",
    "subtext": "#91a4b7",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CLR["bg"],
    plot_bgcolor=CLR["surface"],
    font=dict(color=CLR["subtext"], family="Inter, Segoe UI, sans-serif", size=12),
    margin=dict(l=48, r=24, t=56, b=44),
    xaxis=dict(gridcolor="#193047", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#193047", showgrid=True, zeroline=False),
    hoverlabel=dict(bgcolor=CLR["surface2"], bordercolor=CLR["border"], font_color=CLR["text"]),
)

st.markdown(f"""
<style>
  .stApp {{ background: radial-gradient(circle at 80% -10%, #102b43 0, {CLR['bg']} 34%); color: {CLR['text']}; }}
  .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  #MainMenu, footer {{ visibility: hidden; }}
  div[data-testid="metric-container"] {{
      background: linear-gradient(145deg, {CLR['surface2']}, {CLR['surface']});
      border: 1px solid {CLR['border']};
      border-radius: 12px;
      padding: 16px 18px;
      min-height: 108px;
      box-shadow: 0 12px 32px rgba(0,0,0,.14);
  }}
  div[data-testid="metric-container"] label {{
      color: {CLR['subtext']} !important;
      font-size: 0.7rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      font-weight: 650;
  }}
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
      color: {CLR['accent']} !important;
      font-size: 1.65rem;
      font-family: "Segoe UI", sans-serif;
      font-weight: 650;
  }}
  .status-card {{ display:flex; align-items:center; justify-content:space-between;
                  padding: 13px 17px; margin: 14px 0 20px; border-radius: 10px;
                  border: 1px solid; font-size: .88rem; font-weight: 600; }}
  .status-card span:last-child {{ font-weight: 450; opacity: .88; }}
  .alarm-red   {{ background: rgba(240,100,100,.10); border-color: rgba(240,100,100,.45); color: {CLR['red']}; }}
  .alarm-amber {{ background: rgba(242,184,75,.10); border-color: rgba(242,184,75,.45); color: {CLR['amber']}; }}
  .alarm-ok    {{ background: rgba(35,196,168,.09); border-color: rgba(35,196,168,.38); color: {CLR['accent']}; }}
  .eyebrow {{ color:{CLR['accent']}; letter-spacing:.14em; text-transform:uppercase;
              font-size:.69rem; font-weight:750; margin-bottom:.35rem; }}
  .hero-title {{ color:{CLR['text']}; font-size:2rem; line-height:1.15; font-weight:680;
                 letter-spacing:-.03em; margin:0; }}
  .hero-copy {{ color:{CLR['subtext']}; margin-top:.45rem; font-size:.9rem; }}
  .live-pill {{ display:inline-flex; align-items:center; gap:7px; float:right;
                color:{CLR['accent']}; background:rgba(35,196,168,.08);
                border:1px solid rgba(35,196,168,.3); border-radius:99px;
                padding:7px 12px; font-size:.72rem; font-weight:700; letter-spacing:.06em; }}
  .live-dot {{ width:7px; height:7px; border-radius:50%; background:{CLR['accent']};
               box-shadow:0 0 0 4px rgba(35,196,168,.12); }}
  .drift-note {{ margin-top:.75rem; padding:10px 12px; border-radius:8px;
                 color:{CLR['subtext']}; background:rgba(77,163,255,.055);
                 border:1px solid rgba(77,163,255,.18); font-size:.76rem;
                 line-height:1.45; }}
  .drift-note strong {{ display:block; color:#b8c9d9; font-size:.68rem;
                        letter-spacing:.09em; text-transform:uppercase;
                        margin-bottom:3px; }}
  .ops-panel {{ background:{CLR['surface']}; border:1px solid {CLR['border']};
                border-radius:12px; padding:17px 18px; min-height:190px; }}
  .panel-label {{ color:{CLR['text']}; font-size:.88rem; font-weight:700;
                  margin-bottom:13px; }}
  .health-row {{ display:flex; justify-content:space-between; padding:9px 0;
                 border-bottom:1px solid rgba(31,56,82,.7); color:{CLR['subtext']};
                 font-size:.8rem; }}
  .health-row:last-child {{ border-bottom:0; }}
  .health-value {{ color:{CLR['text']}; font-weight:650; }}
  h1, h2, h3 {{ color: {CLR['text']} !important; font-family: "Segoe UI", sans-serif; }}
  section[data-testid="stSidebar"] {{ background: {CLR['surface']} !important;
                                      border-right: 1px solid {CLR['border']}; }}
  section[data-testid="stSidebar"] .stMarkdown h2 {{ font-size:1.05rem; }}
  div[data-testid="stPlotlyChart"] {{ border:1px solid {CLR['border']}; border-radius:12px;
                                      overflow:hidden; box-shadow:0 12px 32px rgba(0,0,0,.12); }}
  hr {{ border-color:{CLR['border']} !important; opacity:.75; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_predictions() -> dict:
    """
    Load prediction CSVs from results/. Falls back to synthetic Gaussian walk
    data if files are missing, so the dashboard is explorable standalone.
    """
    base     = Path(__file__).parent.parent / "results"
    datasets = {}

    for label, filename in [
        ("XGBoost + MAPIE",    "xgboost_predictions.csv"),
        ("PINN + MC Dropout",  "pinn_predictions.csv"),
    ]:
        path = base / filename
        if path.exists():
            datasets[label] = pd.read_csv(path)

    if not datasets:
        # Synthetic fallback — Gaussian random walk for UI testing
        n   = 2000
        rng = np.random.default_rng(0)
        co_true  = np.clip(2 + rng.normal(0, 0.5, n).cumsum() * 0.03, 0, 20)
        co_pred  = co_true + rng.normal(0, 0.3, n)
        co_std   = np.abs(rng.normal(0.8, 0.2, n))
        nox_true = np.clip(65 + rng.normal(0, 2, n).cumsum() * 0.05, 30, 110)
        nox_pred = nox_true + rng.normal(0, 2, n)
        nox_std  = np.abs(rng.normal(4, 1, n))
        fallback = pd.DataFrame({
            "y_true_CO":  co_true,   "y_pred_CO":  co_pred,
            "pi_lo_CO":   co_pred - 1.96 * co_std,
            "pi_hi_CO":   co_pred + 1.96 * co_std,
            "y_true_NOx": nox_true,  "y_pred_NOx": nox_pred,
            "pi_lo_NOx":  nox_pred - 1.96 * nox_std,
            "pi_hi_NOx":  nox_pred + 1.96 * nox_std,
        })
        datasets["XGBoost + MAPIE (simulated)"]   = fallback
        datasets["PINN + MC Dropout (simulated)"] = fallback.copy()

    return datasets


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(datasets: dict) -> dict:
    st.sidebar.markdown("## Turbine Intelligence")
    st.sidebar.caption("EMISSIONS CONTROL CENTER")
    st.sidebar.divider()

    st.sidebar.markdown("### Signal configuration")
    model  = st.sidebar.selectbox("Inference model", list(datasets.keys()))
    target = st.sidebar.segmented_control(
        "Emission channel", ["CO", "NOx"], default="CO"
    )

    st.sidebar.markdown("### Operating limits")
    defaults     = {"CO": (5.0, 10.0), "NOx": (80.0, 100.0)}
    warn_lo, alarm_hi = defaults[target]

    warn_thresh = st.sidebar.slider(
        "Warning threshold", 0.0, float(alarm_hi), float(warn_lo), step=0.1,
        help="Amber alarm fires above this value",
    )
    alarm_default = max(float(alarm_hi), float(warn_thresh))
    alarm_thresh = st.sidebar.slider(
        "Alarm threshold", float(warn_thresh), float(alarm_hi * 2), alarm_default,
        step=0.1, help="Red alarm fires above this value",
    )

    st.sidebar.markdown("### Timeline")
    window = st.sidebar.slider("Display window (samples)", 50, 500, 200,
                                help="Number of recent samples shown in the live chart")
    live   = st.sidebar.toggle("Live simulation", value=False,
                                help="Step through the test set one sample at a time")
    show_reference = st.sidebar.toggle(
        "Show reference signal", value=False,
        help="Overlay held-out measurements for model review",
    )
    speed  = st.sidebar.slider("Refresh interval (s)", 0.1, 2.0, 0.5, step=0.1,
                                disabled=not live)

    st.sidebar.divider()
    if target == "NOx":
        st.sidebar.markdown(
            '<div class="drift-note"><strong>Model note</strong>'
            'NOx shifted in the 2014–2015 holdout. Interpret wider intervals '
            'as lower model confidence.</div>',
            unsafe_allow_html=True,
        )
    st.sidebar.caption("Dataset · UCI GT 2011–2015\n\nUnits · mg/m³")

    return dict(model=model, target=target,
                warn_thresh=warn_thresh, alarm_thresh=alarm_thresh,
                window=window, live=live, speed=speed,
                show_reference=show_reference)


# ─────────────────────────────────────────────────────────────────────────────
#  KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────

def render_kpis(df: pd.DataFrame, target: str,
                warn_thresh: float, alarm_thresh: float, idx: int):
    pred  = df[f"y_pred_{target}"].iloc[idx]
    lo    = df[f"pi_lo_{target}"].iloc[idx]
    hi    = df[f"pi_hi_{target}"].iloc[idx]
    width = hi - lo
    widths = df[f"pi_hi_{target}"] - df[f"pi_lo_{target}"]
    width_percentile = float((widths <= width).mean() * 100)
    headroom = warn_thresh - pred

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Predicted {target}", f"{pred:.2f} mg/m³")
    col2.metric("Calibrated range", f"{lo:.2f} – {hi:.2f}")
    col3.metric("Warning headroom", f"{headroom:+.2f} mg/m³")
    col4.metric("Interval percentile", f"P{width_percentile:.0f}")

    if pred >= alarm_thresh:
        st.markdown(
            f'<div class="status-card alarm-red"><strong>CRITICAL · {target} LIMIT EXCEEDED</strong>'
            f'<span>{pred:.2f} mg/m³ · limit {alarm_thresh:.1f}</span></div>',
            unsafe_allow_html=True,
        )
    elif pred >= warn_thresh:
        st.markdown(
            f'<div class="status-card alarm-amber"><strong>WARNING · ELEVATED {target}</strong>'
            f'<span>{pred:.2f} mg/m³ · threshold {warn_thresh:.1f}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-card alarm-ok"><strong>NORMAL OPERATION</strong>'
            f'<span>{target} remains within configured limits</span></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def render_timeseries(df: pd.DataFrame, target: str, window: int,
                      warn_thresh: float, alarm_thresh: float, end_idx: int,
                      show_reference: bool = False):
    start = max(0, end_idx - window)
    sl    = df.iloc[start:end_idx]
    x     = np.arange(start, end_idx)

    fig = go.Figure()

    # Filled uncertainty band (95% PI)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([sl[f"pi_hi_{target}"], sl[f"pi_lo_{target}"][::-1]]),
        fill="toself", fillcolor="rgba(77,163,255,0.14)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% PI", hoverinfo="skip",
    ))
    if show_reference:
        fig.add_trace(go.Scatter(
            x=x, y=sl[f"y_true_{target}"],
            mode="lines", line=dict(color=CLR["text"], width=1.1),
            name="Reference",
            hovertemplate=f"True: %{{y:.2f}} mg/m³<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=x, y=sl[f"y_pred_{target}"],
        mode="lines", line=dict(color=CLR["blue"], width=2.0),
        name="Soft-sensor estimate",
        hovertemplate=f"Pred: %{{y:.2f}} mg/m³<extra></extra>",
    ))
    fig.add_hline(y=alarm_thresh, line=dict(color=CLR["red"], width=1, dash="dash"),
                  annotation_text="Alarm", annotation_font_color=CLR["red"])
    fig.add_hline(y=warn_thresh,  line=dict(color=CLR["amber"], width=1, dash="dot"),
                  annotation_text="Warning", annotation_font_color=CLR["amber"])

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"{target} trend and prediction interval",
            font=dict(color=CLR["text"], size=15), x=0.025,
        ),
        yaxis_title=f"{target} [mg/m³]",
        xaxis_title="Sample index (test set)",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=CLR["text"])),
        height=380,
    )
    st.plotly_chart(fig, width="stretch")


def render_bottom_row(df: pd.DataFrame, target: str, end_idx: int):
    sl           = df.iloc[:end_idx]
    col_left, col_right = st.columns(2)

    with col_left:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=sl[f"y_true_{target}"], y=sl[f"y_pred_{target}"],
            mode="markers",
            marker=dict(color=CLR["blue"], size=4, opacity=0.32),
            hovertemplate="True: %{x:.2f}<br>Pred: %{y:.2f}<extra></extra>",
        ))
        lo = sl[f"y_true_{target}"].min()
        hi = sl[f"y_true_{target}"].max()
        fig_scatter.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines", line=dict(color=CLR["subtext"], dash="dash", width=1.2),
            name="Perfect fit",
        ))
        fig_scatter.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text=f"{target} estimate fidelity",
                       font=dict(color=CLR["text"], size=14), x=0.04),
            xaxis_title=f"True {target} [mg/m³]",
            yaxis_title=f"Predicted {target} [mg/m³]",
            height=320,
        )
        st.plotly_chart(fig_scatter, width="stretch")

    with col_right:
        # Rolling mean of PI width highlights confidence drift over time.
        # A rising trend may indicate distribution shift in sensor readings.
        widths       = sl[f"pi_hi_{target}"] - sl[f"pi_lo_{target}"]
        rolling_mean = widths.rolling(window=20, min_periods=1).mean()
        x            = np.arange(len(sl))

        fig_width = go.Figure()
        fig_width.add_trace(go.Scatter(
            x=x, y=widths,
            mode="lines", line=dict(color=CLR["purple"], width=0.8),
            name="PI width", opacity=0.5,
        ))
        fig_width.add_trace(go.Scatter(
            x=x, y=rolling_mean,
            mode="lines", line=dict(color=CLR["amber"], width=1.8),
            name="Rolling mean (20)",
        ))
        fig_width.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(
                text=f"{target} uncertainty trend",
                font=dict(color=CLR["text"], size=14), x=0.04,
            ),
            yaxis_title="PI width [mg/m³]",
            xaxis_title="Sample index",
            height=320,
        )
        st.plotly_chart(fig_width, width="stretch")


def render_control_summary(df: pd.DataFrame, target: str, end_idx: int,
                           warn_thresh: float, alarm_thresh: float):
    """Compact operator summary; detailed validation charts stay off the console."""
    sl = df.iloc[:end_idx]
    recent = sl.tail(min(500, len(sl))).copy()
    pred_col = f"y_pred_{target}"
    width = recent[f"pi_hi_{target}"] - recent[f"pi_lo_{target}"]
    warning_count = int(((recent[pred_col] >= warn_thresh) &
                         (recent[pred_col] < alarm_thresh)).sum())
    alarm_count = int((recent[pred_col] >= alarm_thresh).sum())
    coverage = float(((recent[f"y_true_{target}"] >= recent[f"pi_lo_{target}"]) &
                      (recent[f"y_true_{target}"] <= recent[f"pi_hi_{target}"])).mean())

    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        st.markdown("#### Recent limit events")
        events = recent.loc[recent[pred_col] >= warn_thresh, [pred_col]].tail(5).copy()
        if events.empty:
            st.markdown(
                '<div class="ops-panel"><div class="panel-label">Event queue</div>'
                '<div style="color:#91a4b7;font-size:.82rem;padding-top:35px;'
                'text-align:center">No warning or alarm events in the recent window</div></div>',
                unsafe_allow_html=True,
            )
        else:
            events.insert(0, "Sample", events.index)
            events["Severity"] = np.where(
                events[pred_col] >= alarm_thresh, "ALARM", "WARNING"
            )
            events.rename(columns={pred_col: f"{target} (mg/m³)"}, inplace=True)
            events[f"{target} (mg/m³)"] = events[f"{target} (mg/m³)"].round(2)
            st.dataframe(events.iloc[::-1], hide_index=True, width="stretch", height=190)

    with right:
        reliability = "Nominal" if coverage >= 0.90 else "Review"
        st.markdown(
            '<div class="ops-panel"><div class="panel-label">System health · recent 500</div>'
            f'<div class="health-row"><span>Prediction coverage</span>'
            f'<span class="health-value">{coverage:.1%}</span></div>'
            f'<div class="health-row"><span>Median interval width</span>'
            f'<span class="health-value">{width.median():.2f} mg/m³</span></div>'
            f'<div class="health-row"><span>Warning events</span>'
            f'<span class="health-value">{warning_count}</span></div>'
            f'<div class="health-row"><span>Alarm events</span>'
            f'<span class="health-value">{alarm_count}</span></div>'
            f'<div class="health-row"><span>Model status</span>'
            f'<span class="health-value">{reliability}</span></div></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown(
        '<div class="live-pill"><span class="live-dot"></span>MONITORING ONLINE</div>'
        '<div class="eyebrow">Operations / Emissions intelligence</div>'
        '<div class="hero-title">Gas Turbine Soft Sensor</div>'
        '<div class="hero-copy">Model-assisted CO and NOx monitoring with calibrated uncertainty</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    datasets = load_predictions()
    cfg      = render_sidebar(datasets)
    df       = datasets[cfg["model"]]
    n        = len(df)

    st.caption(
        f"ACTIVE MODEL  ·  {cfg['model'].upper()}     |     "
        f"CHANNEL  ·  {cfg['target']}     |     RECORDS  ·  {n:,}"
    )

    if "sim_idx" not in st.session_state:
        st.session_state.sim_idx = cfg["window"]

    if cfg["live"]:
        # Render one frame per Streamlit run, then rerun. This keeps sidebar
        # controls responsive instead of blocking the app in a permanent loop.
        idx = min(max(st.session_state.sim_idx, cfg["window"]), n)
        render_kpis(df, cfg["target"], cfg["warn_thresh"],
                    cfg["alarm_thresh"], idx - 1)
        render_timeseries(df, cfg["target"], cfg["window"],
                          cfg["warn_thresh"], cfg["alarm_thresh"], idx,
                          cfg["show_reference"])
        render_control_summary(df, cfg["target"], idx,
                               cfg["warn_thresh"], cfg["alarm_thresh"])
        st.session_state.sim_idx = idx + 1 if idx < n else cfg["window"]
        time.sleep(cfg["speed"])
        st.rerun()
    else:
        idx = st.slider("Historical position", cfg["window"], n, n,
                        help="Drag to inspect any point in the test set")
        render_kpis(df, cfg["target"], cfg["warn_thresh"],
                    cfg["alarm_thresh"], idx - 1)
        render_timeseries(df, cfg["target"], cfg["window"],
                          cfg["warn_thresh"], cfg["alarm_thresh"], idx,
                          cfg["show_reference"])
        render_control_summary(df, cfg["target"], idx,
                               cfg["warn_thresh"], cfg["alarm_thresh"])


if __name__ == "__main__":
    main()
