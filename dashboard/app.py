"""
Streamlit process monitoring dashboard for the gas turbine soft sensor.

Reads prediction CSVs from results/:
  results/xgboost_predictions.csv  — XGBoost + MAPIE
  results/pinn_predictions.csv     — calibrated PINN + MC Dropout

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
    page_title="Soft Sensor Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLR = {
    "bg":      "#0f1117",
    "surface": "#1e1e2e",
    "accent":  "#00d4aa",
    "amber":   "#f7b731",
    "red":     "#ff6b6b",
    "purple":  "#a78bfa",
    "text":    "#e0e0e0",
    "subtext": "#888899",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CLR["bg"],
    plot_bgcolor=CLR["surface"],
    font=dict(color=CLR["text"], family="monospace"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#2a2a3a", showgrid=True),
    yaxis=dict(gridcolor="#2a2a3a", showgrid=True),
)

st.markdown(f"""
<style>
  .stApp {{ background-color: {CLR['bg']}; color: {CLR['text']}; }}
  .block-container {{ padding-top: 1.5rem; }}
  div[data-testid="metric-container"] {{
      background: {CLR['surface']};
      border: 1px solid #2a2a3a;
      border-radius: 8px;
      padding: 12px 18px;
  }}
  div[data-testid="metric-container"] label {{
      color: {CLR['subtext']} !important;
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
  }}
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
      color: {CLR['accent']} !important;
      font-size: 1.8rem;
      font-family: monospace;
  }}
  .alarm-red   {{ background: rgba(255,107,107,0.15); border-left: 4px solid {CLR['red']};
                  padding: 8px 14px; border-radius: 4px; color: {CLR['red']};
                  font-family: monospace; }}
  .alarm-amber {{ background: rgba(247,183,49,0.12);  border-left: 4px solid {CLR['amber']};
                  padding: 8px 14px; border-radius: 4px; color: {CLR['amber']};
                  font-family: monospace; }}
  .alarm-ok    {{ background: rgba(0,212,170,0.10);   border-left: 4px solid {CLR['accent']};
                  padding: 8px 14px; border-radius: 4px; color: {CLR['accent']};
                  font-family: monospace; }}
  h1, h2, h3 {{ color: {CLR['accent']} !important; font-family: monospace; }}
  .stSidebar {{ background-color: {CLR['surface']} !important; }}
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
        ("PINN + MC Dropout (calibrated)",  "pinn_predictions.csv"),
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
        datasets["PINN + MC Dropout (calibrated, simulated)"] = fallback.copy()

    return datasets


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(datasets: dict) -> dict:
    st.sidebar.markdown("## ⚙️ Soft Sensor Monitor")
    st.sidebar.markdown("---")

    model  = st.sidebar.selectbox("Model", list(datasets.keys()))
    target = st.sidebar.selectbox("Target variable", ["CO", "NOx"])

    st.sidebar.markdown("### Alarm thresholds")
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

    st.sidebar.markdown("### Playback")
    window = st.sidebar.slider("Display window (samples)", 50, 500, 200,
                                help="Number of recent samples shown in the live chart")
    live   = st.sidebar.toggle("Live simulation", value=False,
                                help="Step through the test set one sample at a time")
    speed  = st.sidebar.slider("Refresh interval (s)", 0.1, 2.0, 0.5, step=0.1,
                                disabled=not live)

    return dict(model=model, target=target,
                warn_thresh=warn_thresh, alarm_thresh=alarm_thresh,
                window=window, live=live, speed=speed)


# ─────────────────────────────────────────────────────────────────────────────
#  KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────

def render_kpis(df: pd.DataFrame, target: str,
                warn_thresh: float, alarm_thresh: float, idx: int):
    pred  = df[f"y_pred_{target}"].iloc[idx]
    lo    = df[f"pi_lo_{target}"].iloc[idx]
    hi    = df[f"pi_hi_{target}"].iloc[idx]
    true  = df[f"y_true_{target}"].iloc[idx]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Predicted {target}", f"{pred:.2f} mg/m³")
    col2.metric("Interval half-width", f"± {(hi - lo) / 2:.2f} mg/m³")
    col3.metric("Absolute error",      f"{abs(pred - true):.2f} mg/m³")
    col4.metric("True value",          f"{true:.2f} mg/m³")

    if pred >= alarm_thresh:
        st.markdown(
            f'<div class="alarm-red">🔴 ALARM — {target} = {pred:.2f} mg/m³ '
            f'exceeds alarm threshold ({alarm_thresh:.1f})</div>',
            unsafe_allow_html=True,
        )
    elif pred >= warn_thresh:
        st.markdown(
            f'<div class="alarm-amber">🟡 WARNING — {target} = {pred:.2f} mg/m³ '
            f'exceeds warning threshold ({warn_thresh:.1f})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="alarm-ok">✓ NORMAL — {target} within operating limits</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def render_timeseries(df: pd.DataFrame, target: str, window: int,
                      warn_thresh: float, alarm_thresh: float, end_idx: int):
    start = max(0, end_idx - window)
    sl    = df.iloc[start:end_idx]
    x     = np.arange(start, end_idx)

    fig = go.Figure()

    # Filled nominal 95% uncertainty band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([sl[f"pi_hi_{target}"], sl[f"pi_lo_{target}"][::-1]]),
        fill="toself", fillcolor="rgba(0,212,170,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=sl[f"y_true_{target}"],
        mode="lines", line=dict(color=CLR["red"], width=1.0),
        name=f"True {target}",
        hovertemplate=f"True: %{{y:.2f}} mg/m³<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=sl[f"y_pred_{target}"],
        mode="lines", line=dict(color=CLR["accent"], width=1.5),
        name=f"Predicted {target}",
        hovertemplate=f"Pred: %{{y:.2f}} mg/m³<extra></extra>",
    ))
    fig.add_hline(y=alarm_thresh, line=dict(color=CLR["red"], width=1, dash="dash"),
                  annotation_text="Alarm", annotation_font_color=CLR["red"])
    fig.add_hline(y=warn_thresh,  line=dict(color=CLR["amber"], width=1, dash="dot"),
                  annotation_text="Warning", annotation_font_color=CLR["amber"])

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"{target} — Soft Sensor Prediction with 95% Calibrated Interval",
            font=dict(color=CLR["accent"], size=13),
        ),
        yaxis_title=f"{target} [mg/m³]",
        xaxis_title="Sample index (test set)",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=CLR["text"])),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_bottom_row(df: pd.DataFrame, target: str, end_idx: int):
    sl           = df.iloc[:end_idx]
    col_left, col_right = st.columns(2)

    with col_left:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=sl[f"y_true_{target}"], y=sl[f"y_pred_{target}"],
            mode="markers",
            marker=dict(color=CLR["accent"], size=3, opacity=0.3),
            hovertemplate="True: %{x:.2f}<br>Pred: %{y:.2f}<extra></extra>",
        ))
        lo = sl[f"y_true_{target}"].min()
        hi = sl[f"y_true_{target}"].max()
        fig_scatter.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines", line=dict(color=CLR["red"], dash="dash", width=1.5),
            name="Perfect fit",
        ))
        fig_scatter.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text=f"{target} — Predicted vs Actual",
                       font=dict(color=CLR["accent"], size=12)),
            xaxis_title=f"True {target} [mg/m³]",
            yaxis_title=f"Predicted {target} [mg/m³]",
            height=300,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

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
                text=f"{target} — 95% Interval Width over Time",
                font=dict(color=CLR["accent"], size=12),
            ),
            yaxis_title="PI width [mg/m³]",
            xaxis_title="Sample index",
            height=300,
        )
        st.plotly_chart(fig_width, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown("# ⚙️ Soft Sensor Monitor")
    st.markdown(
        f'<span style="color:{CLR["subtext"]}; font-family:monospace; font-size:0.85rem;">'
        "UCI Gas Turbine · CO & NOx · XGBoost/MAPIE + calibrated PINN/MC-Dropout</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    datasets = load_predictions()
    cfg      = render_sidebar(datasets)
    df       = datasets[cfg["model"]]
    n        = len(df)

    if "sim_idx" not in st.session_state:
        st.session_state.sim_idx = cfg["window"]

    if cfg["live"]:
        # Render one frame per Streamlit run, then rerun. This keeps sidebar
        # controls responsive instead of blocking the app in a permanent loop.
        idx = min(max(st.session_state.sim_idx, cfg["window"]), n)
        render_kpis(df, cfg["target"], cfg["warn_thresh"],
                    cfg["alarm_thresh"], idx - 1)
        render_timeseries(df, cfg["target"], cfg["window"],
                          cfg["warn_thresh"], cfg["alarm_thresh"], idx)
        render_bottom_row(df, cfg["target"], idx)
        st.session_state.sim_idx = idx + 1 if idx < n else cfg["window"]
        time.sleep(cfg["speed"])
        st.rerun()
    else:
        idx = st.slider("Sample position", cfg["window"], n, n,
                        help="Drag to inspect any point in the test set")
        render_kpis(df, cfg["target"], cfg["warn_thresh"],
                    cfg["alarm_thresh"], idx - 1)
        render_timeseries(df, cfg["target"], cfg["window"],
                          cfg["warn_thresh"], cfg["alarm_thresh"], idx)
        render_bottom_row(df, cfg["target"], idx)


if __name__ == "__main__":
    main()
