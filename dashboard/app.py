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
  Thresholds are adjustable through Model & display settings at runtime.

  Debug: if alarms never fire, verify threshold units match mg/m³ and
  that the selected target (CO vs NOx) matches the threshold range.
"""

import sys
import base64
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Turbine Emissions Intelligence",
    page_icon="GT",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CLR = {
    "bg":      "#e8edf2",
    "surface": "#f8fafc",
    "surface2":"#eef2f6",
    "border":  "#c7d0da",
    "accent":  "#238636",
    "blue":    "#1f6feb",
    "amber":   "#b7791f",
    "red":     "#c53030",
    "purple":  "#7656a8",
    "text":    "#17212b",
    "subtext": "#5f6b78",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CLR["bg"],
    plot_bgcolor=CLR["surface"],
    font=dict(color=CLR["subtext"], family="Inter, Segoe UI, sans-serif", size=14),
    margin=dict(l=48, r=24, t=56, b=44),
    xaxis=dict(gridcolor="#d8e0e8", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#d8e0e8", showgrid=True, zeroline=False),
    hoverlabel=dict(bgcolor=CLR["surface2"], bordercolor=CLR["border"], font_color=CLR["text"]),
)

st.markdown(f"""
<style>
  .stApp {{ background: {CLR['bg']}; color: {CLR['text']}; }}
  .stApp, .stApp button, .stApp input, .stApp select {{ font-size:16px; }}
  .block-container {{
      width:100% !important;
      max-width:none !important;
      padding:1.4rem clamp(.85rem, 1.5vw, 2rem) 2.5rem !important;
  }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stAppDeployButton"], .stDeployButton {{ display:none !important; }}
  [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display:none !important; }}
  #MainMenu, footer {{ visibility: hidden; }}
  div[data-testid="metric-container"] {{
      background: {CLR['surface']};
      border: 1px solid {CLR['border']};
      border-radius: 6px;
      padding: 13px 16px;
      min-height: 96px;
      box-shadow: 0 2px 5px rgba(23,33,43,.09), inset 0 1px 0 rgba(255,255,255,.9);
  }}
  div[data-testid="metric-container"] label {{
      color: {CLR['subtext']} !important;
      font-size: 0.8rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      font-weight: 650;
  }}
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
      color: {CLR['text']} !important;
      font-size: 1.65rem;
      font-family: "Segoe UI", sans-serif;
      font-weight: 650;
  }}
  .status-card {{ display:flex; align-items:center; justify-content:space-between;
                  padding: 13px 17px; margin: 14px 0 20px; border-radius: 10px;
                  border: 1px solid; font-size: .96rem; font-weight: 600; }}
  .status-card span:last-child {{ font-weight: 450; opacity: .88; }}
  .alarm-red   {{ background:#fff1f1; border-color:#e5a0a0; color:{CLR['red']}; }}
  .alarm-amber {{ background:#fff8e6; border-color:#dfc27b; color:{CLR['amber']}; }}
  .alarm-ok    {{ background:#edf8ef; border-color:#9bc9a3; color:{CLR['accent']}; }}
  .eyebrow {{ color:{CLR['accent']}; letter-spacing:.14em; text-transform:uppercase;
              font-size:.78rem; font-weight:750; margin-bottom:.35rem; }}
  .hero-title {{ color:{CLR['text']}; font-size:2rem; line-height:1.15; font-weight:680;
                 letter-spacing:-.03em; margin:0; }}
  .hero-copy {{ color:{CLR['subtext']}; margin-top:.45rem; font-size:1rem; }}
  .live-pill {{ display:inline-flex; align-items:center; gap:7px; float:right;
                color:{CLR['accent']}; background:#edf8ef;
                border:1px solid #9bc9a3; border-radius:99px;
                padding:7px 12px; font-size:.8rem; font-weight:700; letter-spacing:.06em; }}
  .live-dot {{ width:7px; height:7px; border-radius:50%; background:{CLR['accent']};
               box-shadow:0 0 0 4px rgba(35,196,168,.12); }}
  .drift-note {{ margin-top:.75rem; padding:10px 12px; border-radius:6px;
                 color:{CLR['subtext']}; background:#eef5ff;
                 border:1px solid #b7cff2; font-size:.84rem;
                 line-height:1.45; }}
  .drift-note strong {{ display:block; color:#526477; font-size:.75rem;
                        letter-spacing:.09em; text-transform:uppercase;
                        margin-bottom:3px; }}
  .ops-panel {{ background:{CLR['surface']}; border:1px solid {CLR['border']};
                border-radius:6px; padding:17px 18px; min-height:190px;
                box-shadow:0 2px 5px rgba(23,33,43,.07); }}
  .panel-label {{ color:{CLR['text']}; font-size:.96rem; font-weight:700;
                  margin-bottom:13px; }}
  .health-row {{ display:flex; justify-content:space-between; padding:9px 0;
                 border-bottom:1px solid rgba(31,56,82,.7); color:{CLR['subtext']};
                 font-size:.88rem; }}
  .health-row:last-child {{ border-bottom:0; }}
  .health-value {{ color:{CLR['text']}; font-weight:650; }}
  .equipment-panel {{ background:{CLR['surface']}; border:1px solid {CLR['border']};
                      border-radius:6px; padding:14px 16px 12px; min-height:380px;
                      box-shadow:0 2px 7px rgba(23,33,43,.09); }}
  .equipment-head {{ display:flex; justify-content:space-between; align-items:center;
                     color:{CLR['text']}; font-size:.96rem; font-weight:700; }}
  .equipment-state {{ color:{CLR['accent']}; font-size:.76rem; letter-spacing:.1em; }}
  .turbine-stage {{ position:relative; height:218px; display:flex; align-items:center;
                    justify-content:center; overflow:hidden; margin:2px 0 6px; }}
  .turbine-stage::after {{ content:""; position:absolute; left:4%; right:4%; bottom:13px;
                          height:1px; background:linear-gradient(90deg,transparent,{CLR['border']},transparent); }}
  .turbine-stage img {{ width:100%; max-height:210px; object-fit:contain;
                        filter:drop-shadow(0 12px 15px rgba(0,0,0,.55)); }}
  .sensor-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:7px; }}
  .sensor-tag {{ background:{CLR['surface2']}; border-left:2px solid {CLR['blue']};
                 padding:7px 8px; min-width:0; }}
  .sensor-tag.hot {{ border-left-color:{CLR['amber']}; }}
  .sensor-name {{ color:{CLR['subtext']}; font-size:.7rem; letter-spacing:.07em;
                  text-transform:uppercase; white-space:nowrap; }}
  .sensor-value {{ color:{CLR['text']}; font-size:.88rem; font-weight:650;
                   margin-top:2px; white-space:nowrap; }}
  .flow-strip {{ display:flex; align-items:center; justify-content:center; gap:7px;
                 color:{CLR['subtext']}; font-size:.72rem; letter-spacing:.05em;
                 margin-top:9px; }}
  .flow-arrow {{ color:{CLR['blue']}; }}
  .control-bar {{ background:{CLR['surface']}; border:1px solid {CLR['border']};
                  border-radius:6px; padding:9px 12px 5px; margin:12px 0 16px; }}
  div[data-testid="stSegmentedControl"] button {{ min-height:42px; padding:8px 20px;
                                                   border-radius:4px !important;
                                                   border:1px solid #b7c1cc !important;
                                                   background:linear-gradient(#ffffff,#e8edf2) !important;
                                                   color:{CLR['text']} !important;
                                                   box-shadow:0 2px 3px rgba(23,33,43,.13), inset 0 1px 0 #fff;
                                                   font-size:.88rem; font-weight:650; letter-spacing:.02em; }}
  div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
  div[data-testid="stSegmentedControl"] button[aria-checked="true"] {{
      background:linear-gradient(#2b7de9,#1f62c2) !important; color:white !important;
      border-color:#174e9c !important; box-shadow:inset 0 2px 4px rgba(0,0,0,.22) !important;
  }}
  div[data-testid="stPopover"] > button {{ min-height:42px; border:1px solid {CLR['border']};
                                            background:{CLR['surface']} !important;
                                            font-weight:650; }}
  div.stButton > button {{ min-height:42px; border-radius:4px; font-weight:700;
                           letter-spacing:.04em; border:1px solid #aeb9c4;
                           box-shadow:0 3px 5px rgba(23,33,43,.16), inset 0 1px 0 #fff; }}
  div.stButton > button:active {{ transform:translateY(2px);
                                  box-shadow:inset 0 2px 4px rgba(23,33,43,.2); }}
  .tour-kicker {{ color:{CLR['blue']}; font-size:.76rem; font-weight:750;
                  letter-spacing:.11em; text-transform:uppercase; margin-bottom:5px; }}
  .tour-title {{ color:{CLR['text']}; font-size:1.2rem; font-weight:700;
                 margin-bottom:8px; }}
  .tour-copy {{ color:{CLR['subtext']}; font-size:1rem; line-height:1.6;
                padding-bottom:10px; }}
  .tour-tip {{ background:#eef5ff; border-left:3px solid {CLR['blue']};
               color:{CLR['text']}; padding:10px 12px; border-radius:3px;
               font-size:.9rem; margin:8px 0 14px; }}
  .st-key-tour_spotlight {{ position:relative; z-index:999; background:{CLR['surface']};
                            border:2px solid {CLR['blue']}; border-radius:8px;
                            padding:12px; box-shadow:0 0 0 9999px rgba(17,27,38,.62),
                            0 0 0 5px rgba(31,111,235,.22), 0 12px 34px rgba(0,0,0,.28); }}
  .tour-card {{ background:#ffffff; border-left:4px solid {CLR['blue']};
                padding:12px 14px; margin:8px 0 12px; color:{CLR['text']};
                box-shadow:0 3px 10px rgba(23,33,43,.12); }}
  .tour-card-title {{ font-size:1.05rem; font-weight:750; margin:2px 0 5px; }}
  .tour-card-copy {{ color:{CLR['subtext']}; font-size:.92rem; line-height:1.5; }}
  [data-testid="stCaptionContainer"] p {{ font-size:.84rem !important; line-height:1.45; }}
  [data-testid="stWidgetLabel"] p {{ font-size:.88rem !important; }}
  h1, h2, h3 {{ color: {CLR['text']} !important; font-family: "Segoe UI", sans-serif; }}
  section[data-testid="stSidebar"] {{ background: {CLR['surface']} !important;
                                      border-right: 1px solid {CLR['border']}; }}
  section[data-testid="stSidebar"] .stMarkdown h2 {{ font-size:1.05rem; }}
  div[data-testid="stPlotlyChart"] {{ border:1px solid {CLR['border']}; border-radius:6px;
                                      overflow:hidden; box-shadow:0 2px 5px rgba(23,33,43,.07); }}
  hr {{ border-color:{CLR['border']} !important; opacity:.75; }}
  @media (min-width:1800px) {{
      .turbine-stage {{ height:250px; }}
      .turbine-stage img {{ max-height:240px; }}
      .equipment-panel {{ min-height:415px; }}
  }}
  @media (max-width:800px) {{
      .sensor-grid {{ grid-template-columns:repeat(2,1fr); }}
      .block-container {{ padding-left:.65rem !important; padding-right:.65rem !important; }}
  }}
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

    # Prediction rows follow the chronological 2014–2015 test block. Attach
    # operational tags so the equipment panel displays real turbine telemetry.
    processed = base.parent / "data" / "processed" / "syngas_features.csv"
    if datasets and processed.exists():
        sensor_cols = ["year", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP", "TEY"]
        test_sensors = pd.read_csv(processed, usecols=sensor_cols)
        test_sensors = test_sensors.loc[test_sensors["year"] >= 2014].reset_index(drop=True)
        for label, frame in datasets.items():
            if len(frame) == len(test_sensors):
                for column in sensor_cols:
                    if column not in frame:
                        frame[column] = test_sensors[column].to_numpy()

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
            "year":       np.full(n, 2015),
            "AH":         np.clip(65 + rng.normal(0, 8, n), 25, 100),
            "AFDP":       np.clip(4.2 + rng.normal(0, .3, n), 2, 8),
            "GTEP":       np.clip(28 + rng.normal(0, 2, n), 17, 41),
            "TIT":        np.clip(1080 + rng.normal(0, 8, n), 1000, 1101),
            "TAT":        np.clip(540 + rng.normal(0, 3, n), 511, 551),
            "CDP":        np.clip(12.5 + rng.normal(0, .5, n), 9.8, 15.2),
            "TEY":        np.clip(135 + rng.normal(0, 8, n), 100, 180),
        })
        datasets["XGBoost + MAPIE (simulated)"]   = fallback
        datasets["PINN + MC Dropout (simulated)"] = fallback.copy()

    return datasets


@st.cache_data
def load_turbine_asset() -> str:
    """Encode the generated transparent turbine render for inline HTML."""
    path = Path(__file__).parent / "assets" / "gas_turbine_control_room.png"
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


TOUR_STEPS = [
    {
        "title": "Overview",
        "copy": "The Overview is the operator's primary situational-awareness screen. "
                "It shows the latest historical operating point, energy yield, thermal "
                "state, emissions estimates and turbine condition without model diagnostics.",
        "tip": "Start here to answer: What is the turbine doing, and are emissions normal?",
    },
    {
        "title": "Inspect the operating overview",
        "copy": "The TEY dial shows measured turbine energy yield. CO and NOx dials show "
                "soft-sensor estimates at the same operating point. Green, amber and red "
                "bands correspond to configured operating limits.",
        "tip": "These are contemporaneous estimates—not future forecasts.",
    },
    {
        "title": "Open the Live Demo",
        "copy": "Select Live Demo in the top navigation. This is the portfolio workflow for "
                "showing how recorded turbine measurements advance through the interface.",
        "tip": "Action: select Live Demo before continuing.",
    },
    {
        "title": "Run the historical simulation",
        "copy": "This view replays the chronological test dataset as an operator demonstration. "
                "The dials, turbine state, sensor tags and emissions trend advance together "
                "using recorded values and model outputs.",
        "tip": "Action: press START, observe the instruments update, then press PAUSE.",
    },
    {
        "title": "Review emissions analysis",
        "copy": "This engineering-review view focuses on the selected emission channel, its "
                "calibrated prediction interval, warning headroom, recent events and measured "
                "coverage of the interval.",
        "tip": "Action: select Emissions Analysis to inspect model reliability.",
    },
    {
        "title": "You are ready",
        "copy": "You have completed the core workflow: inspect plant state, run the historical demo "
                "and review emissions reliability. Model & display settings contains the less-frequent "
                "engineering controls.",
        "tip": "Use HELP & TOUR at any time to restart this guide.",
    },
]


@st.dialog("Control room interface guide", width="large")
def render_tour_dialog():
    """Offer a first-launch introduction and a reusable interface guide."""
    if not st.session_state.get("tour_started", False):
        st.markdown(
            '<div class="tour-kicker">Welcome</div>'
            '<div class="tour-title">Take a guided interface tour?</div>'
            '<div class="tour-copy">This short guide explains the vital operating views, '
            'data-driven dials, turbine telemetry, emissions analysis and model settings.</div>',
            unsafe_allow_html=True,
        )
        take_col, skip_col = st.columns(2)
        if take_col.button("START TOUR", type="primary", width="stretch"):
            st.session_state.tour_active = True
            st.session_state.tour_step = 0
            st.session_state.tour_dialog_open = False
            st.rerun()
        if skip_col.button("NOT NOW", width="stretch"):
            st.session_state.tour_dialog_open = False
            st.rerun()
        return

def render_tour_card(required_view: str | None = None, require_demo: bool = False):
    """Render the focused explanation and navigation inside one spotlight target."""
    if not st.session_state.get("tour_active", False):
        return
    step_index = int(st.session_state.get("tour_step", 0))
    step = TOUR_STEPS[step_index]
    st.markdown(
        f'<div class="tour-card"><div class="tour-kicker">Guided task · '
        f'{step_index + 1} of {len(TOUR_STEPS)}</div>'
        f'<div class="tour-card-title">{step["title"]}</div>'
        f'<div class="tour-card-copy">{step["copy"]}</div>'
        f'<div class="tour-tip">{step["tip"]}</div></div>',
        unsafe_allow_html=True,
    )
    requirement_met = required_view is None or st.session_state.get("control_room_view") == required_view
    if require_demo:
        requirement_met = requirement_met and st.session_state.get("demo_ever_started", False)
    back_col, next_col, skip_col = st.columns(3)
    if back_col.button("← BACK", disabled=step_index == 0, key=f"tour_back_{step_index}", width="stretch"):
        st.session_state.tour_step = step_index - 1
        st.rerun()
    final_step = step_index == len(TOUR_STEPS) - 1
    next_label = "FINISH" if final_step else "NEXT →"
    if next_col.button(next_label, type="primary", disabled=not requirement_met,
                       key=f"tour_next_{step_index}", width="stretch"):
        if final_step:
            st.session_state.tour_active = False
            st.session_state.tour_completed = True
        else:
            st.session_state.tour_step = step_index + 1
        st.rerun()
    if skip_col.button("SKIP TOUR", key=f"tour_skip_{step_index}", width="stretch"):
        st.session_state.tour_active = False
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  TOP NAVIGATION AND ENGINEERING SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

def _render_controls_content(datasets: dict) -> dict:
    """Keep operator navigation visible and engineering controls out of the way."""
    nav_col, help_col, settings_col = st.columns([4.2, .75, 1.25], gap="small")
    with nav_col:
        view = st.segmented_control(
            "Control room view",
            ["Overview", "Live demo", "Emissions analysis"],
            default="Overview", label_visibility="collapsed", key="control_room_view",
        )

    with help_col:
        if st.button("HELP & TOUR", width="stretch"):
            st.session_state.tour_dialog_open = True
            st.session_state.tour_started = False
            st.session_state.tour_active = False

    with settings_col:
        with st.popover("Model & display settings", width="stretch"):
            st.caption("MODEL AND DISPLAY CONFIGURATION")
            model = st.selectbox("Inference model", list(datasets.keys()))
            target = st.segmented_control(
                "Emission channel", ["CO", "NOx"], default="CO"
            )

            defaults = {"CO": (5.0, 10.0), "NOx": (80.0, 100.0)}
            warn_default, alarm_default = defaults[target]
            st.markdown("**Operating limits**")
            warn_thresh = st.slider(
                "Warning threshold", 0.0, float(alarm_default),
                float(warn_default), step=0.1, key=f"warn_{target}",
            )
            alarm_thresh = st.slider(
                "Alarm threshold", float(warn_thresh), float(alarm_default * 2),
                max(float(alarm_default), float(warn_thresh)), step=0.1,
                key=f"alarm_{target}",
            )
            st.markdown("**Timeline and review**")
            window = st.slider("Trend window", 50, 500, 200, step=25)
            demo_interval = st.slider(
                "Demo frame interval (seconds)", 0.2, 2.0, 0.8, step=0.1,
                help="Playback speed for the historical-data demonstration",
            )
            show_reference = st.toggle(
                "Show measured reference", value=False,
                help="Engineering review only; hidden in the operator view",
            )
            if target == "NOx":
                st.markdown(
                    '<div class="drift-note"><strong>Model note</strong>'
                    'The 2014–2015 NOx holdout contains a distribution shift. '
                    'Use interval width as a reliability signal.</div>',
                    unsafe_allow_html=True,
                )

    return dict(view=view, model=model, target=target,
                warn_thresh=warn_thresh, alarm_thresh=alarm_thresh,
                window=window, demo_interval=demo_interval,
                show_reference=show_reference)


def render_controls(datasets: dict) -> dict:
    """Render controls and spotlight navigation during relevant guided steps."""
    step = int(st.session_state.get("tour_step", 0))
    spotlight = st.session_state.get("tour_active", False) and step in {0, 2, 4}
    with st.container(key="tour_spotlight" if spotlight else "top_control_area"):
        cfg = _render_controls_content(datasets)
        if spotlight:
            required = "Overview" if step == 0 else "Live demo" if step == 2 else "Emissions analysis"
            render_tour_card(required_view=required)
    return cfg


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
            font=dict(color=CLR["text"], size=17), x=0.025,
        ),
        yaxis_title=f"{target} [mg/m³]",
        xaxis_title="Sample index (test set)",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=CLR["text"])),
        height=380,
    )
    st.plotly_chart(fig, width="stretch")


def render_equipment_panel(df: pd.DataFrame, idx: int, target: str,
                           warn_thresh: float, alarm_thresh: float):
    """Render the turbine-generator asset with current operational sensor tags."""
    row = df.iloc[idx]
    pred = float(row[f"y_pred_{target}"])
    if pred >= alarm_thresh:
        state, state_color = "ALARM", CLR["red"]
    elif pred >= warn_thresh:
        state, state_color = "WARNING", CLR["amber"]
    else:
        state, state_color = "RUNNING · NORMAL", CLR["accent"]

    def value(name: str, unit: str, decimals: int = 1) -> str:
        raw = row.get(name, np.nan)
        return "—" if pd.isna(raw) else f"{raw:.{decimals}f} {unit}"

    image_b64 = load_turbine_asset()
    image_html = (
        f'<img src="data:image/png;base64,{image_b64}" alt="Industrial gas turbine generator">'
        if image_b64 else
        '<div style="color:#9fa7b3">Turbine asset unavailable</div>'
    )
    tags = [
        ("Compressor pressure", value("CDP", "mbar", 2), ""),
        ("Exhaust pressure", value("GTEP", "mbar", 2), ""),
        ("Turbine inlet", value("TIT", "°C", 1), "hot"),
        ("Turbine exhaust", value("TAT", "°C", 1), "hot"),
        ("Energy yield", value("TEY", "MWh", 1), ""),
        ("Ambient humidity", value("AH", "%", 1), ""),
    ]
    tag_html = "".join(
        f'<div class="sensor-tag {kind}"><div class="sensor-name">{name}</div>'
        f'<div class="sensor-value">{reading}</div></div>'
        for name, reading, kind in tags
    )
    st.markdown(
        '<div class="equipment-panel">'
        '<div class="equipment-head"><span>GT-01 · TURBINE GENERATOR</span>'
        f'<span class="equipment-state" style="color:{state_color}">● {state}</span></div>'
        f'<div class="turbine-stage">{image_html}</div>'
        f'<div class="sensor-grid">{tag_html}</div>'
        '<div class="flow-strip"><span>INLET AIR</span><span class="flow-arrow">→</span>'
        '<span>COMPRESSOR</span><span style="color:#f2cc0c">→</span>'
        '<span>COMBUSTION</span><span style="color:#f2495c">→</span>'
        '<span>EXHAUST</span></div></div>',
        unsafe_allow_html=True,
    )


def calculate_ramp_state(df: pd.DataFrame, idx: int) -> tuple[str, float]:
    """Classify TEY movement using a tolerance learned from historical changes."""
    changes = df["TEY"].diff().dropna()
    tolerance = float(changes.abs().median())
    start = max(1, idx - 9)
    local_change = float(df["TEY"].iloc[start:idx + 1].diff().median())
    if local_change > tolerance:
        return "RAMPING UP", local_change
    if local_change < -tolerance:
        return "RAMPING DOWN", local_change
    return "STEADY LOAD", local_change


def _gauge(title: str, value: float, minimum: float, maximum: float,
           unit: str, warn: float | None = None,
           alarm: float | None = None) -> go.Figure:
    """Build a compact control-room dial from observed or configured bounds."""
    steps = []
    if warn is not None and alarm is not None:
        steps = [
            {"range": [minimum, warn], "color": "rgba(115,191,105,.22)"},
            {"range": [warn, alarm], "color": "rgba(242,204,12,.25)"},
            {"range": [alarm, maximum], "color": "rgba(242,73,92,.25)"},
        ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": f" {unit}", "font": {"size": 22, "color": CLR["text"]}},
        title={"text": title, "font": {"size": 13, "color": CLR["subtext"]}},
        gauge={
            "axis": {"range": [minimum, maximum], "tickfont": {"size": 10, "color": CLR["subtext"]}},
            "bar": {"color": CLR["blue"], "thickness": .22},
            "bgcolor": CLR["surface2"], "borderwidth": 0,
            "steps": steps,
            "threshold": ({"line": {"color": CLR["red"], "width": 2},
                           "thickness": .75, "value": alarm} if alarm is not None else None),
        },
    ))
    fig.update_layout(
        paper_bgcolor=CLR["surface"], margin=dict(l=18, r=18, t=38, b=8),
        height=205, font=dict(family="Inter, Segoe UI, sans-serif"),
    )
    return fig


def render_operating_overview(df: pd.DataFrame, idx: int, target: str,
                              warn_thresh: float, alarm_thresh: float):
    """Show only measured operating variables and model emission estimates."""
    row = df.iloc[idx]
    ramp_state, tey_change = calculate_ramp_state(df, idx)
    year = int(row.get("year", 0))

    cols = st.columns(5)
    cols[0].metric("Operating state", ramp_state, f"{tey_change:+.2f} MWh/sample")
    cols[1].metric("Energy yield · TEY", f"{row['TEY']:.1f} MWh")
    cols[2].metric("Turbine inlet · TIT", f"{row['TIT']:.1f} °C")
    cols[3].metric("Exhaust temperature · TAT", f"{row['TAT']:.1f} °C")
    cols[4].metric("Historical year", str(year) if year else "Dataset")

    co_warn, co_alarm = (warn_thresh, alarm_thresh) if target == "CO" else (5.0, 10.0)
    nox_warn, nox_alarm = (warn_thresh, alarm_thresh) if target == "NOx" else (80.0, 100.0)
    tey_min, tey_max = float(df["TEY"].min()), float(df["TEY"].max())
    co_max = max(co_alarm * 1.2, float(df["y_pred_CO"].quantile(.99)))
    nox_max = max(nox_alarm * 1.2, float(df["y_pred_NOx"].quantile(.99)))

    gauge_cols = st.columns(3, gap="medium")
    with gauge_cols[0]:
        st.plotly_chart(_gauge("ENERGY YIELD · TEY", float(row["TEY"]),
                               tey_min, tey_max, "MWh"), width="stretch")
    with gauge_cols[1]:
        st.plotly_chart(_gauge("SOFT-SENSOR · CO", float(row["y_pred_CO"]),
                               0, co_max, "mg/m³", co_warn, co_alarm), width="stretch")
    with gauge_cols[2]:
        st.plotly_chart(_gauge("SOFT-SENSOR · NOx", float(row["y_pred_NOx"]),
                               0, nox_max, "mg/m³", nox_warn, nox_alarm), width="stretch")

    render_equipment_panel(df, idx, target, warn_thresh, alarm_thresh)


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
                       font=dict(color=CLR["text"], size=16), x=0.04),
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
                font=dict(color=CLR["text"], size=16), x=0.04,
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
    cfg      = render_controls(datasets)
    df       = datasets[cfg["model"]]
    n        = len(df)

    if "tour_prompted" not in st.session_state:
        st.session_state.tour_prompted = True
        st.session_state.tour_dialog_open = True
        st.session_state.tour_started = False
        st.session_state.tour_active = False
    if st.session_state.get("tour_dialog_open", False):
        render_tour_dialog()

    st.caption(
        f"ACTIVE MODEL  ·  {cfg['model'].upper()}     |     "
        f"CHANNEL  ·  {cfg['target']}     |     RECORDS  ·  {n:,}"
    )

    if cfg["view"] == "Overview":
        idx = st.slider("Historical position", cfg["window"], n, n,
                        help="Drag to inspect any point in the test set")
        overview_spotlight = st.session_state.get("tour_active", False) and st.session_state.get("tour_step") == 1
        with st.container(key="tour_spotlight" if overview_spotlight else "overview_console"):
            if overview_spotlight:
                render_tour_card(required_view="Overview")
            render_operating_overview(df, idx - 1, cfg["target"],
                                      cfg["warn_thresh"], cfg["alarm_thresh"])

    elif cfg["view"] == "Live demo":
        if "demo_idx" not in st.session_state:
            st.session_state.demo_idx = cfg["window"]
        if "demo_running" not in st.session_state:
            st.session_state.demo_running = False

        idx = min(max(int(st.session_state.demo_idx), cfg["window"]), n)
        demo_spotlight = st.session_state.get("tour_active", False) and st.session_state.get("tour_step") == 3
        with st.container(key="tour_spotlight" if demo_spotlight else "demo_controls"):
            st.info(
                "DEMONSTRATION MODE · Replaying chronological historical test data. "
                "This is not a live plant connection.",
                icon="ℹ️",
            )
            start_col, pause_col, reset_col, progress_col = st.columns([1, 1, 1, 4])
            if start_col.button("START", type="primary", width="stretch",
                                help="Begin playback of chronological recorded test samples"):
                st.session_state.demo_running = True
                st.session_state.demo_ever_started = True
                st.rerun()
            if pause_col.button("PAUSE", width="stretch",
                                help="Pause playback at the current historical sample"):
                st.session_state.demo_running = False
                st.rerun()
            if reset_col.button("RESET", width="stretch",
                                help="Return the demonstration to its first display window"):
                st.session_state.demo_running = False
                st.session_state.demo_idx = cfg["window"]
                st.rerun()
            progress_col.progress(idx / n, text=f"Historical sample {idx:,} of {n:,}")
            if demo_spotlight:
                render_tour_card(required_view="Live demo", require_demo=True)
        render_operating_overview(df, idx - 1, cfg["target"],
                                  cfg["warn_thresh"], cfg["alarm_thresh"])
        render_timeseries(df, cfg["target"], cfg["window"],
                          cfg["warn_thresh"], cfg["alarm_thresh"], idx,
                          cfg["show_reference"])

        if st.session_state.demo_running:
            st.session_state.demo_idx = idx + 1 if idx < n else cfg["window"]
            time.sleep(cfg["demo_interval"])
            st.rerun()

    else:  # Emissions analysis
        idx = st.slider("Historical position", cfg["window"], n, n,
                        help="Drag to inspect any point in the test set")
        analysis_spotlight = st.session_state.get("tour_active", False) and st.session_state.get("tour_step") == 5
        with st.container(key="tour_spotlight" if analysis_spotlight else "analysis_console"):
            if analysis_spotlight:
                render_tour_card(required_view="Emissions analysis")
            render_kpis(df, cfg["target"], cfg["warn_thresh"],
                        cfg["alarm_thresh"], idx - 1)
            trend_col, equipment_col = st.columns([1.35, 1], gap="medium")
            with trend_col:
                render_timeseries(df, cfg["target"], cfg["window"],
                                  cfg["warn_thresh"], cfg["alarm_thresh"], idx,
                                  cfg["show_reference"])
            with equipment_col:
                render_equipment_panel(df, idx - 1, cfg["target"],
                                       cfg["warn_thresh"], cfg["alarm_thresh"])
            render_control_summary(df, cfg["target"], idx,
                                   cfg["warn_thresh"], cfg["alarm_thresh"])


if __name__ == "__main__":
    main()
