# 🌍 Digital Twin Soft-Sensor for Power Plant Emission Monitoring

A physics-constrained AI system for real-time prediction of gas turbine emissions using Digital Twin technology, uncertainty-aware machine learning, and thermodynamic feature engineering.

---

## 📌 Project Overview

Modern gas turbine power plants generate electricity efficiently, but they also emit harmful pollutants such as **Carbon Monoxide (CO)** and **Nitrogen Oxides (NOx)**.

Monitoring these emissions in real time is challenging because traditional high-temperature emission sensors are expensive, slow to respond, difficult to maintain, and prone to calibration drift.

This project introduces a **Digital Twin Soft-Sensor** capable of predicting emissions instantly using operational sensor data and physics-constrained machine learning.

---

## 🎯 Objective

Build a **real-time virtual emissions sensor** that:

✅ Predicts turbine emissions from operational parameters  
✅ Detects when combustion becomes inefficient ("running dirty")  
✅ Provides uncertainty-aware predictions for safer deployment  
✅ Helps operators reduce emissions before environmental violations occur  

---

## 📐 Project Scope

The project covers:

- Estimation of CO and NOx from the nine operational variables in the UCI gas-turbine dataset
- Physics-derived feature engineering and physics-constrained neural-network training
- Comparison of XGBoost and neural soft sensors using a chronological train/test split
- Prediction intervals, held-out performance checks, and detection of potential concept drift
- Historical-data playback and operator-oriented model monitoring in the Streamlit dashboard

The current scope does not cover:

- A live connection to plant sensors, SCADA, or control systems
- Automated turbine control or operating-point optimisation
- Pollutants or plant configurations outside the supplied dataset
- Root-cause diagnosis of combustion or equipment faults
- Regulatory certification or replacement of required emissions-monitoring equipment

---

## 🧠 Core Idea: Physics + AI

Most machine learning models only learn patterns from raw data. This project also incorporates **thermodynamic knowledge** through engineered features and a differentiable gradient regularizer.

The neural soft sensor minimises prediction error while respecting physically motivated directional priors for its predicted CO response. It uses a physics-constrained multi-task neural-network formulation based on gradient regularisation.

---

## ⚙️ Technical Architecture

### Layer 1 — Data Ingestion & Physics Feature Engineering

Raw operational turbine data is validated and transformed into physically meaningful features.

**Raw sensor inputs (9):** AT, AP, AH, AFDP, GTEP, TIT, TAT, CDP, TEY

**Physics-derived features (7):**

| Feature | Formula | Physical meaning |
|---|---|---|
| `T_ratio` | TIT / TAT | Isentropic expansion efficiency proxy |
| `compression` | CDP / AP × 1000 | Scaled compressor-pressure proxy |
| `T_drop` | TIT − TAT | Temperature drop across turbine |
| `humidity_abs` | f(AH, AT) | Absolute-humidity proxy via Magnus approximation |
| `fouling_idx` | AFDP / AP × 1000 | Scaled filter-pressure-drop proxy |
| `specific_work` | TEY / CDP | Energy yield per unit compression |
| `dT_AT_TIT` | TIT − AT | Total temperature rise from ambient |

---

### Layer 2 — Hybrid AI Modelling

Two approaches with uncertainty-aware outputs:

#### 🌲 XGBoost + MAPIE
- Hyperparameter search: `RandomizedSearchCV` with `TimeSeriesSplit` (5 folds), with preprocessing fitted independently inside each CV fold
- Final 20% of the 2011–2013 block is reserved chronologically for conformalization before preprocessing/tuning
- Uncertainty: split-conformal prediction via MAPIE `SplitConformalRegressor`

#### 🧠 Physics-Constrained Neural Soft Sensor

Architecture: `BatchNorm → Linear(128, SiLU) → Linear(64, SiLU) → Linear(32, SiLU) → CO head / NOx head`

Hybrid loss function:

```
L_total = L_data + λ · L_physics
```

The data loss covers both CO and NOx. The physics loss directly constrains only CO: it penalises local gradients that violate the directional prior that CO should decrease with TIT/compression and increase with humidity. NOx is predicted jointly but currently has no direct physics constraint. Set `λ = 0` to recover a plain multi-output MLP baseline.

This formulation applies gradient-based physical regularisation. The internal `pinn_*` function and artefact names are retained for compatibility.

Training: AdamW + CosineAnnealingLR, gradient clipping, early stopping (patience = 20).

---

## 📉 Uncertainty Quantification — MC Dropout

At inference the network stays in evaluation mode so BatchNorm statistics remain fixed, while only Dropout layers are re-enabled. 100 stochastic forward passes are averaged to produce a predictive mean and an epistemic uncertainty estimate. A separate chronological calibration block converts the varying MC standard deviations into normalized split-conformal intervals.

```
Calibrated interval = mean ± conformal_scale × MC_std
```

The split-conformal guarantee depends on calibration and deployment observations being exchangeable. Temporal concept drift can invalidate that assumption, so coverage is also reported on held-out data.

---

## 📊 Results

### 1️⃣ Prediction Fidelity

Run `compare_models.py` after training to generate the reproducible RMSE, MAE, R², empirical coverage, and interval-width table. Generate figures separately with `results/scientific_plots.py`.

---

### 2️⃣ Uncertainty Reliability

Use the generated comparison metrics to check empirical coverage and interval width for the current run. MAPIE intervals are conformalized on a held-out chronological block; MC-Dropout intervals are approximate and should be judged by their measured held-out coverage before any deployment claim is made.

---

## 🖥️ Interactive Monitoring Dashboard

An operator-style monitoring and historical demonstration dashboard built with **Streamlit + Plotly**:

- Emission estimates with reported uncertainty bands from saved model outputs
- Configurable alarm and warning thresholds
- Early approaching-limit assessment based on recent estimate direction, threshold headroom, and the upper prediction bound
- Operator-focused trend, limit-event queue, and model-health summary
- Visible measured-reference overlay in Emissions Analysis for held-out estimate validation
- Separate operator Overview, historical Live Demo, and engineering Emissions Analysis views
- Start, pause, and reset controls for portfolio demonstration playback
- First-launch spotlight tour with action-based steps, contextual tooltips, and a reusable restart control

---

## 🗂️ Project Structure

```
project/
├── data/
│   ├── raw/                    # Unmodified UCI download
│   └── processed/              # Validated data + physics features
├── models/                     # Saved model artefacts (.joblib, .pt)
├── src/
│   ├── preprocessing.py        # Data fetch, validation, train/test split
│   ├── feature_engineering.py  # Physics-derived feature construction
│   ├── pinn_model.py           # Physics-constrained NN and gradient regularizer
│   ├── uncertainty.py          # MAPIE wrapper and evaluation utilities
│   └── training.py             # Full training pipelines for both models
├── dashboard/
│   ├── assets/                 # Dashboard turbine artwork
│   └── app.py                  # Streamlit monitoring dashboard
├── results/
│   └── scientific_plots.py     # Explicit scientific figure generation
├── build_dataset.py            # Data pipeline entry point
├── compare_models.py           # Side-by-side model comparison
└── requirements.txt
```

---

## 🚀 Quickstart

```bash
# Python 3.10+
pip install -r requirements.txt

# Create output directories (first time only)
mkdir models results data/raw data/processed

# 1. Fetch and process data
python build_dataset.py

# 2. Train models (can be run independently)
python -c "from src.training import run_xgboost; run_xgboost()"
python -c "from src.training import run_pinn; run_pinn()"

# 3. Generate the numerical comparison only
python compare_models.py

# 4. Generate publication-style figures when needed
python results/scientific_plots.py

# 5. Launch dashboard
streamlit run dashboard/app.py
```

---

## 📚 Data Source

**Gas Turbine CO and NOx Emission Data Set (2019)**  
UCI Machine Learning Repository · DOI: `10.24432/C5WC95`  
36,733 hourly instances from a working gas turbine in Turkey (2011–2015).

**Split:** Train = 2011–2013 | Test = 2014–2015 (temporal, no leakage)

**Research reference:**  
Heysem Kaya, Pinar Tüfekci, and Erdinç Uzun (2019). *Predicting CO and NOx emissions from gas turbines: multi-layer perceptron and ensemble learning approaches and a new large-scale dataset.*

---

## 🧰 Tech Stack

| Category | Libraries |
|---|---|
| Machine Learning | PyTorch, XGBoost, Scikit-learn, MAPIE |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Deployment | Streamlit |

## ⚠️ Deployment Caveat

The 2014-2015 holdout contains a material NOx concept shift: its annual mean is
about 60 mg/m3 versus 68-70 mg/m3 in 2011-2013. The available sensor features do
not fully explain this change. Poor held-out NOx R2 or coverage should trigger a
drift review while the test labels remain reserved for evaluation. Production use
requires
periodic labeled recalibration or an additional operating-regime/configuration
signal.

Physics-constrained NN MC-Dropout widths are normalized split-conformal intervals calibrated on a
dedicated chronological block that is excluded from fitting and early stopping.

---

## 🙏 Acknowledgements and Attributions

### Data and research

- The project uses the **UCI Gas Turbine CO and NOx Emission Data Set** contributed by Heysem Kaya, Pinar Tüfekci, and Erdinç Uzun. The dataset DOI and associated research paper are cited in the [Data Source](#-data-source) section.
- The implementation builds on the open-source Python ecosystem, particularly PyTorch, XGBoost, Scikit-learn, MAPIE, Pandas, NumPy, Streamlit, Plotly, Matplotlib, and Seaborn. Their respective licences and documentation govern those dependencies.

### Dashboard artwork and visual references

- `dashboard/assets/gas_turbine_control_room.png`, the turbine artwork displayed in the dashboard, was generated with **OpenAI image-generation tooling** and selected, reviewed, and integrated by the project author.
- `dashboard/assets/gas_turbine_nasa.svg` is a bundled NASA educational illustration retained as an unused reference asset. NASA should be credited when the asset is reused, and users should check the applicable [NASA media usage guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/).
- Industrial monitoring screenshots and the linked Free3D turbine preview supplied during design informed the control-room layout, hierarchy, and colour ideas.

### AI-assisted development disclosure

The dashboard redesign and parts of the supporting documentation were developed with assistance from **OpenAI ChatGPT and Codex**. AI assistance included interface ideation, Streamlit and CSS implementation, debugging, test support, documentation editing, and generation of the turbine artwork. The project author directed the work, selected the final design, reviewed the generated changes, and remains responsible for the code, scientific interpretation, validation, and published results.

---

## 📝 Changelog

### 2026-08-21

- Rebuilt the Streamlit dashboard in a professional light control-room style with robust top-level navigation and model settings separated from the operator view.
- Added Overview, historical Live Demo, and Emissions Analysis views driven by available turbine measurements and model outputs.
- Added the turbine equipment panel, operational dials, sensor tags, alert states, and clearly labelled historical-data playback controls.
- Simplified playback to one stateful Start/Pause control and reduced non-essential status colour using high-performance HMI principles.
- Replaced oversized semicircular gauges with compact range indicators and clarified measured versus model-estimated values.
- Renamed the heuristic ramp label to Recent TEY Trend and added a data-backed soft-sensor interval summary.
- Added an early approaching-limit assessment and documented its uncertainty/headroom/trend logic throughout the dashboard tour.
- Moved the measured-reference overlay from hidden settings into the Emissions Analysis validation toolbar.
- Added a first-launch spotlight tour with action-based steps, progress, Back/Next/Skip controls, contextual tooltips, completion guidance, and a restart option.
- Separated scientific figure generation from training and comparison so figures are produced only when the dedicated plotting script is run.
- Updated the README with dashboard guidance, changelog entries, third-party attribution, and an AI-assisted development disclosure.

### 2026-08-20

- Added leakage-safe normalized conformal calibration for neural-network MC-Dropout intervals.
- Documented the 2014–2015 NOx concept shift and deployment limitation.
- Redesigned the Streamlit interface as an operator-focused control-room dashboard.
- Added focused uncertainty-calibration tests and Windows/headless execution fixes.
