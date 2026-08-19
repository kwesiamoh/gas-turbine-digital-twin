# 🌍 Digital Twin Soft-Sensor for Power Plant Emission Monitoring

A **physics-constrained, uncertainty-aware AI soft sensor** for near-real-time prediction of gas-turbine CO and NOx emissions using operational data, thermodynamically motivated feature engineering, and calibrated machine-learning uncertainty estimates.

---

## 📌 Project Overview

Modern gas-turbine power plants generate electricity efficiently, but they also emit pollutants such as **Carbon Monoxide (CO)** and **Nitrogen Oxides (NOx)**.

Continuous emissions monitoring requires dedicated instrumentation, calibration, maintenance, and reliable measurement infrastructure. A virtual soft sensor can complement these systems by estimating emissions directly from routinely measured turbine operating conditions.

This project develops a **Digital Twin Soft-Sensor** that estimates CO and NOx from operational sensor data using physics-constrained machine learning, thermodynamically motivated feature engineering, and uncertainty quantification.

---

## 🎯 Objective

Build a **near-real-time virtual emissions sensor** that:

✅ Predicts CO and NOx emissions from turbine operating parameters  
✅ Identifies operating conditions associated with inefficient or high-emission combustion  
✅ Provides uncertainty-aware predictions alongside point estimates  
✅ Supports early identification of potentially high-emission operating conditions  

---

## 🧠 Core Idea: Physics + AI

Purely data-driven machine-learning models learn statistical relationships from observed operating data. This project augments that approach with **thermodynamically motivated features and physically informed directional constraints**.

The neural soft sensor jointly predicts CO and NOx while incorporating prior engineering knowledge into the CO learning objective. In particular, gradient-based regularization encourages the predicted CO response to follow selected physically motivated directional relationships with turbine inlet temperature, compression, and humidity.

The result is a **physics-constrained multi-task neural network**: the model introduces physical structure into learning without requiring a complete first-principles combustion model or governing-equation residuals.

---

## ⚙️ Technical Architecture

### Layer 1 — Data Ingestion & Physics-Motivated Feature Engineering

Raw turbine operating data is validated and transformed into additional features designed to capture useful thermodynamic and operating relationships.

**Raw sensor inputs (9):** AT, AP, AH, AFDP, GTEP, TIT, TAT, CDP, TEY

**Physics-motivated derived features (7):**

| Feature | Formula | Physical interpretation |
|---|---|---|
| `T_ratio` | TIT / TAT | Turbine inlet-to-outlet temperature-ratio proxy |
| `compression` | CDP / AP × 1000 | Scaled compressor discharge-to-ambient pressure proxy |
| `T_drop` | TIT − TAT | Temperature drop across the turbine |
| `humidity_abs` | f(AH, AT) | Absolute-humidity proxy via Magnus approximation |
| `fouling_idx` | AFDP / AP × 1000 | Scaled filter-pressure-drop proxy |
| `specific_work` | TEY / CDP | Power-to-compressor-discharge-pressure ratio |
| `dT_AT_TIT` | TIT − AT | Temperature rise from ambient conditions to turbine inlet |

These engineered variables are used as physically motivated proxies rather than exact thermodynamic state quantities.

---

### Layer 2 — Hybrid AI Modelling

Two approaches with uncertainty-aware outputs:

#### 🌲 XGBoost + MAPIE

- Hyperparameter search: `RandomizedSearchCV` with `TimeSeriesSplit` (5 folds), with preprocessing fitted independently inside each CV fold
- Final 20% of the 2011–2013 block is reserved chronologically for conformalization before preprocessing/tuning
- Uncertainty: split-conformal prediction via MAPIE `SplitConformalRegressor`

#### 🧠 Physics-Constrained Neural Soft Sensor

Architecture:

`BatchNorm → Linear(128, SiLU) → Linear(64, SiLU) → Linear(32, SiLU) → CO head / NOx head`

Hybrid loss function:

```text
L_total = L_data + λ · L_physics
```

The data loss trains both CO and NOx outputs jointly.

The physics loss acts directly on the CO prediction by penalising local gradients that violate selected directional priors:

- predicted CO should decrease as TIT increases
- predicted CO should decrease as the compression proxy increases
- predicted CO should increase with humidity

NOx is learned jointly through the shared representation but currently has no direct physics regularization term.

Setting `λ = 0` recovers a standard multi-output MLP baseline, allowing the contribution of the physics constraint to be tested directly.

This is a **gradient-regularized physics-constrained neural network** rather than an equation-residual PINN. The existing `pinn_*` function and artefact names are retained for compatibility with the project codebase.

Training: AdamW + CosineAnnealingLR, gradient clipping, early stopping (patience = 20).

---

## 📉 Uncertainty Quantification — MC Dropout

At inference time, the network remains in evaluation mode so BatchNorm statistics stay fixed while Dropout layers are selectively re-enabled.

For each input, **100 stochastic forward passes** are performed. Their mean provides the final prediction, while their spread provides a model-dependent measure of epistemic uncertainty.

A separate chronological calibration block is then used to convert the MC-Dropout standard deviations into normalized split-conformal prediction intervals:

```text
Calibrated interval = mean ± conformal_scale × MC_std
```

This combines input-dependent MC-Dropout uncertainty with conformal calibration.

The validity of split-conformal coverage depends on calibration and future observations being sufficiently exchangeable. Because temporal concept drift can violate this assumption, empirical coverage is also evaluated explicitly on the held-out 2014–2015 period.

---

## 📊 Results

### 1️⃣ Prediction Fidelity

Run `compare_models.py` after training to generate the reproducible RMSE, MAE, R², empirical coverage, and interval-width table. This command writes numerical results only; it does not generate figures.

---

### 2️⃣ Uncertainty Reliability

Uncertainty quality is assessed using both **empirical coverage** and **normalized interval width**.

A useful prediction interval should be well calibrated while remaining as narrow as possible:

- **Coverage** measures how often the true observation falls inside the predicted interval
- **Interval width** measures how sharp or informative those intervals are

The XGBoost model uses MAPIE split-conformal intervals, while the neural model combines MC Dropout with chronological conformal calibration.

Held-out performance therefore reflects both predictive accuracy and the trade-off between uncertainty calibration and interval sharpness.

<img width="2337" height="1790" alt="project-1-02" src="https://github.com/user-attachments/assets/951f8260-c8b4-4baa-a1b0-6714c82208cc" />

### Key Result

On the 2014–2015 holdout set, the physics-constrained neural model achieves lower normalized prediction error than XGBoost for both CO and NOx.

For CO, the neural model produces substantially sharper intervals, although with lower-than-nominal coverage. For NOx, it achieves much stronger interval coverage than XGBoost, but requires wider uncertainty bounds.

The results highlight a clear **calibration–sharpness trade-off** and show that the temporal NOx shift remains the more challenging deployment case.

---

## 🖥️ Real-Time Dashboard

A monitoring dashboard built with **Streamlit + Plotly** for inspecting model behaviour under turbine operating conditions:

- Near-real-time CO and NOx predictions with uncertainty bands
- Configurable warning and alarm thresholds
- Operator-focused trends and limit-event tracking
- Model-health and uncertainty summaries
- Optional reference-data overlay for engineering review
- Live simulation mode that steps through the held-out test period

<img width="2535" height="1313" alt="project-1-01" src="https://github.com/user-attachments/assets/39349c0e-5404-40f6-bc16-f1480d0b5433" />

---

## 🗂️ Project Structure

```text
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
│   └── app.py                  # Streamlit monitoring dashboard
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

# 4. Launch dashboard
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

---

## ⚠️ Deployment Caveat

The 2014–2015 holdout period exhibits a substantial shift in the NOx distribution: its annual mean is approximately 60 mg/m³ compared with roughly 68–70 mg/m³ during 2011–2013.

The available sensor variables do not fully explain this shift, making NOx prediction significantly more difficult under temporal distribution change. Poor held-out NOx R² or interval coverage should therefore be interpreted as evidence of **concept drift**, rather than corrected by tuning against test labels.

For production deployment, the model would require periodic labelled recalibration, drift monitoring, or additional operating-regime and turbine-configuration signals.

The neural-network uncertainty intervals are calibrated on a dedicated chronological calibration block excluded from model fitting and early stopping.

---

## 📝 Changelog

### 2026-08-19

- Added leakage-safe normalized conformal calibration for neural-network MC-Dropout intervals.
- Documented the 2014–2015 NOx concept shift and deployment limitation.
- Redesigned the Streamlit interface as an operator-focused control-room dashboard.
- Added focused uncertainty-calibration tests and Windows/headless execution fixes.
