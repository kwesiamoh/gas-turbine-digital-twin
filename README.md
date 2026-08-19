# 🌍 Digital Twin Soft-Sensor for Power Plant Emission Monitoring

A Physics-Informed AI system for real-time prediction of gas turbine emissions using Digital Twin technology, uncertainty-aware machine learning, and thermodynamic feature engineering.

---

## 📌 Project Overview

Modern gas turbine power plants generate electricity efficiently, but they also emit harmful pollutants such as **Carbon Monoxide (CO)** and **Nitrogen Oxides (NOx)**.

Monitoring these emissions in real time is challenging because traditional high-temperature emission sensors are expensive, slow to respond, difficult to maintain, and prone to calibration drift.

This project introduces a **Digital Twin Soft-Sensor** capable of predicting emissions instantly using operational sensor data and physics-informed machine learning.

---

## 🎯 Objective

Build a **real-time virtual emissions sensor** that:

✅ Predicts turbine emissions from operational parameters  
✅ Detects when combustion becomes inefficient ("running dirty")  
✅ Provides uncertainty-aware predictions for safer deployment  
✅ Helps operators reduce emissions before environmental violations occur  

---

## 🧠 Core Idea: Physics + AI

Most machine learning models only learn patterns from raw data. This project goes further by integrating **thermodynamic principles** directly into the learning process through a Physics-Informed Neural Network (PINN).

The model is trained not only to minimise prediction error, but also to respect physically motivated directional priors governing gas turbine combustion.

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

#### 🧠 Physics-Informed Neural Network (PINN)

Architecture: `BatchNorm → Linear(128, SiLU) → Linear(64, SiLU) → Linear(32, SiLU) → CO head / NOx head`

Hybrid loss function:

```
L_total = L_data + λ · L_physics
```

The model is penalised when predictions differ from measured emissions **and** when local CO gradients violate the directional prior: CO should decrease with TIT/compression and increase with humidity. Set `λ = 0` to recover a plain MLP baseline.

Training: AdamW + CosineAnnealingLR, gradient clipping, early stopping (patience = 20).

---

## 📉 Uncertainty Quantification — MC Dropout

At inference the network stays in evaluation mode so BatchNorm statistics remain fixed, while only Dropout layers are re-enabled. 100 stochastic forward passes are averaged to produce a predictive mean and an epistemic uncertainty estimate. A separate chronological calibration block converts the varying MC standard deviations into normalized split-conformal intervals.

```
Calibrated interval = mean ± conformal_scale × MC_std
```

The split-conformal guarantee depends on calibration and deployment observations being exchangeable. Temporal concept drift can invalidate that assumption, so coverage is also reported on held-out data.

---

## 📊 Scientific Results

### 1️⃣ Prediction Fidelity

Run `compare_models.py` after training to generate the reproducible RMSE, MAE, R², empirical coverage, and interval-width table. This command writes numerical results only; it does not generate figures.

---

### 2️⃣ Uncertainty Reliability

Use the generated comparison metrics to check empirical coverage and interval width for the current run. MAPIE intervals are conformalized on a held-out chronological block; MC-Dropout intervals are approximate and should be judged by their measured held-out coverage before any deployment claim is made.

---

### 3️⃣ Publication Figures

Run `python results/scientific_plots.py` explicitly when figures are required. It produces a concise four-panel comparison of normalized prediction error, explained variance, interval calibration, and interval sharpness. Outputs are saved as a 300-DPI PNG and a vector PDF using a colorblind-safe, grayscale-compatible style.

---

## 🖥️ Real-Time Dashboard

A live monitoring dashboard built with **Streamlit + Plotly**:

- Real-time emission predictions with reported uncertainty bands
- Configurable alarm and warning thresholds
- Operator-focused trend, limit-event queue, and model-health summary
- Optional reference overlay for engineering review
- Live simulation mode stepping through the test set

---

## 🗂️ Project Structure

```
project/
├── data/
│   ├── raw/                    # Unmodified UCI download
│   └── processed/              # Validated data + physics features
├── notebooks/                  # EDA and experiments
├── models/                     # Saved model artefacts (.joblib, .pt)
├── src/
│   ├── preprocessing.py        # Data fetch, validation, train/test split
│   ├── feature_engineering.py  # Physics-derived feature construction
│   ├── pinn_model.py           # PINN architecture and physics residual
│   ├── uncertainty.py          # MAPIE wrapper and evaluation utilities
│   └── training.py             # Full training pipelines for both models
├── dashboard/
│   └── app.py                  # Streamlit monitoring dashboard
├── results/
│   ├── scientific_plots.py     # Explicit publication-figure entry point
│   └── *.csv / *.png / *.pdf   # Generated metrics and figure outputs
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

# 4. Generate publication figures explicitly
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
not fully explain this change. Treat poor held-out NOx R2 or coverage as a drift
alarm, not as a reason to tune against the test labels. Production use requires
periodic labeled recalibration or an additional operating-regime/configuration
signal.

PINN MC-Dropout widths are normalized split-conformal intervals calibrated on a
dedicated chronological block that is excluded from fitting and early stopping.

---

## 📝 Changelog

### 2026-08-19

- Added leakage-safe normalized conformal calibration for PINN MC-Dropout intervals.
- Documented the 2014–2015 NOx concept shift and deployment limitation.
- Redesigned the Streamlit interface as an operator-focused control-room dashboard.
- Separated numerical comparison from scientific figure generation.
- Added publication-ready PNG and vector PDF outputs with normalized metrics and a colorblind-safe style.
- Added focused uncertainty-calibration tests and Windows/headless execution fixes.
