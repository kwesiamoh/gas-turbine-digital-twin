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

The model is trained not only to minimise prediction error, but also to respect known physical laws governing gas turbine combustion.

---

## ⚙️ Technical Architecture

### Layer 1 — Data Ingestion & Physics Feature Engineering

Raw operational turbine data is validated and transformed into physically meaningful features.

**Raw sensor inputs (9):** AT, AP, AH, AFDP, GTEP, TIT, TAT, CDP, TEY

**Physics-derived features (7):**

| Feature | Formula | Physical meaning |
|---|---|---|
| `T_ratio` | TIT / TAT | Isentropic expansion efficiency proxy |
| `compression` | CDP / AP | Compressor pressure ratio |
| `T_drop` | TIT − TAT | Temperature drop across turbine |
| `humidity_abs` | f(AH, AP, AT) | Absolute humidity via Magnus approximation |
| `fouling_idx` | AFDP / AP | Normalised filter fouling |
| `specific_work` | TEY / CDP | Energy yield per unit compression |
| `dT_AT_TIT` | TIT − AT | Total temperature rise from ambient |

---

### Layer 2 — Hybrid AI Modelling

Two approaches with calibrated uncertainty intervals:

#### 🌲 XGBoost + MAPIE
- Hyperparameter search: `RandomizedSearchCV` with `TimeSeriesSplit` (5 folds, preserves temporal ordering)
- Uncertainty: split-conformal prediction via MAPIE `SplitConformalRegressor`
- Coverage guarantee is distribution-free — no Gaussian assumption on residuals

#### 🧠 Physics-Informed Neural Network (PINN)

Architecture: `BatchNorm → Linear(128, SiLU) → Linear(64, SiLU) → Linear(32, SiLU) → CO head / NOx head`

Hybrid loss function:

```
L_total = L_data + λ · L_physics
```

The model is penalised when predictions differ from measured emissions **and** when they violate the thermodynamic directional constraint on CO w.r.t. TIT, compression, and humidity. Set `λ = 0` to recover a plain MLP baseline.

Training: AdamW + CosineAnnealingLR, gradient clipping, early stopping (patience = 20).

---

## 📉 Uncertainty Quantification — MC Dropout

Dropout layers remain active at inference. 100 stochastic forward passes are averaged to produce a predictive mean and standard deviation.

```
95% Prediction Interval = mean ± 1.96 × std
```

✅ Detects unreliable predictions  
✅ Flags high-risk operating conditions  
✅ Provides confidence bands for operators  

---

## 📊 Scientific Results

### 1️⃣ Prediction Fidelity

| Target | R² |
|--------|----|
| CO     | 0.64 |
| NOx    | 0.68 |

The model captures major physical trends of turbine emissions. Reliability is strongest during standard low-emission operation; dense hexbin regions in the parity plots indicate high confidence in normal operating conditions.

---

### 2️⃣ Uncertainty Reliability

The calibration curve shows signs of **overconfidence** — predicted uncertainty intervals are narrower than the empirical coverage warrants. Uncertainty bounds should be widened before deployment in safety-critical environments.

---

### 3️⃣ Residual Analysis

Most residuals cluster around zero with no systematic bias during low-emission conditions. However, error variance increases during high CO emission events, indicating that high-emission combustion physics are more nonlinear and may benefit from additional physics constraints or targeted training data.

---

## 🖥️ Real-Time Dashboard

A live monitoring dashboard built with **Streamlit + Plotly**:

- Real-time emission predictions with 95% uncertainty bands
- Configurable alarm and warning thresholds
- Predicted vs actual scatter and PI width drift charts
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
│   ├── scientific_plots.py     # Publication-quality figures (PDF output)
│   └── *.csv / *.png           # Model outputs (generated at runtime)
├── build_dataset.py            # Data pipeline entry point
├── compare_models.py           # Side-by-side model comparison
└── requirements.txt
```

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt

# Create output directories (first time only)
mkdir models results data/raw data/processed

# 1. Fetch and process data
python build_dataset.py

# 2. Train models (can be run independently)
python -c "from src.training import run_xgboost; run_xgboost()"
python -c "from src.training import run_pinn; run_pinn()"

# 3. Compare models
python compare_models.py

# 4. Generate scientific figures
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
