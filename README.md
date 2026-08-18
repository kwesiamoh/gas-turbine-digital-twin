# 🌍 Digital Twin Soft-Sensor for Power Plant Emission Monitoring

A physics-informed, uncertainty-aware machine-learning framework for estimating gas-turbine **CO** and **NOx** emissions from routinely measured operating variables.

The project combines thermodynamic feature engineering, a tree-based conformal model, a physics-informed neural model, temporal validation, uncertainty calibration, and a Streamlit monitoring dashboard.

---

## 📌 Project Overview

Continuous emission measurements can be expensive to maintain and can suffer from delay or calibration drift. A **soft sensor** estimates emissions from other plant measurements so operators can monitor combustion behaviour continuously and identify high-risk operating conditions earlier.

This project compares two complementary soft-sensor approaches:

- **XGBoost + MAPIE** — strong nonlinear tabular regression with split-conformal uncertainty.
- **Physics-Informed Neural Network (PINN) + MC Dropout** — a multi-output neural model with soft physical sensitivity constraints, post-hoc point calibration, and held-out uncertainty calibration.

The final test period is always **2014–2015**. No 2014–2015 target values are used for model fitting, checkpoint selection, bias calibration, or interval calibration.

---


## 🛠️ Fixes and Improvements Implemented

Fixed **MC Dropout + BatchNorm** so only Dropout remains active during uncertainty sampling.
- Replaced the old hand-written CO physics equation with **autograd-based directional physics constraints**.
- Added **NOx physics constraints** to the PINN.
- Removed **XGBoost conformal-calibration leakage** using a chronological holdout.
- Removed **preprocessing leakage** inside `TimeSeriesSplit`.
- Added **held-out PINN point-prediction calibration** to reduce systematic temporal bias.
- Added **held-out MC-Dropout uncertainty calibration** instead of assuming `1.96 × std` gives 95% coverage.
- Improved **year handling and dataset compatibility** to avoid unsafe temporal reconstruction.
- Fixed **Streamlit dashboard** threshold, playback, and indexing issues.
- Added **prediction bias and year-by-year evaluation** for 2014 and 2015.
- Fixed **MAPIE dependency compatibility**, output-directory handling, and methodology wording in the documentation.

These changes improve leakage prevention, uncertainty reliability, temporal evaluation, and scientific defensibility while keeping the original project structure, models, features, targets, and train/test protocol unchanged.


---

## 🎯 Objectives

The framework is designed to:

- Predict **CO** and **NOx** from turbine operating measurements.
- Detect inefficient/high-emission operating conditions.
- Quantify predictive uncertainty instead of returning point predictions alone.
- Use physically motivated information without forcing the network to obey an invented closed-form emissions equation.
- Test generalisation under a strict chronological split rather than an easier random split.
- Expose temporal drift through overall and year-by-year diagnostics.

---

## 🗃️ Dataset and Temporal Protocol

**Dataset:** UCI Gas Turbine CO and NOx Emission Data Set (2011–2015)

**Raw inputs (9):**

`AT`, `AP`, `AH`, `AFDP`, `GTEP`, `TIT`, `TAT`, `CDP`, `TEY`

**Targets:**

`CO`, `NOx`

### Top-level split

```text
2011 ───── 2012 ───── 2013 │ 2014 ───── 2015
       development data     │   untouched test
```

The project never uses the 2014–2015 targets to tune or calibrate either model.

### XGBoost internal split

Within 2011–2013:

```text
first 80%                         final 20%
fit + TimeSeriesSplit tuning  │  MAPIE conformalization
```

Preprocessing is fitted independently inside each `TimeSeriesSplit` fold during hyperparameter search, preventing future-fold feature statistics from leaking backward.

### PINN internal split

Within 2011–2013:

```text
first 75%     next 15%      next 5%          final 5%
model fit  │  validation  │ point calibration │ interval calibration
```

- **Model fit:** neural-network parameter learning.
- **Validation:** early stopping/checkpoint selection.
- **Point calibration:** corrects systematic level/scale bias.
- **Interval calibration:** calibrates MC-Dropout uncertainty on data untouched by model fitting, checkpoint selection, and point-calibrator fitting.

This keeps the overall project structure unchanged while giving each calibration step its own chronological holdout.

---

## ⚙️ Physics-Derived Features

Seven engineered features augment the nine raw measurements:

| Feature | Formula | Interpretation |
|---|---|---|
| `T_ratio` | `TIT / TAT` | turbine temperature-ratio proxy |
| `compression` | `CDP / AP × 1000` | scaled compressor-pressure proxy |
| `T_drop` | `TIT - TAT` | turbine temperature drop |
| `humidity_abs` | `f(AH, AT)` | absolute-humidity proxy using the Magnus approximation |
| `fouling_idx` | `AFDP / AP × 1000` | scaled filter-pressure-drop proxy |
| `specific_work` | `TEY / CDP` | energy yield per unit compression |
| `dT_AT_TIT` | `TIT - AT` | total temperature rise from ambient to turbine inlet |

These features are retained for both XGBoost and PINN so the model comparison uses the same information base.

---

## 🌲 Model 1 — XGBoost + MAPIE

The XGBoost pipeline uses:

- `RandomizedSearchCV`
- 5-fold `TimeSeriesSplit`
- fold-local median imputation + `RobustScaler`
- a final chronological conformalization block
- MAPIE `SplitConformalRegressor`

The model produces a point estimate and a nominal **95% split-conformal interval** for each pollutant.

Main outputs:

```text
models/preprocessor_xgb.joblib
models/mapie_co.joblib
models/mapie_nox.joblib
results/xgboost_predictions.csv
results/xgboost_feature_importance.csv
```

---

## 🧠 Model 2 — Physics-Informed Neural Network

### Architecture

```text
Input
  ↓
BatchNorm1d
  ↓
Linear(128) + SiLU + Dropout
  ↓
Linear(64) + SiLU + Dropout
  ↓
Linear(32) + SiLU
  ↓
┌─────────────┬──────────────┐
│   CO head   │   NOx head   │
└─────────────┴──────────────┘
```

The shared trunk allows CO and NOx to learn common turbine-state representations while retaining separate output heads.

### Training objective

```text
L_total = L_data + λ × L_physics
```

`L_data` is the multi-output MSE. `L_physics` is a soft derivative-sign penalty computed with PyTorch autograd.

### CO directional priors

The network is penalised when its local sensitivity violates:

```text
d(CO)/d(TIT)          ≤ 0
d(CO)/d(compression)  ≤ 0
d(CO)/d(humidity)     ≥ 0
```

### NOx directional priors

NOx now has its own physics guidance:

```text
d(NOx)/d(TIT)       ≥ 0
d(NOx)/d(humidity)  ≤ 0
```

These are **soft directional priors**, not hard mass-balance equations. They guide local trends without forcing predictions toward an arbitrary hand-written emissions formula.

Training uses:

- AdamW
- cosine-annealing learning rate
- gradient clipping
- early stopping
- RobustScaler target scaling

---

## 🎯 PINN Point-Prediction Calibration

Temporal drift can preserve correlation while shifting the prediction level. To address systematic bias without touching the test set, the PINN uses a held-out affine calibrator for each target:

```text
y_calibrated = slope × y_raw + intercept
```

The slope is constrained to be non-negative so calibration does not reverse the ranking learned by the network.

The calibrator is fitted only on the dedicated **point-calibration block** from late 2011–2013 development data.

Saved artifact:

```text
models/pinn_calibration.joblib
```

---

## 📉 PINN Uncertainty Calibration

MC Dropout is run with **BatchNorm frozen** and only Dropout layers active. This avoids test-batch statistics modifying BatchNorm state and also supports single-sample inference.

For each observation, 100 stochastic forward passes produce:

```text
predictive mean
MC-Dropout standard deviation
```

A separate interval-calibration block computes normalized residual scores:

```text
score = |y_true - y_pred| / max(MC_std, ε)
```

The finite-sample 95% conformal quantile `q` is then used to form:

```text
lower = y_pred - q × MC_std
upper = y_pred + q × MC_std
```

This retains the relative MC-Dropout uncertainty pattern while calibrating its scale on held-out observations.

**Important:** a nominal 95% interval can still miss 95% coverage when the future operating regime differs substantially from the calibration regime. That is why the project reports empirical test coverage and year-by-year diagnostics rather than assuming calibration remains perfect under drift.

Calibration diagnostics are saved to:

```text
results/pinn_calibration_summary.csv
```

---

## 📊 Evaluation

Run:

```bash
python compare_models.py
```

The comparison reports for both CO and NOx:

- RMSE
- MAE
- R²
- mean prediction bias
- empirical interval coverage
- average interval width

Outputs:

```text
results/model_comparison.csv
results/model_comparison_by_year.csv
results/model_comparison.png
```

`model_comparison_by_year.csv` separates 2014 and 2015 so temporal degradation is visible instead of being hidden inside one aggregate score.

### Interpreting R²

A negative test R² does not mean the code failed. It means the model generalised worse than a constant-mean baseline on that future period. In this project, that should trigger investigation of temporal bias and operating-condition drift rather than replacement with a random train/test split.

### Interpreting coverage

For a nominal 95% interval, empirical coverage near 0.95 is desirable, but interval width matters too. Very high coverage obtained only through extremely wide intervals is not necessarily operationally useful.

---

## 🖥️ Streamlit Monitoring Dashboard

Launch with:

```bash
streamlit run dashboard/app.py
```

The dashboard provides:

- calibrated CO/NOx predictions
- nominal 95% uncertainty bands
- configurable warning/alarm thresholds
- predicted-vs-actual diagnostics
- interval-width monitoring
- live playback through the chronological test sequence

If prediction files do not exist, the dashboard falls back to synthetic data for UI testing.

---

## 🗂️ Project Structure

```text
gas-turbine-digital/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── models/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── pinn_model.py
│   ├── uncertainty.py
│   └── training.py
├── dashboard/
│   └── app.py
├── results/
├── build_dataset.py
├── compare_models.py
├── requirements.txt
└── README.md
```

The project layout and main entry points remain unchanged.

---

## 🚀 Quickstart

From the repository root:

```bash
pip install -r requirements.txt

# 1. Fetch, validate and engineer the dataset
python build_dataset.py

# 2. Train XGBoost + MAPIE
python -c "from src.training import run_xgboost; run_xgboost()"

# 3. Train and calibrate the PINN
python -c "from src.training import run_pinn; run_pinn()"

# 4. Compare models and generate diagnostics
python compare_models.py

# 5. Launch dashboard
streamlit run dashboard/app.py
```

You do **not** need to manually create `models/` or `results/`; the training entry points create them when required.

---

## 🔁 Recommended Experiment Workflow

After changing model assumptions or hyperparameters:

```text
1. Retrain
2. Compare overall metrics
3. Inspect 2014 vs 2015 metrics
4. Check mean bias
5. Check interval coverage and width
6. Inspect feature importance / residual plots
7. Only then update any performance claims in reports or presentations
```

Do not tune parameters against the 2014–2015 test results. Use the test period only for final evaluation.

---

## 📚 Data Source

**Gas Turbine CO and NOx Emission Data Set (2019)**  
UCI Machine Learning Repository  
DOI: `10.24432/C5WC95`

36,733 hourly observations from a gas turbine power plant in Turkey, covering 2011–2015.

Research reference:

Heysem Kaya, Pinar Tüfekci, and Erdinç Uzun (2019), *Predicting CO and NOx emissions from gas turbines: multi-layer perceptron and ensemble learning approaches and a new large-scale dataset.*

---

## 🧰 Tech Stack

| Category | Libraries |
|---|---|
| Machine learning | PyTorch, XGBoost, Scikit-learn, MAPIE |
| Data processing | Pandas, NumPy |
| Visualisation | Matplotlib, Plotly |
| Deployment | Streamlit |
| Persistence | Joblib, PyTorch checkpoints |

---
