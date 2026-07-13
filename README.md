# 🌍 Digital Twin Soft-Sensor for Power Plant Emission Monitoring

This project develops a digital twin soft-sensor for real-time gas turbine emission monitoring.

The system predicts **Carbon Monoxide (CO)** and **Nitrogen Oxides (NOx)** from operational turbine data using physics-informed machine learning, uncertainty estimation, and thermodynamic feature engineering.

The aim is to replace slow or difficult-to-maintain physical emission measurements with a virtual sensor that can estimate emissions continuously from signals already available in the plant.

---

## 📌 Project Overview

Gas turbines are widely used for electricity generation because they are flexible, efficient, and able to respond quickly to changing power demand.

At the same time, their combustion process can produce pollutants such as **CO** and **NOx**. These emissions are strongly affected by operating conditions such as temperature, pressure, humidity, air flow, turbine load, and combustion efficiency.

Measuring emissions directly in real time is not always straightforward. High-temperature emission sensors can be expensive, slow to respond, difficult to maintain, and affected by calibration drift.

This project approaches the problem through a **digital twin soft-sensor**: a data-driven virtual sensor that uses turbine operating conditions to estimate emissions instantly.

The model is designed not only to predict emissions, but also to give operators a clearer view of when the turbine may be moving toward inefficient or higher-emission combustion.

---

## 🎯 Objective

The objective is to build a real-time virtual emissions sensor that can:

- predict **CO** and **NOx** emissions from operational turbine parameters
- identify operating conditions linked to inefficient combustion
- provide uncertainty-aware predictions instead of only point estimates
- support earlier intervention before emissions exceed acceptable limits
- connect machine learning predictions with physically meaningful turbine behavior

The focus is not only accuracy. The model should also be interpretable enough to make sense in an engineering context.

---

## 🧠 Core Idea: Physics + Machine Learning

A purely data-driven model can learn correlations in historical turbine data, but it may not always respect the physical behavior of the system.

This project adds thermodynamic structure to the learning process.

The model uses physics-derived features and a physics-informed neural network design so that predictions are guided by both:

- observed emission measurements
- known relationships between turbine operating conditions and combustion behavior

This is especially important for emission monitoring, where the model should behave sensibly under changing operating regimes.

Rather than asking the model to learn everything from raw sensor values, the pipeline gives it features that already carry engineering meaning.

---

## ⚙️ Technical Architecture

## Layer 1 — Data Ingestion and Physics Feature Engineering

The first layer validates raw turbine operating data and transforms it into physically meaningful features.

### Raw Sensor Inputs

| Variable | Description |
|---|---|
| `AT` | Ambient temperature |
| `AP` | Ambient pressure |
| `AH` | Ambient humidity |
| `AFDP` | Air filter difference pressure |
| `GTEP` | Gas turbine exhaust pressure |
| `TIT` | Turbine inlet temperature |
| `TAT` | Turbine after temperature |
| `CDP` | Compressor discharge pressure |
| `TEY` | Turbine energy yield |

From these signals, the pipeline derives additional features that describe turbine behavior more directly.

### Physics-Derived Features

| Feature | Formula | Physical meaning |
|---|---|---|
| `T_ratio` | `TIT / TAT` | Proxy for expansion behavior across the turbine |
| `compression` | `CDP / AP` | Compressor pressure ratio |
| `T_drop` | `TIT - TAT` | Temperature drop across the turbine |
| `humidity_abs` | `f(AH, AP, AT)` | Absolute humidity estimated from ambient conditions |
| `fouling_idx` | `AFDP / AP` | Normalized air filter fouling indicator |
| `specific_work` | `TEY / CDP` | Energy output relative to compression level |
| `dT_AT_TIT` | `TIT - AT` | Temperature rise from ambient air to turbine inlet |

These features make the learning problem more grounded in turbine physics.

For example, humidity affects combustion temperature and NOx formation, while pressure ratios and turbine temperature differences are linked to the thermodynamic state of the machine.

---

## Layer 2 — Hybrid AI Modelling

The project compares two modelling approaches, both with uncertainty-aware outputs.

### 🌲 XGBoost + MAPIE

XGBoost is used as a strong tree-based baseline.

Training is performed with temporal validation rather than random splitting, so the model is tested on future-like data instead of mixed historical samples.

Main setup:

- `RandomizedSearchCV` for hyperparameter search
- `TimeSeriesSplit` with 5 folds
- no temporal shuffling
- split-conformal prediction using `MAPIE`
- uncertainty intervals without assuming Gaussian residuals

This gives a practical benchmark: a strong classical machine learning model with calibrated prediction intervals.

---

### 🧠 Physics-Informed Neural Network

The second model is a physics-informed neural network designed for multi-target emission prediction.

Architecture:

```text
BatchNorm
→ Linear(128, SiLU)
→ Linear(64, SiLU)
→ Linear(32, SiLU)
→ CO head / NOx head

The model predicts **CO** and **NOx** through separate output heads while sharing the same learned representation of turbine operating conditions.
The training objective combines data fitting with a physics-based penalty:

L_total = L_data + λ · L_physics
```

- `L_data` measures the prediction error against observed emissions.
- `L_physics` penalises predictions that violate expected thermodynamic directionality, especially the relationship between CO, turbine inlet temperature, compression, and humidity.

Setting `λ = 0` removes the physics penalty and recovers a standard MLP baseline. This makes it possible to compare the effect of adding physical constraints directly.

Training setup:

- AdamW optimizer
- cosine annealing learning-rate schedule
- gradient clipping
- early stopping with patience of 20 epochs

---

## 📉 Uncertainty Quantification — MC Dropout

For the neural network, uncertainty is estimated using Monte Carlo Dropout.

Dropout remains active during inference, and the model performs 100 stochastic forward passes for each prediction.

The mean of these passes is used as the final prediction, while the spread gives an estimate of predictive uncertainty.

```text
95% Prediction Interval = mean ± 1.96 × standard deviation
```

This allows the model to communicate when it is confident and when the operating condition is less familiar or harder to predict.

Uncertainty is important in this setting because emission monitoring is operationally sensitive. A point estimate alone may look precise, even when the model is uncertain.

The uncertainty layer helps flag:

- less reliable predictions
- high-risk operating conditions
- periods where operator attention may be needed
- cases where the model moves outside familiar operating regimes

---

## 📊 Results

### 1️⃣ Prediction Fidelity

| Target | R² |
|---|---|
| CO | 0.64 |
| NOx | 0.68 |

The model captures the main emission trends in the turbine dataset.

Performance is strongest during standard low-emission operating conditions, where the training data is denser and the combustion behavior is more regular.

The parity plots show this clearly: dense regions follow the expected prediction trend, while higher-emission events are more difficult to estimate precisely.

This pattern is realistic for emission modelling. Normal operation is usually well represented in historical data, while abnormal or high-emission regimes are less frequent and more nonlinear.

---

### 2️⃣ Uncertainty Reliability

The calibration analysis shows that the uncertainty estimates are not fully conservative.

The model tends to be overconfident, meaning the predicted intervals are narrower than the empirical error distribution would require.

This is an important result rather than a failure. It shows that the current uncertainty estimates are useful diagnostically, but should be widened or recalibrated before deployment in safety-critical monitoring.

Possible improvements include:

- conformal calibration of neural network intervals
- wider empirical prediction intervals
- target-specific uncertainty scaling
- additional high-emission training samples
- separate calibration for normal and abnormal operating regimes

---

### 3️⃣ Residual Analysis

Most residuals are centered close to zero during normal low-emission operation.

This suggests that the model is not strongly biased under common operating conditions.

However, residual variance increases during high CO events. These cases are harder to predict because they are less frequent and may involve more complex combustion dynamics.

This result points to a useful direction for improvement: the model would likely benefit from more targeted representation of high-emission regimes, either through additional data, weighted training, or stronger physics constraints.

---

## 🖥️ Real-Time Dashboard

The project includes a live monitoring dashboard built with **Streamlit** and **Plotly**.

The dashboard is designed to make the model outputs usable from an operator perspective, not only as offline evaluation metrics.

Main dashboard functions:

- real-time CO and NOx prediction
- 95% uncertainty bands
- configurable warning and alarm thresholds
- predicted vs actual emission visualisation
- prediction interval width tracking
- live simulation mode over the test set

The dashboard helps translate model predictions into operational signals: whether emissions are stable, whether uncertainty is increasing, and whether the turbine may be entering a higher-risk state.
<img width="954" height="449" alt="Screenshot 2026-05-13 084055" src="https://github.com/user-attachments/assets/5c026ed2-ed78-418c-b4d0-bd73e6cdd5d7" />

---

## 🗂️ Project Structure

```text
project/
├── data/
│   ├── raw/                    # Unmodified UCI download
│   └── processed/              # Validated data and physics features
├── notebooks/                  # EDA and experiments
├── models/                     # Saved model artefacts (.joblib, .pt)
├── src/
│   ├── preprocessing.py        # Data fetch, validation, train/test split
│   ├── feature_engineering.py  # Physics-derived feature construction
│   ├── pinn_model.py           # PINN architecture and physics residual
│   ├── uncertainty.py          # MAPIE wrapper and evaluation utilities
│   └── training.py             # Training pipelines for both models
├── dashboard/
│   └── app.py                  # Streamlit monitoring dashboard
├── results/
│   ├── scientific_plots.py     # Publication-quality figures
│   └── *.csv / *.png           # Generated model outputs
├── build_dataset.py            # Data pipeline entry point
├── compare_models.py           # Side-by-side model comparison
└── requirements.txt
```

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt

# Create output directories
mkdir models results data/raw data/processed

# 1. Fetch and process data
python build_dataset.py

# 2. Train models
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
UCI Machine Learning Repository  
DOI: `10.24432/C5WC95`

The dataset contains **36,733 hourly observations** from a working gas turbine in Turkey, covering the period from 2011 to 2015.

The split is temporal to avoid leakage:

| Split | Period |
|---|---|
| Train | 2011–2013 |
| Test | 2014–2015 |

**Research reference:**

Heysem Kaya, Pinar Tüfekci, and Erdinç Uzun (2019).  
*Predicting CO and NOx emissions from gas turbines: multi-layer perceptron and ensemble learning approaches and a new large-scale dataset.*

---

## 🧰 Tech Stack

| Category | Libraries |
|---|---|
| Machine Learning | PyTorch, XGBoost, Scikit-learn, MAPIE |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Deployment | Streamlit |

---

## 🧪 Research Contribution

This project shows how a digital twin soft-sensor can be built from standard turbine operating data.

The main contribution is the combination of:

- operational gas turbine data
- thermodynamic feature engineering
- physics-informed neural network training
- uncertainty-aware prediction
- real-time dashboard deployment

The results show that the model can capture the dominant emission behavior of the turbine, especially under normal operating conditions.

The calibration and residual analyses also show where the model still needs care: high-emission events are harder to predict, and uncertainty intervals need stronger calibration before safety-critical use.

This makes the project useful not only as a prediction pipeline, but also as an engineering study of where machine learning performs well and where physical complexity still matters.

---

## 🔍 Key Takeaway

A virtual emissions sensor can estimate CO and NOx continuously from turbine operating data, while also indicating when its predictions are less certain.

For plant operation, this type of system can support earlier detection of inefficient combustion, better emission awareness, and more informed intervention before regulatory or environmental limits become a problem.

## ## 🤖 Development Process

This project was developed with support from Claude Code as an AI-assisted programming tool. The core concept, architecture, technical decisions, and implementation direction are my own.
