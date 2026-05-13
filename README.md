# 🌍 Digital Twin Soft-Sensor for Power Plant Emission Monitoring

A Physics-Informed AI system for real-time prediction of gas turbine emissions using Digital Twin technology, uncertainty-aware machine learning, and thermodynamic feature engineering.

---

## 📌 Project Overview

Modern gas turbine power plants generate electricity efficiently, but they also emit harmful pollutants such as:

- **Carbon Monoxide (CO)**
- **Nitrogen Oxides (NOx)**

Monitoring these emissions in real time is challenging because traditional high-temperature emission sensors are:

- Expensive
- Slow to respond
- Difficult to maintain
- Prone to calibration drift

This project introduces a **Digital Twin Soft-Sensor** capable of predicting emissions instantly using operational sensor data and physics-informed machine learning.

---

## 🎯 Objective

The goal of this project is to build a **real-time virtual emissions sensor** that:

✅ Predicts turbine emissions from operational parameters  
✅ Detects when combustion becomes inefficient ("running dirty")  
✅ Provides uncertainty-aware predictions for safer deployment  
✅ Helps operators reduce emissions before environmental violations occur  

---

# 🧠 Core Idea: Physics + AI

Most machine learning models only learn patterns from raw data.

This project goes further by integrating **thermodynamic principles** directly into the learning process through a:

# 🔬 Physics-Informed Neural Network (PINN)

The model is trained not only to minimize prediction error, but also to respect known physical laws governing gas turbine combustion.

---

# ⚙️ Technical Architecture

## Layer 1 — Data Ingestion & Physics Feature Engineering

Raw operational turbine data is collected and transformed into physically meaningful features.

### Example Inputs

- Ambient temperature
- Pressure
- Airflow rate
- Fuel flow
- Turbine operating conditions

### Physics-Based Derived Features

Instead of relying only on raw sensor values, the system computes thermodynamic indicators such as:

- Compression Ratio
- Isentropic Efficiency
- Temperature Ratios
- Pressure Ratios

These variables are directly linked to combustion quality and emission behavior.

---

## Layer 2 — Hybrid AI Modeling

Two major modeling approaches were explored:

### 🌲 XGBoost Baseline Model

Used as a strong traditional machine learning benchmark.

### 🧠 Physics-Informed Neural Network (PINN)

The PINN uses a **Hybrid Loss Function**:

```python
Total Loss = Data Loss + Physics Loss

This means the model is penalized when:

- Predictions differ from measured emissions
- Predictions violate thermodynamic constraints

---

# 📉 Uncertainty Quantification

Industrial systems require not only accurate predictions, but also confidence estimation.

## Method Used — MC Dropout

The model performs multiple stochastic forward passes during inference:

```text
100 forward passes per prediction
```

The variation across predictions is used to estimate uncertainty.

## Benefits

✅ Detects unreliable predictions  
✅ Flags high-risk operating conditions  
✅ Provides confidence bands for operators  

---

# 📊 Scientific Results & Analysis

## 1️⃣ Prediction Fidelity (Accuracy Assessment)

Parity plots compare:

- Experimental ground truth
- Soft-sensor predictions

### Key Results

| Target | R² Score |
|--------|----------|
| CO     | 0.64     |
| NOx    | 0.68     |

### Interpretation

- The model captures major physical trends of turbine emissions
- Reliability is strongest during standard low-emission operation
- Dense hexbin regions indicate high confidence in normal operating conditions

---

## 📌 Placeholder — Parity Plot
<img width="1538" height="715" alt="Figure_1" src="https://github.com/user-attachments/assets/e9a11b34-0aea-49f2-afa1-32a10973463d" />

---

# 2️⃣ Uncertainty Reliability (Calibration Assessment)

This analysis evaluates whether model confidence matches actual prediction accuracy.

## Observation

The model currently shows signs of **overconfidence**:

- Predicted uncertainty intervals are too narrow
- Calibration curve falls below the ideal calibration line

## Implication

While the Digital Twin performs well operationally, uncertainty bounds should be widened before deployment in safety-critical environments.

---

## 📌 Placeholder — Calibration Plot
<img width="974" height="770" alt="Figure_2" src="https://github.com/user-attachments/assets/a5031121-2c2f-4c93-86e9-fc8f522ace15" />

---

# 3️⃣ Residual Analysis (Error Behavior)

Residual plots evaluate whether prediction errors are random or systematic.

## Findings

✅ Most residuals cluster around zero  
✅ Errors are largely unbiased during low-emission conditions  

However:

⚠️ Error variance increases during high CO emission events

## Interpretation

This suggests that:

- High-emission combustion physics are more nonlinear
- Additional physics constraints or more training data may improve robustness

---

## 📌 Placeholder — Residual Plot
<img width="914" height="731" alt="Figure_3" src="https://github.com/user-attachments/assets/484f83f6-7e37-41db-b64b-aa5c1dcd6343" />

---

# 🖥️ Real-Time Dashboard

A live monitoring dashboard was developed using **Streamlit**.

## Features

- Real-time emission predictions
- Live uncertainty bands
- Alarm thresholds
- Trend visualization
- Operational Digital Twin monitoring

---

## 📌 Placeholder — Dashboard Screenshot
<img width="954" height="449" alt="image" src="https://github.com/user-attachments/assets/3ebaeab4-53a0-4f5b-bde8-93e943564f05" />

---

# 📚 Data Sources & Credits

The data used in this project is sourced from credible academic and research repositories.

## Primary Dataset

### Gas Turbine CO and NOx Emission Data Set (2019)

Hosted by the **UCI Machine Learning Repository**

- DOI: `10.24432/C5WC95`
- Contains **36,733 hourly instances** collected from a working gas turbine
- Includes operational sensor measurements and emission outputs for:
  - Carbon Monoxide (CO)
  - Nitrogen Oxides (NOx)

---

## Research Reference

**Heysem Kaya, Pinar Tüfekci, and Erdinç Uzun (2019)**

> *"Predicting CO and NOx emissions from gas turbines: multi-layer perceptron and ensemble learning approaches and a new large-scale dataset."*

This research introduced the dataset and explored machine learning approaches for turbine emission prediction.

---

# 🧰 Tech Stack

## Machine Learning

- Python
- PyTorch
- XGBoost
- Scikit-learn

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn
- Plotly

## Deployment

- Streamlit
