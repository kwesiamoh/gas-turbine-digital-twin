"""
Training pipelines for XGBoost+MAPIE and PINN+MC-Dropout soft sensor models.

Entry points:
  run_xgboost(data_path)  — trains, evaluates, and saves XGBoost model
  run_pinn(data_path)     — trains, evaluates, and saves PINN model

Both functions:
  - Read from data/processed/syngas_features.csv
  - Write prediction CSVs to results/
  - Save model artefacts to models/
"""

from pathlib import Path
import sys
from src.uncertainty import (wrap_with_mapie, evaluate_predictions,
                             calibrate_mc_dropout)
from src.pinn_model import (SoftSensorPINN, physics_residual,
                            LAMBDA_PHYSICS, DROPOUT_RATE, MC_SAMPLES)
from src.feature_engineering import PHYSICS_FEATS
from src.preprocessing import RAW_SENSORS, TARGETS
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# Windows may expose a legacy CP-1252 stdout even though project messages are
# UTF-8. Reconfigure text output so status symbols cannot abort a training run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ALL_FEATURES = RAW_SENSORS + PHYSICS_FEATS
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def build_preprocessor() -> Pipeline:
    """
    RobustScaler: uses median and IQR — resistant to sensor spike outliers.
    SimpleImputer: median fill for any residual NaNs in the feature matrix.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])


def _load_splits(path: str) -> tuple:
    """Load processed CSV and reconstruct temporal train/test split by year."""
    df = pd.read_csv(path)
    train = df[df["year"] <= 2013].reset_index(drop=True)
    test = df[df["year"] >= 2014].reset_index(drop=True)
    return train, test


def _ensure_output_dirs():
    """Create model/result output directories relative to the active project root."""
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  XGBOOST PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

XGB_PARAM_GRID = {
    "n_estimators":     [300, 500, 800],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha":        [0, 0.1, 0.5],      # L1
    "reg_lambda":       [1.0, 2.0, 5.0],    # L2
    "min_child_weight": [1, 3, 5],
}


def tune_xgboost(X_train: pd.DataFrame,
                 y_col: np.ndarray,
                 target_name: str) -> dict:
    """
    RandomizedSearchCV with TimeSeriesSplit and fold-local preprocessing.

    The imputer/scaler is fitted independently inside every temporal CV fold,
    so feature statistics from later fold observations cannot leak backward
    into earlier training folds. The returned dict contains only XGBoost params.
    """
    print(f"    Tuning XGBoost for '{target_name}'...")
    tscv = TimeSeriesSplit(n_splits=5)
    base = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    param_grid = {f"model__{k}": v for k, v in XGB_PARAM_GRID.items()}
    search = RandomizedSearchCV(
        base,
        param_distributions=param_grid,
        n_iter=30,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_col)
    best_params = {
        key.removeprefix("model__"): value
        for key, value in search.best_params_.items()
    }
    print(f"      Best CV RMSE : {-search.best_score_:.4f}")
    print(f"      Best params  : {best_params}")
    return best_params


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract gain-based feature importance from a fitted XGBRegressor."""
    imp = model.feature_importances_
    return (pd.DataFrame({"feature": feature_names, "importance": imp})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True))


def run_xgboost(data_path: str = "data/processed/syngas_features.csv"):
    """
    Full XGBoost + MAPIE pipeline.

    Outputs:
      models/preprocessor_xgb.joblib
      models/mapie_co.joblib
      models/mapie_nox.joblib
      results/xgboost_predictions.csv
      results/xgboost_feature_importance.csv
    """
    _ensure_output_dirs()
    print("=" * 65)
    print("  XGBoost Soft Sensor + MAPIE Conformal Prediction")
    print("=" * 65)

    train, test = _load_splits(data_path)
    print(f"[1] Data loaded — Train: {len(train):,}  Test: {len(test):,}")

    # Preserve chronological order: the final 20% of 2011–2013 is reserved
    # exclusively for conformalization and is never seen during tuning/fitting.
    calib_size = max(1, int(0.20 * len(train)))
    fit_df = train.iloc[:-calib_size].reset_index(drop=True)
    calib_df = train.iloc[-calib_size:].reset_index(drop=True)

    X_fit, y_fit = fit_df[ALL_FEATURES], fit_df[TARGETS]
    X_calib, y_calib = calib_df[ALL_FEATURES], calib_df[TARGETS]
    X_test, y_test = test[ALL_FEATURES], test[TARGETS]

    print(f"[2] Preprocessing (impute + RobustScale) — Fit: {len(fit_df):,}  "
          f"Conformalize: {len(calib_df):,}")
    preprocessor = build_preprocessor()
    X_fit_proc = preprocessor.fit_transform(X_fit)
    X_calib_proc = preprocessor.transform(X_calib)
    X_test_proc = preprocessor.transform(X_test)
    joblib.dump(preprocessor, "models/preprocessor_xgb.joblib")

    all_metrics, all_mapie, all_fi = [], {}, {}

    print("[3] Tuning, fitting, and wrapping each target...")
    for target in TARGETS:
        y_fit_target = y_fit[target].values
        y_calib_target = y_calib[target].values

        best_params = tune_xgboost(X_fit, y_fit_target, target)
        best_xgb = XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        mapie = wrap_with_mapie(
            best_xgb, X_fit_proc, y_fit_target,
            X_calib_proc, y_calib_target, target
        )
        all_mapie[target] = mapie
        all_fi[target] = get_feature_importance(best_xgb, ALL_FEATURES)

        joblib.dump(mapie, f"models/mapie_{target.lower()}.joblib")
        print(f"    Saved → models/mapie_{target.lower()}.joblib")

    print("[4] Evaluating on test set (2014–2015)...")
    preds = {}
    for target in TARGETS:
        y_pred, y_pis = all_mapie[target].predict_interval(X_test_proc)
        # MAPIE v1 returns (n_samples, 2, n_confidence_levels).
        y_pis_2d = y_pis[:, :, 0]
        preds[target] = (y_pred, y_pis_2d)
        m = evaluate_predictions(
            y_test[target].values, y_pred,
            y_pis_2d[:, 0], y_pis_2d[:, 1], target
        )
        all_metrics.append(m)

    co_pred,  co_pis = preds["CO"]
    nox_pred, nox_pis = preds["NOx"]

    out = pd.DataFrame({
        "year":       test["year"].values,
        "y_true_CO":  y_test["CO"].values,
        "y_pred_CO":  co_pred,
        "pi_lo_CO":   co_pis[:, 0],
        "pi_hi_CO":   co_pis[:, 1],
        "y_true_NOx": y_test["NOx"].values,
        "y_pred_NOx": nox_pred,
        "pi_lo_NOx":  nox_pis[:, 0],
        "pi_hi_NOx":  nox_pis[:, 1],
    })
    out.to_csv("results/xgboost_predictions.csv", index=False)
    print("  Predictions saved → results/xgboost_predictions.csv")

    fi_combined = pd.concat([
        all_fi["CO"].assign(target="CO"),
        all_fi["NOx"].assign(target="NOx"),
    ])
    fi_combined.to_csv("results/xgboost_feature_importance.csv", index=False)

    print("\nXGBoost pipeline complete.\n")
    return all_metrics, all_mapie, preprocessor


# ─────────────────────────────────────────────────────────────────────────────
#  PINN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS = 120
BATCH_SIZE = 512
LR = 3e-4

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def _to_tensors(*arrays):
    return [torch.tensor(a).to(DEVICE) for a in arrays]


def _preprocess_pinn(train: pd.DataFrame, val: pd.DataFrame,
                     test: pd.DataFrame) -> tuple:
    """
    Fit preprocessor on training data. Targets are also scaled for
    better PINN convergence; the y_scaler is saved for inverse transform.

    Returns scaled numpy arrays for X and y, raw y_test, and the y_scaler.
    """
    print("[2] Preprocessing...")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])
    X_tr = pipe.fit_transform(train[ALL_FEATURES])
    X_va = pipe.transform(val[ALL_FEATURES])
    X_te = pipe.transform(test[ALL_FEATURES])

    y_tr = train[TARGETS].values.astype(np.float32)
    y_va = val[TARGETS].values.astype(np.float32)
    y_te = test[TARGETS].values.astype(np.float32)

    # Target scaling: RobustScaler fitted on train targets only
    y_scaler = RobustScaler()
    y_tr_scaled = y_scaler.fit_transform(y_tr)
    y_va_scaled = y_scaler.transform(y_va)

    joblib.dump(pipe,     "models/pinn_preprocessor.joblib")
    joblib.dump(y_scaler, "models/pinn_y_scaler.joblib")
    print("    Preprocessors saved → models/")

    return (X_tr.astype(np.float32), X_va.astype(np.float32),
            X_te.astype(np.float32),
            y_tr_scaled.astype(np.float32),
            y_va_scaled.astype(np.float32),
            y_te, y_scaler)


def train_pinn(X_tr, y_tr, X_va, y_va) -> tuple:
    """
    Training loop with early stopping (patience=20 epochs).

    Best checkpoint is saved to models/pinn_best.pt and reloaded after
    training. Gradient clipping (max_norm=1.0) prevents exploding gradients,
    which can occur when LAMBDA_PHYSICS is set high.

    Returns (model, history) where history contains per-epoch loss values.
    """
    print(
        f"[3] Training PINN on {DEVICE}  (epochs={EPOCHS}, λ={LAMBDA_PHYSICS})...")

    ds_tr = TensorDataset(*_to_tensors(X_tr, y_tr))
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE,
                       shuffle=True,  drop_last=True)
    ds_va = TensorDataset(*_to_tensors(X_va, y_va))
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    model = SoftSensorPINN(n_features=len(ALL_FEATURES)).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.05)
    mse_loss = nn.MSELoss()

    history = {"train_data": [], "train_phys": [],
               "train_total": [], "val": []}
    best_val_loss = float("inf")
    patience = 0
    patience_limit = 20

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_data, tr_phys = 0.0, 0.0

        for xb, yb in dl_tr:
            optimizer.zero_grad()
            xb = xb.detach().requires_grad_(True)
            pred = model(xb)
            l_data = mse_loss(pred, yb)
            l_phys = physics_residual(xb, pred)
            l_total = l_data + LAMBDA_PHYSICS * l_phys
            l_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_data += l_data.item()
            tr_phys += l_phys.item()

        scheduler.step()

        model.eval()
        val_data, val_phys = 0.0, 0.0
        for xb, yb in dl_va:
            xb = xb.detach().requires_grad_(True)
            pred = model(xb)
            l_val_data = mse_loss(pred, yb)
            l_val_phys = physics_residual(xb, pred)
            val_data += l_val_data.item()
            val_phys += l_val_phys.item()

        n_tr = len(dl_tr)
        n_va = len(dl_va)
        history["train_data"].append(tr_data / n_tr)
        history["train_phys"].append(tr_phys / n_tr)
        history["train_total"].append(
            (tr_data + LAMBDA_PHYSICS * tr_phys) / n_tr)
        history["val"].append(
            (val_data + LAMBDA_PHYSICS * val_phys) / n_va
        )

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Ep {epoch:3d}/{EPOCHS}  "
                  f"L_data={history['train_data'][-1]:.4f}  "
                  f"L_phys={history['train_phys'][-1]:.4f}  "
                  f"L_val_total={history['val'][-1]:.4f}")

        # Save best checkpoint and apply patience-based early stopping
        if history["val"][-1] < best_val_loss - 1e-4:
            best_val_loss = history["val"][-1]
            torch.save(model.state_dict(), "models/pinn_best.pt")
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"    Early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(
        "models/pinn_best.pt", map_location=DEVICE))
    print(f"    Best validation loss: {best_val_loss:.5f}")
    return model, history


def _mc_predictions(model, X: np.ndarray, y_scaler) -> tuple[np.ndarray, np.ndarray]:
    """Return MC-Dropout mean and standard deviation in original target units."""
    X_t = torch.tensor(X).to(DEVICE)
    mean_scaled, std_scaled = model.predict_with_uncertainty(X_t)
    mean_scaled = mean_scaled.cpu().numpy()
    std_scaled = std_scaled.cpu().numpy()
    return (y_scaler.inverse_transform(mean_scaled),
            std_scaled * y_scaler.scale_)


def evaluate_pinn(model, X_cal: np.ndarray, y_cal_raw: np.ndarray,
                  X_te: np.ndarray, y_te_raw: np.ndarray, y_scaler) -> tuple:
    """
    MC Dropout inference on the test set.

    Scales predictions back to original units using y_scaler.
    The dropout standard deviation is an epistemic uncertainty estimate; the
    A dedicated chronological calibration block converts MC standard deviations
    into normalized split-conformal prediction intervals.
    """
    print("[6] Evaluating PINN on test set (2014–2015)...")
    y_cal_pred, y_cal_std = _mc_predictions(model, X_cal, y_scaler)
    interval_scales = calibrate_mc_dropout(y_cal_raw, y_cal_pred, y_cal_std)
    print(f"    MC conformal scales: {dict(zip(TARGETS, interval_scales.round(3)))}")

    y_pred, y_std = _mc_predictions(model, X_te, y_scaler)

    metrics = []
    for i, target in enumerate(TARGETS):
        y_true = y_te_raw[:, i]
        y_mu = y_pred[:, i]
        y_s = y_std[:, i]
        lo = y_mu - interval_scales[i] * y_s
        hi = y_mu + interval_scales[i] * y_s
        m = evaluate_predictions(
            y_true, y_mu, lo, hi, target
        )
        metrics.append(m)

    return y_pred, y_std, interval_scales, metrics


def run_pinn(data_path: str = "data/processed/syngas_features.csv"):
    """
    Full PINN + MC Dropout pipeline.

    Outputs:
      models/pinn_preprocessor.joblib
      models/pinn_y_scaler.joblib
      models/pinn_best.pt
      models/pinn_final.pt
      results/pinn_predictions.csv
      results/pinn_training_history.csv
    """
    _ensure_output_dirs()
    print("=" * 65)
    print("  Physics-Informed Neural Network (PINN) Soft Sensor")
    print("=" * 65)

    train_df, test_df = _load_splits(data_path)

    # Keep the last block exclusively for interval calibration. The preceding
    # block is used for early stopping and cannot leak into calibration scores.
    calib_size = max(1, int(0.10 * len(train_df)))
    val_size = max(1, int(0.15 * len(train_df)))
    calib_df = train_df.iloc[-calib_size:].reset_index(drop=True)
    val_df = train_df.iloc[-(calib_size + val_size):-calib_size].reset_index(drop=True)
    train_core = train_df.iloc[:-(calib_size + val_size)].reset_index(drop=True)
    print(f"[1] Data loaded — Train: {len(train_core):,}  "
          f"Val: {len(val_df):,}  Calibrate: {len(calib_df):,}  "
          f"Test: {len(test_df):,}")

    X_tr, X_va, X_cal, y_tr, y_va, y_cal_raw, y_scaler = _preprocess_pinn(
        train_core, val_df, calib_df
    )
    preprocessor = joblib.load("models/pinn_preprocessor.joblib")
    X_te = preprocessor.transform(test_df[ALL_FEATURES]).astype(np.float32)
    y_te_raw = test_df[TARGETS].values.astype(np.float32)

    model, history = train_pinn(X_tr, y_tr, X_va, y_va)
    torch.save(model.state_dict(), "models/pinn_final.pt")
    print("    Final model saved → models/pinn_final.pt")

    y_pred, y_std, interval_scales, metrics = evaluate_pinn(
        model, X_cal, y_cal_raw, X_te, y_te_raw, y_scaler
    )
    joblib.dump(interval_scales, "models/pinn_interval_scales.joblib")

    out = pd.DataFrame({
        "year":       test_df["year"].values,
        "y_true_CO":  y_te_raw[:, 0],
        "y_pred_CO":  y_pred[:, 0],
        "y_std_CO":   y_std[:, 0],
        "pi_lo_CO":   y_pred[:, 0] - interval_scales[0] * y_std[:, 0],
        "pi_hi_CO":   y_pred[:, 0] + interval_scales[0] * y_std[:, 0],
        "y_true_NOx": y_te_raw[:, 1],
        "y_pred_NOx": y_pred[:, 1],
        "y_std_NOx":  y_std[:, 1],
        "pi_lo_NOx":  y_pred[:, 1] - interval_scales[1] * y_std[:, 1],
        "pi_hi_NOx":  y_pred[:, 1] + interval_scales[1] * y_std[:, 1],
    })
    out.to_csv("results/pinn_predictions.csv", index=False)
    print("  Predictions saved → results/pinn_predictions.csv")

    pd.DataFrame(history).to_csv(
        "results/pinn_training_history.csv", index=False)

    print("\nPINN pipeline complete.\n")
    return model, metrics, history
