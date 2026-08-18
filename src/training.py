"""
Training pipelines for XGBoost+MAPIE and PINN+MC-Dropout soft sensor models.

Entry points:
  run_xgboost(data_path)  — trains, evaluates, and saves XGBoost model
  run_pinn(data_path)     — trains, calibrates, evaluates, and saves PINN model

Both functions preserve the top-level temporal protocol:
  Training/calibration data: 2011–2013
  Final untouched test data: 2014–2015
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from src.feature_engineering import PHYSICS_FEATS
from src.pinn_model import (
    LAMBDA_PHYSICS,
    MC_SAMPLES,
    SoftSensorPINN,
    physics_residual,
)
from src.preprocessing import RAW_SENSORS, TARGETS
from src.uncertainty import (
    ALPHA,
    apply_affine_calibrator,
    calibrated_interval,
    conformal_mc_scale,
    evaluate_predictions,
    fit_affine_calibrator,
    wrap_with_mapie,
)

warnings.filterwarnings("ignore")

ALL_FEATURES = RAW_SENSORS + PHYSICS_FEATS
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def build_preprocessor() -> Pipeline:
    """Median imputation followed by RobustScaler."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])


def _load_splits(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the processed CSV and reconstruct the temporal train/test split."""
    df = pd.read_csv(path)
    train = df[df["year"] <= 2013].reset_index(drop=True)
    test = df[df["year"] >= 2014].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(
            "Temporal split is empty. Expected 2011–2013 train and 2014–2015 test rows."
        )
    return train, test


def _ensure_output_dirs():
    """Create model/result output directories relative to the project root."""
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  XGBOOST PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

XGB_PARAM_GRID = {
    "n_estimators": [300, 500, 800],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1.0, 2.0, 5.0],
    "min_child_weight": [1, 3, 5],
}


def tune_xgboost(X_train: pd.DataFrame,
                 y_col: np.ndarray,
                 target_name: str) -> dict:
    """TimeSeriesSplit tuning with preprocessing fitted inside every fold."""
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
    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_xgboost(data_path: str = "data/processed/syngas_features.csv"):
    """Full XGBoost + MAPIE pipeline."""
    _ensure_output_dirs()
    print("=" * 65)
    print("  XGBoost Soft Sensor + MAPIE Conformal Prediction")
    print("=" * 65)

    train, test = _load_splits(data_path)
    print(f"[1] Data loaded — Train: {len(train):,}  Test: {len(test):,}")

    # Final 20% of the 2011–2013 block is untouched during tuning/fitting and
    # used only to conformalize the final XGBoost estimators.
    calib_size = max(1, int(0.20 * len(train)))
    fit_df = train.iloc[:-calib_size].reset_index(drop=True)
    calib_df = train.iloc[-calib_size:].reset_index(drop=True)

    X_fit, y_fit = fit_df[ALL_FEATURES], fit_df[TARGETS]
    X_calib, y_calib = calib_df[ALL_FEATURES], calib_df[TARGETS]
    X_test, y_test = test[ALL_FEATURES], test[TARGETS]

    print(
        f"[2] Preprocessing (impute + RobustScale) — Fit: {len(fit_df):,}  "
        f"Conformalize: {len(calib_df):,}"
    )
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
            best_xgb,
            X_fit_proc,
            y_fit_target,
            X_calib_proc,
            y_calib_target,
            target,
        )
        all_mapie[target] = mapie
        all_fi[target] = get_feature_importance(best_xgb, ALL_FEATURES)

        joblib.dump(mapie, f"models/mapie_{target.lower()}.joblib")
        print(f"    Saved → models/mapie_{target.lower()}.joblib")

    print("[4] Evaluating on test set (2014–2015)...")
    preds = {}
    for target in TARGETS:
        y_pred, y_pis = all_mapie[target].predict_interval(X_test_proc)
        y_pis_2d = y_pis[:, :, 0]
        preds[target] = (y_pred, y_pis_2d)
        all_metrics.append(evaluate_predictions(
            y_test[target].values,
            y_pred,
            y_pis_2d[:, 0],
            y_pis_2d[:, 1],
            target,
        ))

    co_pred, co_pis = preds["CO"]
    nox_pred, nox_pis = preds["NOx"]

    out = pd.DataFrame({
        "year": test["year"].values,
        "y_true_CO": y_test["CO"].values,
        "y_pred_CO": co_pred,
        "pi_lo_CO": co_pis[:, 0],
        "pi_hi_CO": co_pis[:, 1],
        "y_true_NOx": y_test["NOx"].values,
        "y_pred_NOx": nox_pred,
        "pi_lo_NOx": nox_pis[:, 0],
        "pi_hi_NOx": nox_pis[:, 1],
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
PINN_VAL_FRACTION = 0.15
PINN_CALIB_FRACTION = 0.10

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def _to_tensors(*arrays):
    return [torch.tensor(a).to(DEVICE) for a in arrays]


def _split_pinn_training_block(train_df: pd.DataFrame) -> tuple:
    """
    Chronologically split 2011–2013 into fit, validation and calibration blocks.

    The final calibration block is itself divided chronologically:
      first half  -> affine point/bias calibration
      second half -> MC-Dropout interval calibration

    Thus the interval-calibration observations are untouched by model fitting,
    checkpoint selection, and point-calibrator fitting.
    """
    n = len(train_df)
    calib_size = max(4, int(PINN_CALIB_FRACTION * n))
    val_size = max(2, int(PINN_VAL_FRACTION * n))
    fit_end = n - val_size - calib_size
    if fit_end < BATCH_SIZE:
        raise ValueError(
            "Not enough rows for the configured PINN temporal split. "
            "Reduce validation/calibration fractions or batch size."
        )

    train_core = train_df.iloc[:fit_end].reset_index(drop=True)
    val_df = train_df.iloc[fit_end:fit_end + val_size].reset_index(drop=True)
    calib_df = train_df.iloc[fit_end + val_size:].reset_index(drop=True)

    bias_size = len(calib_df) // 2
    bias_cal_df = calib_df.iloc[:bias_size].reset_index(drop=True)
    interval_cal_df = calib_df.iloc[bias_size:].reset_index(drop=True)
    if len(bias_cal_df) < 2 or len(interval_cal_df) < 2:
        raise ValueError("PINN calibration partitions must each contain >= 2 rows")

    return train_core, val_df, bias_cal_df, interval_cal_df


def _preprocess_pinn(train: pd.DataFrame, val: pd.DataFrame,
                     bias_cal: pd.DataFrame, interval_cal: pd.DataFrame,
                     test: pd.DataFrame) -> tuple:
    """Fit preprocessing/target scaling on model-fit data only."""
    print("[2] Preprocessing...")
    pipe = build_preprocessor()
    X_tr = pipe.fit_transform(train[ALL_FEATURES])
    X_va = pipe.transform(val[ALL_FEATURES])
    X_bias = pipe.transform(bias_cal[ALL_FEATURES])
    X_interval = pipe.transform(interval_cal[ALL_FEATURES])
    X_te = pipe.transform(test[ALL_FEATURES])

    y_tr = train[TARGETS].values.astype(np.float32)
    y_va = val[TARGETS].values.astype(np.float32)
    y_bias = bias_cal[TARGETS].values.astype(np.float32)
    y_interval = interval_cal[TARGETS].values.astype(np.float32)
    y_te = test[TARGETS].values.astype(np.float32)

    y_scaler = RobustScaler()
    y_tr_scaled = y_scaler.fit_transform(y_tr)
    y_va_scaled = y_scaler.transform(y_va)

    joblib.dump(pipe, "models/pinn_preprocessor.joblib")
    joblib.dump(y_scaler, "models/pinn_y_scaler.joblib")
    print("    Preprocessors saved → models/")

    return (
        X_tr.astype(np.float32),
        X_va.astype(np.float32),
        X_bias.astype(np.float32),
        X_interval.astype(np.float32),
        X_te.astype(np.float32),
        y_tr_scaled.astype(np.float32),
        y_va_scaled.astype(np.float32),
        y_bias,
        y_interval,
        y_te,
        y_scaler,
    )


def train_pinn(X_tr, y_tr, X_va, y_va) -> tuple:
    """Train with derivative physics regularisation and temporal early stopping."""
    print(
        f"[3] Training PINN on {DEVICE}  "
        f"(epochs={EPOCHS}, λ={LAMBDA_PHYSICS})..."
    )

    ds_tr = TensorDataset(*_to_tensors(X_tr, y_tr))
    dl_tr = DataLoader(
        ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    ds_va = TensorDataset(*_to_tensors(X_va, y_va))
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    model = SoftSensorPINN(n_features=len(ALL_FEATURES)).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.05
    )
    mse_loss = nn.MSELoss()

    history = {
        "train_data": [],
        "train_phys": [],
        "train_total": [],
        "val": [],
    }
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
            (tr_data + LAMBDA_PHYSICS * tr_phys) / n_tr
        )
        history["val"].append(
            (val_data + LAMBDA_PHYSICS * val_phys) / n_va
        )

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"    Ep {epoch:3d}/{EPOCHS}  "
                f"L_data={history['train_data'][-1]:.4f}  "
                f"L_phys={history['train_phys'][-1]:.4f}  "
                f"L_val_total={history['val'][-1]:.4f}"
            )

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
        "models/pinn_best.pt", map_location=DEVICE
    ))
    print(f"    Best validation loss: {best_val_loss:.5f}")
    return model, history


def _mc_predict_original(model, X: np.ndarray, y_scaler) -> tuple:
    """Run MC Dropout and inverse-transform both mean and standard deviation."""
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    mean_scaled, std_scaled = model.predict_with_uncertainty(
        X_t, n_samples=MC_SAMPLES
    )
    mean_scaled = mean_scaled.cpu().numpy()
    std_scaled = std_scaled.cpu().numpy()
    y_pred = y_scaler.inverse_transform(mean_scaled)
    y_std = std_scaled * y_scaler.scale_
    return y_pred, y_std


def fit_pinn_calibration(model, X_bias: np.ndarray, y_bias_raw: np.ndarray,
                         X_interval: np.ndarray, y_interval_raw: np.ndarray,
                         y_scaler) -> dict:
    """
    Fit point calibration, then calibrate the MC-Dropout interval scale.

    Point calibration and interval calibration use different chronological
    held-out subsets. Test observations are never used here.
    """
    print("[5] Calibrating PINN point predictions and uncertainty...")

    bias_pred_raw, _ = _mc_predict_original(model, X_bias, y_scaler)
    interval_pred_raw, interval_std_raw = _mc_predict_original(
        model, X_interval, y_scaler
    )

    calibration = {
        "alpha": ALPHA,
        "mc_samples": MC_SAMPLES,
        "targets": {},
    }
    summary_rows = []

    for i, target in enumerate(TARGETS):
        point_cal = fit_affine_calibrator(
            y_bias_raw[:, i], bias_pred_raw[:, i]
        )

        interval_pred = apply_affine_calibrator(
            interval_pred_raw[:, i], point_cal
        )
        # For y' = a*y + b, standard deviation scales by |a|.
        interval_std = (
            abs(point_cal["slope"]) * interval_std_raw[:, i]
        )
        q = conformal_mc_scale(
            y_interval_raw[:, i], interval_pred, interval_std, alpha=ALPHA
        )
        lo, hi = calibrated_interval(interval_pred, interval_std, q)

        before_bias = float(np.mean(
            bias_pred_raw[:, i] - y_bias_raw[:, i]
        ))
        bias_pred_cal = apply_affine_calibrator(
            bias_pred_raw[:, i], point_cal
        )
        after_bias = float(np.mean(bias_pred_cal - y_bias_raw[:, i]))
        interval_coverage = float(np.mean(
            (y_interval_raw[:, i] >= lo) & (y_interval_raw[:, i] <= hi)
        ))

        calibration["targets"][target] = {
            **point_cal,
            "q": float(q),
        }
        summary_rows.append({
            "target": target,
            "slope": point_cal["slope"],
            "intercept": point_cal["intercept"],
            "bias_before": before_bias,
            "bias_after": after_bias,
            "interval_q": q,
            "interval_calibration_coverage": interval_coverage,
            "nominal_coverage": 1 - ALPHA,
        })

        print(
            f"    {target:3s}: y_cal={point_cal['slope']:.4f}*y + "
            f"{point_cal['intercept']:+.4f}, q={q:.3f}, "
            f"interval-cal coverage={interval_coverage:.3f}"
        )

    joblib.dump(calibration, "models/pinn_calibration.joblib")
    pd.DataFrame(summary_rows).to_csv(
        "results/pinn_calibration_summary.csv", index=False
    )
    print("    Calibration saved → models/pinn_calibration.joblib")
    return calibration


def evaluate_pinn(model, X_te: np.ndarray, y_te_raw: np.ndarray,
                  y_scaler, calibration: dict) -> tuple:
    """Evaluate calibrated PINN predictions and calibrated MC-Dropout bounds."""
    print("[6] Evaluating calibrated PINN on test set (2014–2015)...")
    y_pred_raw, y_std_raw = _mc_predict_original(model, X_te, y_scaler)

    y_pred = np.zeros_like(y_pred_raw, dtype=float)
    y_std = np.zeros_like(y_std_raw, dtype=float)
    lo_all = np.zeros_like(y_pred_raw, dtype=float)
    hi_all = np.zeros_like(y_pred_raw, dtype=float)
    metrics = []

    for i, target in enumerate(TARGETS):
        target_cal = calibration["targets"][target]
        y_pred[:, i] = apply_affine_calibrator(
            y_pred_raw[:, i], target_cal
        )
        y_std[:, i] = abs(target_cal["slope"]) * y_std_raw[:, i]
        lo, hi = calibrated_interval(
            y_pred[:, i], y_std[:, i], target_cal["q"]
        )
        lo_all[:, i] = lo
        hi_all[:, i] = hi
        metrics.append(evaluate_predictions(
            y_te_raw[:, i],
            y_pred[:, i],
            lo,
            hi,
            target,
            nominal_coverage=1 - ALPHA,
        ))

    return y_pred, y_std, lo_all, hi_all, metrics, y_pred_raw, y_std_raw


def run_pinn(data_path: str = "data/processed/syngas_features.csv"):
    """
    Full PINN + calibrated MC-Dropout pipeline.

    Outputs include the original artifacts plus:
      models/pinn_calibration.joblib
      results/pinn_calibration_summary.csv

    The required prediction CSV columns remain unchanged; additional audit
    columns expose raw pre-calibration predictions and the test year.
    """
    _ensure_output_dirs()
    print("=" * 65)
    print("  Physics-Informed Neural Network (PINN) Soft Sensor")
    print("=" * 65)

    train_df, test_df = _load_splits(data_path)
    train_core, val_df, bias_cal_df, interval_cal_df = _split_pinn_training_block(
        train_df
    )
    print(
        f"[1] Data loaded — Fit: {len(train_core):,}  "
        f"Val: {len(val_df):,}  Bias-cal: {len(bias_cal_df):,}  "
        f"Interval-cal: {len(interval_cal_df):,}  Test: {len(test_df):,}"
    )

    (
        X_tr, X_va, X_bias, X_interval, X_te,
        y_tr, y_va, y_bias_raw, y_interval_raw, y_te_raw, y_scaler,
    ) = _preprocess_pinn(
        train_core, val_df, bias_cal_df, interval_cal_df, test_df
    )

    model, history = train_pinn(X_tr, y_tr, X_va, y_va)
    torch.save(model.state_dict(), "models/pinn_final.pt")
    print("    Final model saved → models/pinn_final.pt")

    calibration = fit_pinn_calibration(
        model, X_bias, y_bias_raw, X_interval, y_interval_raw, y_scaler
    )

    (
        y_pred, y_std, lo_all, hi_all, metrics, y_pred_raw, y_std_raw,
    ) = evaluate_pinn(
        model, X_te, y_te_raw, y_scaler, calibration
    )

    out = pd.DataFrame({
        "year": test_df["year"].values,
        "y_true_CO": y_te_raw[:, 0],
        "y_pred_raw_CO": y_pred_raw[:, 0],
        "y_pred_CO": y_pred[:, 0],
        "mc_std_raw_CO": y_std_raw[:, 0],
        "y_std_CO": y_std[:, 0],
        "pi_lo_CO": lo_all[:, 0],
        "pi_hi_CO": hi_all[:, 0],
        "y_true_NOx": y_te_raw[:, 1],
        "y_pred_raw_NOx": y_pred_raw[:, 1],
        "y_pred_NOx": y_pred[:, 1],
        "mc_std_raw_NOx": y_std_raw[:, 1],
        "y_std_NOx": y_std[:, 1],
        "pi_lo_NOx": lo_all[:, 1],
        "pi_hi_NOx": hi_all[:, 1],
    })
    out.to_csv("results/pinn_predictions.csv", index=False)
    print("  Predictions saved → results/pinn_predictions.csv")

    pd.DataFrame(history).to_csv(
        "results/pinn_training_history.csv", index=False
    )

    print("\nPINN pipeline complete.\n")
    return model, metrics, history
