"""
Uncertainty and calibration utilities for both model families.

XGBoost / MAPIE:
  SplitConformalRegressor is fitted on a model-fit partition and conformalized
  on a separate chronological calibration partition.

PINN / MC Dropout:
  MC Dropout estimates epistemic variability. A separate held-out block is used
  for point-prediction bias calibration and another untouched held-out block is
  used to calibrate the MC-Dropout scale with normalized conformal residuals.
"""

import numpy as np
from mapie.regression import SplitConformalRegressor as MapieRegressor
from sklearn.linear_model import LinearRegression

ALPHA = 0.05
EPS_STD = 1e-6


def wrap_with_mapie(estimator, X_fit: np.ndarray,
                    y_fit: np.ndarray, X_calib: np.ndarray,
                    y_calib: np.ndarray, target_name: str) -> MapieRegressor:
    """Fit the estimator, then conformalize on an untouched calibration set."""
    print(f"    Wrapping '{target_name}' with MAPIE (alpha={ALPHA})...")

    estimator.fit(X_fit, y_fit)
    mapie = MapieRegressor(
        estimator=estimator,
        prefit=True,
        confidence_level=1 - ALPHA,
    )
    mapie.conformalize(X_calib, y_calib)
    return mapie


def fit_affine_calibrator(y_true: np.ndarray,
                          y_pred: np.ndarray) -> dict:
    """
    Fit y_calibrated = slope * y_pred + intercept on held-out observations.

    A non-negative slope preserves the ranking direction learned by the model
    while allowing both level bias and scale bias to be corrected.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        raise ValueError("Calibration arrays must have the same length >= 2")

    reg = LinearRegression(positive=True)
    reg.fit(y_pred.reshape(-1, 1), y_true)
    return {
        "slope": float(reg.coef_[0]),
        "intercept": float(reg.intercept_),
    }


def apply_affine_calibrator(y_pred: np.ndarray,
                            calibration: dict) -> np.ndarray:
    """Apply a saved affine point-prediction calibration."""
    return (
        float(calibration["slope"]) * np.asarray(y_pred, dtype=float)
        + float(calibration["intercept"])
    )


def conformal_mc_scale(y_true: np.ndarray, y_pred: np.ndarray,
                       y_std: np.ndarray, alpha: float = ALPHA) -> float:
    """
    Calibrate MC-Dropout scale using normalized absolute residuals.

    score_i = |y_i - mean_i| / max(std_i, EPS_STD)

    The returned finite-sample conformal quantile q is later used as:
      [mean - q * std, mean + q * std]

    The calibration set used here must not have been used to train the network,
    choose the checkpoint, or fit the affine point calibrator.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_std = np.asarray(y_std, dtype=float).reshape(-1)

    if not (len(y_true) == len(y_pred) == len(y_std)) or len(y_true) < 2:
        raise ValueError("Interval calibration arrays must have equal length >= 2")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")

    scale = np.maximum(np.abs(y_std), EPS_STD)
    scores = np.abs(y_true - y_pred) / scale
    n = len(scores)
    quantile_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, quantile_level, method="higher"))


def calibrated_interval(y_pred: np.ndarray, y_std: np.ndarray,
                        q: float) -> tuple[np.ndarray, np.ndarray]:
    """Construct symmetric calibration-adjusted bounds around the prediction."""
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    half_width = float(q) * np.maximum(np.abs(y_std), EPS_STD)
    return y_pred - half_width, y_pred + half_width


def coverage_score(y_true: np.ndarray, lo: np.ndarray,
                   hi: np.ndarray) -> float:
    """Fraction of true values falling within [lo, hi]."""
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def average_width(lo: np.ndarray, hi: np.ndarray) -> float:
    """Mean prediction interval width."""
    return float(np.mean(hi - lo))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         lo: np.ndarray, hi: np.ndarray,
                         target_name: str,
                         nominal_coverage: float | None = 1 - ALPHA) -> dict:
    """Compute and print RMSE, MAE, R², interval coverage, width, and bias."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    coverage = coverage_score(y_true, lo, hi)
    width = average_width(lo, hi)
    bias = float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))

    print(f"\n  [{target_name}]")
    print(f"    RMSE         : {rmse:.4f}")
    print(f"    MAE          : {mae:.4f}")
    print(f"    R²           : {r2:.4f}")
    print(f"    Bias         : {bias:+.4f}")
    if nominal_coverage is None:
        print(f"    PI Coverage  : {coverage:.3f}  (empirical; no nominal guarantee)")
    else:
        print(f"    PI Coverage  : {coverage:.3f}  (nominal {nominal_coverage:.2f})")
    print(f"    Avg PI width : {width:.4f}")

    return dict(
        target=target_name,
        rmse=rmse,
        mae=mae,
        r2=r2,
        bias=bias,
        coverage=coverage,
        avg_width=width,
    )
