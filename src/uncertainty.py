"""
Uncertainty quantification utilities for both model types.

XGBoost / MAPIE:
  Uses SplitConformalRegressor from the mapie library.
  Splits training data into a fit set and a calibration set. The calibration
  set is used to compute residual quantiles that determine PI width.
  Coverage guarantee is distribution-free (no Gaussian assumption on residuals).

  If test-set coverage falls significantly below the target alpha, the
  calibration set may be too small. Increase test_size in wrap_with_mapie.

PINN / MC Dropout:
  Dropout is kept active at inference. The std across N forward passes serves
  as the uncertainty estimate. 95% PI = mean ± 1.96 * std.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from mapie.regression import SplitConformalRegressor as MapieRegressor

ALPHA        = 0.05          # 1 - 0.05 = 95% prediction intervals
RANDOM_STATE = 42


def wrap_with_mapie(estimator, X_train: np.ndarray,
                    y_train: np.ndarray, target_name: str) -> MapieRegressor:
    """
    Wrap a pre-configured (but unfitted) XGBRegressor with MAPIE conformal PI.

    Steps:
      1. Split training data 80/20 into fit and calibration partitions.
      2. Fit the estimator on the fit partition.
      3. Conformalize (calibrate residual quantiles) on the calibration partition.

    The estimator is passed in unfitted; MAPIE handles the fit internally.
    """
    print(f"    Wrapping '{target_name}' with MAPIE (alpha={ALPHA})...")

    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    estimator.fit(X_fit, y_fit)

    mapie = MapieRegressor(
        estimator=estimator,
        prefit=True,
        confidence_level=1 - ALPHA,
    )
    mapie.conformalize(X_calib, y_calib)
    return mapie


def coverage_score(y_true: np.ndarray, lo: np.ndarray,
                   hi: np.ndarray) -> float:
    """Fraction of true values falling within [lo, hi]."""
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def average_width(lo: np.ndarray, hi: np.ndarray) -> float:
    """Mean prediction interval width."""
    return float(np.mean(hi - lo))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         lo: np.ndarray, hi: np.ndarray,
                         target_name: str) -> dict:
    """
    Compute and print RMSE, MAE, R², PI coverage, and average PI width.

    Returns a dict suitable for building comparison DataFrames.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse     = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae      = float(mean_absolute_error(y_true, y_pred))
    r2       = float(r2_score(y_true, y_pred))
    coverage = coverage_score(y_true, lo, hi)
    width    = average_width(lo, hi)

    print(f"\n  [{target_name}]")
    print(f"    RMSE         : {rmse:.4f}")
    print(f"    MAE          : {mae:.4f}")
    print(f"    R²           : {r2:.4f}")
    print(f"    PI Coverage  : {coverage:.3f}  (target ≥ {1 - ALPHA:.2f})")
    print(f"    Avg PI width : {width:.4f}")

    return dict(target=target_name, rmse=rmse, mae=mae,
                r2=r2, coverage=coverage, avg_width=width)
