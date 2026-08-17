"""
Uncertainty quantification utilities for both model types.

XGBoost / MAPIE:
  Uses SplitConformalRegressor from the mapie library. Training and
  conformalization data are supplied separately so the conformalization set is
  never used for preprocessing fitting, hyperparameter selection, or model fit.

PINN / MC Dropout:
  Only Dropout layers are kept active at inference. The standard deviation
  across N stochastic passes is an epistemic uncertainty estimate. The project
  reports mean ± 1.96 * std as an approximate uncertainty interval and evaluates
  its empirical coverage; it is not a conformal 95% coverage guarantee.
"""

import numpy as np
from mapie.regression import SplitConformalRegressor as MapieRegressor

ALPHA        = 0.05          # 1 - 0.05 = 95% prediction intervals


def wrap_with_mapie(estimator, X_fit: np.ndarray,
                    y_fit: np.ndarray, X_calib: np.ndarray,
                    y_calib: np.ndarray, target_name: str) -> MapieRegressor:
    """
    Fit an XGBRegressor on the model-fit partition and conformalize it on a
    separate, untouched chronological conformalization partition.

    The caller is responsible for creating the temporal split before any
    preprocessing is fitted. This prevents conformalization labels from
    influencing preprocessing, hyperparameter selection, or estimator fitting.
    """
    print(f"    Wrapping '{target_name}' with MAPIE (alpha={ALPHA})...")

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
                         target_name: str,
                         nominal_coverage: float | None = 1 - ALPHA) -> dict:
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
    if nominal_coverage is None:
        print(f"    PI Coverage  : {coverage:.3f}  (empirical; no nominal guarantee)")
    else:
        print(f"    PI Coverage  : {coverage:.3f}  (nominal {nominal_coverage:.2f})")
    print(f"    Avg PI width : {width:.4f}")

    return dict(target=target_name, rmse=rmse, mae=mae,
                r2=r2, coverage=coverage, avg_width=width)
