"""Generate the tabular XGBoost/MAPIE and PINN/MC-Dropout comparison.

This command intentionally does not create figures. Publication figures are a
separate, explicit workflow provided by ``results/scientific_plots.py``.

Usage:
    python compare_models.py

Output:
    results/model_comparison.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.uncertainty import average_width, coverage_score


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGETS = ("CO", "NOx")
RESULTS = Path(__file__).resolve().parent / "results"


def load_and_score(path: Path) -> pd.DataFrame:
    """Load saved predictions and calculate the comparison criteria."""
    df = pd.read_csv(path)
    rows = []
    for target in TARGETS:
        y_true = df[f"y_true_{target}"].to_numpy()
        y_pred = df[f"y_pred_{target}"].to_numpy()
        lo = df[f"pi_lo_{target}"].to_numpy()
        hi = df[f"pi_hi_{target}"].to_numpy()
        rows.append({
            "target": target,
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R²": float(r2_score(y_true, y_pred)),
            "Coverage": coverage_score(y_true, lo, hi),
            "PI Width": average_width(lo, hi),
        })
    return pd.DataFrame(rows)


def compare() -> pd.DataFrame | None:
    """Create and save the reproducible numerical model comparison."""
    print("=" * 65)
    print("  Model Comparison: XGBoost+MAPIE vs. PINN+MC-Dropout")
    print("=" * 65)

    try:
        xgb = load_and_score(RESULTS / "xgboost_predictions.csv")
        pinn = load_and_score(RESULTS / "pinn_predictions.csv")
    except FileNotFoundError as exc:
        print(f"\n  {exc}")
        print("  Run both training pipelines first.\n")
        return None

    xgb.insert(0, "model", "XGBoost + MAPIE")
    pinn.insert(0, "model", "PINN + MC Dropout")
    combined = pd.concat([xgb, pinn], ignore_index=True)

    print("\n── Comparison Table ─────────────────────────────────────────────")
    print(combined.to_string(index=False, float_format="{:.4f}".format))
    output = RESULTS / "model_comparison.csv"
    combined.to_csv(output, index=False)
    print(f"\n  Saved → {output.relative_to(Path(__file__).resolve().parent)}")
    print("  No figures generated. Run: python results/scientific_plots.py")
    return combined


if __name__ == "__main__":
    compare()
