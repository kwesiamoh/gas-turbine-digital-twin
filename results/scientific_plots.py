"""
Publication-quality diagnostic figures for the PINN soft sensor.

Reads results/pinn_predictions.csv and generates three figures:
  fig1_parity.pdf    — density parity plot (hexbin) for CO and NOx
  fig2_calibration.pdf — uncertainty reliability (calibration) curve for CO
  fig3_residuals.pdf — residual analysis for CO

Usage:
  python results/scientific_plots.py
  (run after PINN training)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.family":   "serif",
    "font.serif":    ["Times New Roman"],
    "font.size":     10,
    "axes.labelsize": 11,
    "figure.dpi":    200,
    "axes.grid":     True,
    "grid.alpha":    0.3,
})

PREDICTIONS_PATH = Path(__file__).parent / "pinn_predictions.csv"


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{PREDICTIONS_PATH} not found. Run the PINN training pipeline first."
        )
    return pd.read_csv(PREDICTIONS_PATH)


def fig1_parity(df: pd.DataFrame):
    """
    Hexbin parity plots for CO and NOx.

    Hexbin is preferred over scatter for large N because overplotting
    obscures density — hexbin encodes point density as colour.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    targets   = [("CO", "mg/m³"), ("NOx", "mg/m³")]

    for ax, (t, unit) in zip(axes, targets):
        y_true = df[f"y_true_{t}"]
        y_pred = df[f"y_pred_{t}"]

        ax.hexbin(y_true, y_pred, gridsize=35, cmap="Greys", mincnt=1)
        ax.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                "r--", lw=1.5, label=r"Ideal ($y=\hat{y}$)")

        r2 = stats.pearsonr(y_true, y_pred)[0] ** 2
        ax.annotate(f"$R^2 = {r2:.3f}$", xy=(0.05, 0.95), xycoords="axes fraction")

        ax.set_title(f"Prediction Fidelity: {t}")
        ax.set_xlabel(f"Experimental {t} [{unit}]")
        ax.set_ylabel(f"Soft-Sensor {t} [{unit}]")
        ax.legend(loc="lower right")

    plt.tight_layout()
    out = Path(__file__).parent / "fig1_parity.pdf"
    plt.savefig(out)
    print(f"  Saved → {out}")
    plt.close()


def fig2_calibration(df: pd.DataFrame):
    """
    Uncertainty reliability diagram for CO.

    A well-calibrated model's empirical coverage at confidence level p should
    closely match p. Curves above the diagonal are over-confident; curves
    below are under-confident.

    Requires y_std_CO column (MC Dropout std). If absent, this figure is skipped.
    """
    if "y_std_CO" not in df.columns:
        print("  fig2: y_std_CO not found — skipping calibration plot")
        return

    expected_p, observed_p = np.linspace(0.1, 0.99, 20), []
    for p in expected_p:
        z     = stats.norm.ppf((1 + p) / 2)
        lo    = df["y_pred_CO"] - z * df["y_std_CO"]
        hi    = df["y_pred_CO"] + z * df["y_std_CO"]
        cov   = np.mean((df["y_true_CO"] >= lo) & (df["y_true_CO"] <= hi))
        observed_p.append(cov)

    plt.figure(figsize=(6, 5))
    plt.plot(expected_p, observed_p, "o-", color="#00d4aa", label="PINN MC-Dropout")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.title("Uncertainty Reliability Diagram")
    plt.xlabel("Target Confidence Level")
    plt.ylabel("Empirical Data Coverage")
    plt.legend()

    out = Path(__file__).parent / "fig2_calibration.pdf"
    plt.savefig(out)
    print(f"  Saved → {out}")
    plt.close()


def fig3_residuals(df: pd.DataFrame):
    """
    Residual vs predicted value plot for CO.

    Residuals should be randomly scattered around zero with no visible
    trend. A funnel shape indicates heteroscedasticity; a curve suggests
    a systematic non-linearity the model is not capturing.
    """
    residuals = df["y_true_CO"] - df["y_pred_CO"]

    plt.figure(figsize=(8, 5))
    plt.scatter(df["y_pred_CO"], residuals, alpha=0.2, s=2, color="gray")
    plt.axhline(0, color="red", lw=1, ls="--")
    plt.title("Residual Analysis: CO Error Distribution")
    plt.xlabel("Predicted CO [mg/m³]")
    plt.ylabel("Residual Error [mg/m³]")

    out = Path(__file__).parent / "fig3_residuals.pdf"
    plt.savefig(out)
    print(f"  Saved → {out}")
    plt.close()


def generate_all():
    print("Generating scientific figures from results/pinn_predictions.csv ...")
    df = load_predictions()
    fig1_parity(df)
    fig2_calibration(df)
    fig3_residuals(df)
    print("Done.")


if __name__ == "__main__":
    generate_all()
