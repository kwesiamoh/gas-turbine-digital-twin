"""
Model comparison: XGBoost+MAPIE vs PINN+MC-Dropout.

Reads prediction CSVs from results/ and produces a side-by-side comparison
table and diagnostic plot.

Metrics: RMSE, MAE, R², PI Coverage, average PI width.

Usage:
  python compare_models.py
  (run after training both models)

Outputs:
  results/model_comparison.csv
  results/model_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.uncertainty import coverage_score, average_width

TARGETS = ["CO", "NOx"]

CLR = {
    "bg":      "#0f1117",
    "surface": "#1e1e2e",
    "accent":  "#00d4aa",
    "amber":   "#f7b731",
    "red":     "#ff6b6b",
    "purple":  "#a78bfa",
    "text":    "#e0e0e0",
}


def load_and_score(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a predictions CSV and compute metrics for each target.
    Expected columns: y_true_{T}, y_pred_{T}, pi_lo_{T}, pi_hi_{T}
    """
    df   = pd.read_csv(path)
    rows = []
    for t in TARGETS:
        yt = df[f"y_true_{t}"].values
        yp = df[f"y_pred_{t}"].values
        lo = df[f"pi_lo_{t}"].values
        hi = df[f"pi_hi_{t}"].values
        rows.append({
            "target":   t,
            "RMSE":     float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE":      float(mean_absolute_error(yt, yp)),
            "R²":       float(r2_score(yt, yp)),
            "Coverage": coverage_score(yt, lo, hi),
            "PI Width": average_width(lo, hi),
        })
    return pd.DataFrame(rows), df


def compare():
    print("=" * 65)
    print("  Model Comparison: XGBoost+MAPIE  vs.  PINN+MC-Dropout")
    print("=" * 65)

    try:
        xgb_scores,  xgb_df  = load_and_score("results/xgboost_predictions.csv")
        pinn_scores, pinn_df = load_and_score("results/pinn_predictions.csv")
    except FileNotFoundError as e:
        print(f"\n  {e}")
        print("  Run training pipelines first (python -m src.training).\n")
        return

    xgb_scores.insert(0,  "model", "XGBoost + MAPIE")
    pinn_scores.insert(0, "model", "PINN + MC Dropout")
    combined = pd.concat([xgb_scores, pinn_scores], ignore_index=True)

    print("\n── Comparison Table ─────────────────────────────────────────────")
    print(combined.to_string(index=False, float_format="{:.4f}".format))
    combined.to_csv("results/model_comparison.csv", index=False)
    print("\n  Saved → results/model_comparison.csv")

    _plot_comparison(combined, xgb_df, pinn_df)


def _plot_comparison(combined: pd.DataFrame,
                     xgb_df:  pd.DataFrame,
                     pinn_df: pd.DataFrame):
    """
    Six-panel diagnostic figure: metric bar charts, residual distributions,
    and PI width scatter.
    """
    plt.rcParams.update({
        "text.color":       CLR["text"],
        "axes.labelcolor":  CLR["text"],
        "xtick.color":      CLR["text"],
        "ytick.color":      CLR["text"],
    })

    fig = plt.figure(figsize=(18, 10), facecolor=CLR["bg"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)

    model_colours = {
        "XGBoost + MAPIE":    CLR["accent"],
        "PINN + MC Dropout":  CLR["purple"],
    }

    metrics_to_plot = ["RMSE", "R²", "Coverage", "PI Width"]
    positions       = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (r, c), metric in zip(positions, metrics_to_plot):
        ax = fig.add_subplot(gs[r, c])
        x  = np.arange(len(TARGETS))
        w  = 0.35

        for i, (model, grp) in enumerate(combined.groupby("model")):
            vals  = [grp[grp["target"] == t][metric].values[0] for t in TARGETS]
            colour = list(model_colours.values())[i]
            ax.bar(x + i * w, vals, width=w, color=colour,
                   alpha=0.85, edgecolor=CLR["bg"], lw=0.4, label=model)

        if metric == "Coverage":
            # Reference line for the 95% target — useful to see which model misses
            ax.axhline(0.95, color=CLR["red"], lw=1.5, ls="--", label="Target 0.95")

        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(TARGETS)
        ax.set_title(metric, color=CLR["text"], fontsize=11)
        ax.set_facecolor(CLR["surface"])
        ax.grid(True, color="#111120", lw=0.4, axis="y")
        ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor=CLR["text"])

    # Panel 5: CO residual distributions
    ax5 = fig.add_subplot(gs[0, 2])
    resid_xgb  = xgb_df["y_true_CO"].values  - xgb_df["y_pred_CO"].values
    resid_pinn = pinn_df["y_true_CO"].values  - pinn_df["y_pred_CO"].values
    ax5.hist(resid_xgb,  bins=60, color=CLR["accent"], alpha=0.6,
             edgecolor=CLR["bg"], lw=0.3, label="XGBoost")
    ax5.hist(resid_pinn, bins=60, color=CLR["purple"], alpha=0.5,
             edgecolor=CLR["bg"], lw=0.3, label="PINN")
    ax5.axvline(0, color=CLR["red"], lw=1.5, ls="--")
    ax5.set_xlabel("Residual (true − pred) [mg/m³]")
    ax5.set_ylabel("Count")
    ax5.set_title("CO Residual Distributions", color=CLR["text"], fontsize=10)
    ax5.set_facecolor(CLR["surface"])
    ax5.grid(True, color="#111120", lw=0.4)
    ax5.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=CLR["text"])

    # Panel 6: PI width vs prediction value (CO only)
    # Wider intervals in a specific prediction range suggest higher aleatoric
    # uncertainty there — useful for calibration analysis.
    ax6 = fig.add_subplot(gs[1, 2])
    w_xgb  = xgb_df["pi_hi_CO"].values  - xgb_df["pi_lo_CO"].values
    w_pinn = pinn_df["pi_hi_CO"].values  - pinn_df["pi_lo_CO"].values
    ax6.scatter(xgb_df["y_pred_CO"].values,  w_xgb,  s=1.5, alpha=0.15,
                color=CLR["accent"], label="XGBoost PI width")
    ax6.scatter(pinn_df["y_pred_CO"].values, w_pinn, s=1.5, alpha=0.15,
                color=CLR["purple"], label="PINN PI width")
    ax6.set_xlabel("Predicted CO [mg/m³]")
    ax6.set_ylabel("PI width (95%)")
    ax6.set_title("Interval Width vs. Prediction (CO)", color=CLR["text"], fontsize=10)
    ax6.set_facecolor(CLR["surface"])
    ax6.grid(True, color="#111120", lw=0.4)
    ax6.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=CLR["text"])

    fig.suptitle(
        "Model Comparison: XGBoost+MAPIE  vs.  PINN+MC-Dropout  |  UCI Gas Turbine",
        fontsize=12, color=CLR["accent"], y=0.98, fontweight="bold",
    )

    plt.savefig("results/model_comparison.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Comparison plot saved → results/model_comparison.png")
    plt.close()


if __name__ == "__main__":
    compare()
