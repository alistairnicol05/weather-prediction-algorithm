"""
analysis.py
===========
Plotting utilities for the weather forecasting report.

Usage
-----
1. Add `feature_cols` to design_matrix's return:
       return X, y, feature_cols

2. Collect results in your main loop:
       thetas_dict[horizon]        = theta
       feature_cols_dict[horizon]  = feature_cols
       y_true_dict[horizon]        = y_val_raw
       y_pred_dict[horizon]        = y_pred

3. After the loop, call whichever functions you want:
       from analysis import *
       plot_feature_importance(thetas_dict, feature_cols_dict)
       plot_metrics_vs_horizon(results)
       plot_residuals(y_true_dict, y_pred_dict)
       plot_error_breakdown(y_true_dict, y_pred_dict, df_val.index)
       plot_forecast_report(y_true_dict[h], y_pred_dict[h], h, start_date, end_date)
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Report figure dimensions (from template)
GOLDEN   = (5**0.5 - 1) / 2   # ≈ 0.618
W_SINGLE = 3.45                # inches
W_DOUBLE = 7.10                # inches

def _save(fig, column="double"):
    """Resize figure to report dimensions and save as PDF."""
    w = W_SINGLE if column == "single" else W_DOUBLE
    old_w, old_h = fig.get_size_inches()
    fig.set_size_inches(w, w * old_h / old_w)
    plt.show()
    plt.close(fig)


def plot_metrics_vs_horizon(results, out_dir="figures/analysis"):
    """
    Two-panel plot: RMSE and MAE against forecast horizon.
    `results` is the list of dicts you already build in the main loop.
    """
    df       = pd.DataFrame(results)
    horizons = df["Horizon (h)"].values
    rmses    = df["RMSE"].values
    maes     = df["MAE"].values

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, W_DOUBLE * GOLDEN / 2))

    for ax, label, vals in zip(axes, ["RMSE", "MAE"], [rmses, maes]):
        ax.plot(horizons, vals, marker="o")
        for h, v in zip(horizons, vals):
            ax.annotate(f"{v:.2f}", (h, v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_ylabel(f"{label} (°C)")
        ax.set_title(f"{label} vs horizon")
        ax.set_xticks(horizons)

    fig.tight_layout()
    _save(fig, f"metrics_vs_horizon")


def plot_residuals(y_true_dict, y_pred_dict, out_dir="figures/analysis"):
    """
    4-panel residual figure per horizon:
      top-left  : predicted vs observed (points on y=x line = perfect)
      top-right : residuals over time   (flat around 0 = no drift)
      bot-left  : residual histogram    (bell around 0 = unbiased)
      bot-right : residuals vs predicted (flat = no heteroscedasticity)
    """
    for h in sorted(y_true_dict):
        y_true = y_true_dict[h]
        y_pred = y_pred_dict[h]
        resid  = y_pred - y_true

        fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, W_DOUBLE * GOLDEN))

        # Predicted vs Observed
        ax = axes[0, 0]
        lo = min(y_true.min(), y_pred.min()) - 1
        hi = max(y_true.max(), y_pred.max()) + 1
        ax.scatter(y_true, y_pred, s=2, alpha=0.3)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Observed T (°C)"); ax.set_ylabel("Predicted T (°C)")
        ax.set_title("Predicted vs Observed"); ax.legend(fontsize=8)

        # Residuals over time
        ax = axes[0, 1]
        ax.plot(resid, linewidth=0.4)
        ax.axhline(0, color="r", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Validation sample index")
        ax.set_ylabel("Residual (°C)"); ax.set_title("Residuals over time")

        # Residual histogram
        ax = axes[1, 0]
        ax.hist(resid, bins=60)
        ax.axvline(0,            color="r", linewidth=0.8, linestyle="--", label="zero")
        ax.axvline(resid.mean(), color="k", linewidth=0.8,
                   label=f"mean = {resid.mean():.2f}")
        ax.set_xlabel("Residual (°C)"); ax.set_ylabel("Count")
        ax.set_title("Residual distribution"); ax.legend(fontsize=8)

        # Residuals vs Predicted
        ax = axes[1, 1]
        ax.scatter(y_pred, resid, s=2, alpha=0.3)
        ax.axhline(0, color="r", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Predicted T (°C)"); ax.set_ylabel("Residual (°C)")
        ax.set_title("Residuals vs Predicted")

        rmse = np.sqrt(np.mean(resid**2))
        mae  = np.mean(np.abs(resid))
        fig.suptitle(f"Residual diagnostics — {h}h  (RMSE={rmse:.2f} °C, MAE={mae:.2f} °C)")
        fig.tight_layout()
        _save(fig, f"residuals_{h}h")

