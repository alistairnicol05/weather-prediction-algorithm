from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Report figure dimensions (from template)
GOLDEN   = (5**0.5 - 1) / 2   # ≈ 0.618
W_SINGLE = 3.45                # inches
W_DOUBLE = 7.10                # inches

def _show_fig(fig, column="double"):
    """Resize figure to report dimensions and display it."""
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
    _show_fig(fig)


def plot_residuals(y_true_dict, y_pred_dict, out_dir="figures/analysis"):
    
    for h in sorted(y_true_dict):
        y_true = y_true_dict[h]
        y_pred = y_pred_dict[h]
        resid  = y_pred - y_true
        rmse = np.sqrt(np.mean(resid**2))
        mae  = np.mean(np.abs(resid))

        fig, axes = plt.subplots(1,2,figsize=(W_DOUBLE, W_DOUBLE * GOLDEN / 1.6))

        # Predicted vs Observed
        ax=axes[0]
        lo = min(y_true.min(), y_pred.min()) - 1
        hi = max(y_true.max(), y_pred.max()) + 1
        ax.scatter(y_true, y_pred, s=2, alpha=0.3)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Observed T (°C)"); ax.set_ylabel("Predicted T (°C)")
        ax.set_title("Predicted vs Observed")
        ax.legend(fontsize=8)
        
        
        # Residual histogram
        ax=axes[1]
        ax.hist(resid, bins=60)
        ax.axvline(0,            color="r", linewidth=0.8, linestyle="--", label="zero")
        ax.axvline(resid.mean(), color="k", linewidth=0.8,
                   label=f"mean = {resid.mean():.2f}")
        ax.set_xlabel("Residual (°C)"); ax.set_ylabel("Count")
        ax.set_title("Residual distribution")
        ax.legend(fontsize=8)

        fig.suptitle(f"{h}h Model Analysis [2024-12-01] - [2025-03-01]")
        fig.tight_layout()
        
        _show_fig(fig)
        
        

def plot_horizon_comparison(
    y_true_dict: dict,
    y_pred_dict: dict,
    y_baseline_dict: dict,
    start_date: str,
    end_date: str,
    horizons: list = [1, 48],
):
    
    fig, axes = plt.subplots(
        1, 2,
        figsize=(W_DOUBLE, W_DOUBLE * GOLDEN),
        sharey=True
    )

    for ax, h in zip(axes, horizons):
        y_true     = y_true_dict[h]
        y_pred     = y_pred_dict[h]
        y_baseline = y_baseline_dict[h]

        ax.plot(y_true, label="Observed",  linewidth=1.0)
        ax.plot(y_pred, label="Predicted", linewidth=1.0, alpha=0.8)
        ax.plot(y_baseline, label="Baseline",  linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"{h}h horizon")
        ax.set_xlabel("Hour index (validation period)")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("T [°C]")
    fig.suptitle(f"Forecast comparison — {horizons[0]}h vs {horizons[1]}h ahead [{start_date}] - [{end_date}]")
    fig.tight_layout()

    _show_fig(fig)
    