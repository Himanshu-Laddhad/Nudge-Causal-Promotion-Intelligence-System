"""
Uplift visualisation utilities.

All plots use a consistent dark theme and return matplotlib Figure objects
for embedding in notebooks or Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Sequence

from .metrics import qini_curve, auuc


PALETTE = {
    "naive":    "#e74c3c",
    "t_learner":"#3498db",
    "s_learner":"#2ecc71",
    "x_learner":"#f39c12",
    "dr":       "#9b59b6",
    "causal_forest": "#1abc9c",
    "random":   "#95a5a6",
}
FIGSIZE = (9, 5)


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#1a1a2e")
    ax.figure.set_facecolor("#16213e")
    ax.tick_params(colors="#ecf0f1")
    ax.xaxis.label.set_color("#ecf0f1")
    ax.yaxis.label.set_color("#ecf0f1")
    ax.title.set_color("#ecf0f1")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2c3e50")


def plot_qini_curves(
    results: list[dict],
    title: str = "Qini Curves – Model Comparison",
    figsize: tuple = FIGSIZE,
) -> plt.Figure:
    """
    Plot Qini curves for multiple models on one axes.

    Parameters
    ----------
    results : list of dicts with keys:
        'name'          : model name (str)
        'y_true'        : np.ndarray
        'uplift_score'  : np.ndarray
        'treatment'     : np.ndarray
    title : figure title.
    figsize : (width, height) in inches.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _style_ax(ax)

    for r in results:
        fracs, qini = qini_curve(
            r["y_true"], r["uplift_score"], r["treatment"], normalize=True
        )
        color = PALETTE.get(r["name"].lower().replace(" ", "_"), "#ecf0f1")
        auc_val = np.trapezoid(qini, fracs)
        ax.plot(fracs * 100, qini, label=f"{r['name']}  (Qini={auc_val:.4f})",
                color=color, linewidth=2)

    max_qini = max(
        qini_curve(r["y_true"], r["uplift_score"], r["treatment"])[-1].max()
        for r in results
    )
    ax.plot([0, 100], [0, max_qini], "--", color=PALETTE["random"],
            linewidth=1.2, label="Random targeting")

    ax.set_xlabel("Population targeted (%)")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.3,
               labelcolor="#ecf0f1", facecolor="#1a1a2e")
    fig.tight_layout()
    return fig


def plot_cate_distribution(
    cate_scores: np.ndarray,
    model_name: str = "model",
    bins: int = 60,
    figsize: tuple = FIGSIZE,
) -> plt.Figure:
    """
    Histogram of predicted CATE values with vertical lines at key quantiles.

    Distribution shape is a primary diagnostic:
    - Bimodal: model separates persuadables from sleeping dogs (good).
    - Narrow spike near 0: model is under-fitting (bad).
    - Heavy left tail: many customers actively harmed by promotions.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _style_ax(ax)

    color = PALETTE.get(model_name.lower().replace(" ", "_"), "#3498db")
    ax.hist(cate_scores, bins=bins, color=color, alpha=0.75, edgecolor="none")

    for q, ls in [(0.25, ":"), (0.5, "--"), (0.75, "-.")]:
        v = np.quantile(cate_scores, q)
        ax.axvline(v, color="#ecf0f1", linestyle=ls, linewidth=1,
                   label=f"Q{int(q*100)} = {v:.4f}")

    ax.axvline(0, color="#e74c3c", linewidth=1.5, label="CATE = 0")

    frac_neg = (cate_scores <= 0).mean()
    ax.set_title(
        f"CATE Distribution – {model_name}\n"
        f"Deadweight loss: {frac_neg:.1%} of population",
        pad=10,
    )
    ax.set_xlabel("Predicted CATE")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, framealpha=0.3, labelcolor="#ecf0f1",
              facecolor="#1a1a2e")
    fig.tight_layout()
    return fig


def plot_segment_waterfall(
    summary_df: pd.DataFrame,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Horizontal bar chart showing treatment-effect segments across models.

    Parameters
    ----------
    summary_df : DataFrame with columns:
        model, persuadable_pct, sleeping_dog_pct, neutral_pct

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _style_ax(ax)

    models = summary_df["model"].tolist()
    x = np.arange(len(models))
    w = 0.25

    ax.bar(x - w, summary_df["persuadable_pct"], w, label="Persuadable (CATE>0)",
           color="#2ecc71", alpha=0.85)
    ax.bar(x,     summary_df["neutral_pct"],     w, label="Neutral (CATE≈0)",
           color="#f39c12", alpha=0.85)
    ax.bar(x + w, summary_df["sleeping_dog_pct"], w, label="Sleeping dog (CATE<0)",
           color="#e74c3c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylabel("% of population")
    ax.set_title("Customer Segment Distribution by Model")
    ax.legend(fontsize=8, framealpha=0.3, labelcolor="#ecf0f1",
              facecolor="#1a1a2e")
    fig.tight_layout()
    return fig
