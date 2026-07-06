"""Uplift evaluation metrics: Qini, AUUC, deadweight loss, ATE/ATT."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils import check_consistent_length


def qini_curve(
    y_true: np.ndarray,
    uplift_score: np.ndarray,
    treatment: np.ndarray,
    *,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Qini curve (Radcliffe, 2007).

    Ranks customers by descending uplift_score and computes cumulative
    incremental conversions vs. the random targeting line at each percentile.

    Parameters
    ----------
    y_true       : binary outcome (0/1).
    uplift_score : predicted CATE / uplift score (higher = more persuadable).
    treatment    : binary treatment indicator (1 = treated).
    normalize    : if True, divide x-axis by total population size.

    Returns
    -------
    (fractions, qini_values) – arrays suitable for plt.plot().
    """
    check_consistent_length(y_true, uplift_score, treatment)
    y_true = np.asarray(y_true)
    uplift_score = np.asarray(uplift_score)
    treatment = np.asarray(treatment)

    order = np.argsort(-uplift_score)
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    n = len(y_true)
    n_treat = treatment.sum()
    n_ctrl = n - n_treat

    cum_treat_conv = np.cumsum(y_sorted * t_sorted)
    cum_ctrl_conv = np.cumsum(y_sorted * (1 - t_sorted))
    cum_treat_n = np.cumsum(t_sorted)
    cum_ctrl_n = np.cumsum(1 - t_sorted)

    safe_treat_n = np.where(cum_treat_n == 0, 1, cum_treat_n)
    safe_ctrl_n = np.where(cum_ctrl_n == 0, 1, cum_ctrl_n)

    # Radcliffe Qini: incremental conversions adjusted for exposure ratio
    qini = (
        cum_treat_conv
        - cum_ctrl_conv * (cum_treat_n / np.where(cum_ctrl_n == 0, 1, cum_ctrl_n))
    )

    fractions = np.arange(1, n + 1)
    if normalize:
        fractions = fractions / n

    fractions = np.concatenate([[0], fractions])
    qini = np.concatenate([[0.0], qini])
    return fractions, qini


def qini_auc(
    y_true: np.ndarray,
    uplift_score: np.ndarray,
    treatment: np.ndarray,
) -> float:
    """
    Qini coefficient = area under the Qini curve minus the random baseline.

    Higher is better; positive value means the model beats random targeting.
    """
    fracs, qini = qini_curve(y_true, uplift_score, treatment, normalize=True)
    random_line = np.linspace(0, qini[-1], len(qini))
    return float(np.trapezoid(qini - random_line, fracs))


def auuc(
    y_true: np.ndarray,
    uplift_score: np.ndarray,
    treatment: np.ndarray,
) -> float:
    """
    Area Under the Uplift Curve (AUUC).

    Y-axis = incremental responders (treatment − control conversion rate)
    in the top-k fraction of the scored population.
    """
    check_consistent_length(y_true, uplift_score, treatment)
    y = np.asarray(y_true)
    u = np.asarray(uplift_score)
    t = np.asarray(treatment)

    order = np.argsort(-u)
    y_s, t_s = y[order], t[order]
    n = len(y)

    cum_t = np.cumsum(t_s)
    cum_c = np.cumsum(1 - t_s)
    safe_t = np.where(cum_t == 0, 1, cum_t)
    safe_c = np.where(cum_c == 0, 1, cum_c)

    uplift_vals = y_s[t_s == 1].cumsum()
    inc = (
        np.cumsum(y_s * t_s) / safe_t
        - np.cumsum(y_s * (1 - t_s)) / safe_c
    )

    fracs = np.arange(1, n + 1) / n
    fracs = np.concatenate([[0.0], fracs])
    inc = np.concatenate([[0.0], inc])
    return float(np.trapezoid(inc, fracs))


def deadweight_loss_fraction(uplift_scores: np.ndarray, threshold: float = 0.0) -> float:
    """
    Fraction of customers with predicted CATE ≤ threshold.

    Customers at or below threshold are sleeping dogs: the promotion has
    no incremental effect or actively reduces conversion.

    Parameters
    ----------
    uplift_scores : predicted individual treatment effects.
    threshold     : CATE value below which a customer is considered wasted.

    Returns
    -------
    float in [0, 1]. Lower is better.
    """
    scores = np.asarray(uplift_scores)
    return float((scores <= threshold).mean())


def ate_from_rct(y: np.ndarray, t: np.ndarray) -> float:
    """
    Difference-in-means ATE estimator (unbiased under random assignment).

    Returns E[Y(1)] - E[Y(0)].
    """
    y, t = np.asarray(y), np.asarray(t)
    return float(y[t == 1].mean() - y[t == 0].mean())


def att_from_rct(y: np.ndarray, t: np.ndarray) -> float:
    """
    ATT = E[Y(1) - Y(0) | T=1].

    Under random assignment ATT = ATE; computed explicitly for correctness
    when treatment arm sizes differ.
    """
    y, t = np.asarray(y), np.asarray(t)
    # Under RCT: ATT ≈ ATE; use treated mean minus imputed control mean
    ctrl_rate = y[t == 0].mean()
    return float(y[t == 1].mean() - ctrl_rate)


def evaluation_summary(
    y_true: np.ndarray,
    uplift_score: np.ndarray,
    treatment: np.ndarray,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Compute all uplift metrics and return a one-row DataFrame.

    Suitable for accumulating results across multiple models in a loop.
    """
    return pd.DataFrame([{
        "model": model_name,
        "qini_auc": qini_auc(y_true, uplift_score, treatment),
        "auuc": auuc(y_true, uplift_score, treatment),
        "deadweight_loss_pct": deadweight_loss_fraction(uplift_score) * 100,
        "ate_rct": ate_from_rct(y_true, treatment),
        "att_rct": att_from_rct(y_true, treatment),
        "mean_pred_cate": float(np.mean(uplift_score)),
        "std_pred_cate": float(np.std(uplift_score)),
    }])
