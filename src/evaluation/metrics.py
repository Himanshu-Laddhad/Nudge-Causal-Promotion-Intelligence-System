"""
Uplift evaluation metrics.

Causal framing
--------------
Standard classification metrics (AUC-ROC, accuracy) are *wrong* for uplift
models because we never observe the counterfactual outcome.  Instead, we
rank customers by predicted CATE (uplift score) and measure how much of the
total observed incremental lift is captured in the top-K percentile.

Key metrics
-----------
Qini coefficient
    Area between the Qini curve and the random-targeting diagonal.
    Proposed by Radcliffe (2007).  Normalised Qini = Qini / perfect-Qini.

AUUC (Area Under the Uplift Curve)
    Variant where y-axis is incremental conversions (not Radcliffe's
    adjusted count).  Standard in causalml / sklift.

Deadweight-loss fraction
    Fraction of treated customers with predicted CATE ≤ 0.
    These are "sleeping dogs" – promotions wasted on people who either
    would have bought anyway or are negatively persuaded by promotions.

ATT / ATE
    Average Treatment Effect on the Treated / overall population.
    Estimated by a simple difference-in-means on the held-out RCT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils import check_consistent_length


# ---------------------------------------------------------------------------
# Qini curve & coefficient
# ---------------------------------------------------------------------------


def qini_curve(
    y_true: np.ndarray,
    uplift_score: np.ndarray,
    treatment: np.ndarray,
    *,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Qini curve.

    Ranks customers by descending uplift_score, then at each percentile
    computes the *incremental* conversion rate vs. the random targeting line.

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

    # Cumulative treated & control conversions at each rank
    cum_treat_conv = np.cumsum(y_sorted * t_sorted)
    cum_ctrl_conv = np.cumsum(y_sorted * (1 - t_sorted))
    cum_treat_n = np.cumsum(t_sorted)
    cum_ctrl_n = np.cumsum(1 - t_sorted)

    # Avoid division by zero
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

    # Prepend (0, 0) for proper AUC calculation
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

    A positive value means the model beats random targeting.
    Higher is better; theoretical maximum equals the perfect-model Qini.
    """
    fracs, qini = qini_curve(y_true, uplift_score, treatment, normalize=True)
    random_line = np.linspace(0, qini[-1], len(qini))
    return float(np.trapezoid(qini - random_line, fracs))


# ---------------------------------------------------------------------------
# AUUC
# ---------------------------------------------------------------------------


def auuc(
    y_true: np.ndarray,
    uplift_score: np.ndarray,
    treatment: np.ndarray,
) -> float:
    """
    Area Under the Uplift Curve (AUUC).

    Uplift curve y-axis = incremental responders (treatment – control
    conversion rate) in the top-k fraction of scored population.

    A random model scores 0.5 * total_incremental_lift.
    Models above this baseline add value.
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

    uplift_vals = y_s[t_s == 1].cumsum()  # simplification; full version below
    # Full incremental rate at each step
    inc = (
        np.cumsum(y_s * t_s) / safe_t
        - np.cumsum(y_s * (1 - t_s)) / safe_c
    )

    fracs = np.arange(1, n + 1) / n
    fracs = np.concatenate([[0.0], fracs])
    inc = np.concatenate([[0.0], inc])
    return float(np.trapezoid(inc, fracs))


# ---------------------------------------------------------------------------
# Deadweight loss
# ---------------------------------------------------------------------------


def deadweight_loss_fraction(uplift_scores: np.ndarray, threshold: float = 0.0) -> float:
    """
    Fraction of the *treatment* group with predicted CATE ≤ threshold.

    Causal interpretation
    ---------------------
    Customers with CATE ≤ 0 are "sleeping dogs": the promotion either has
    no incremental effect (CATE ≈ 0, deadweight cost) or actively reduces
    conversion (CATE < 0, backfire effect).  Targeting them wastes budget
    and can erode brand trust.

    Parameters
    ----------
    uplift_scores : predicted individual treatment effects.
    threshold     : CATE value below which a customer is considered wasted.
                    Default 0 = any non-positive CATE.

    Returns
    -------
    float in [0, 1].  Lower is better.
    """
    scores = np.asarray(uplift_scores)
    return float((scores <= threshold).mean())


# ---------------------------------------------------------------------------
# Simple RCT estimators
# ---------------------------------------------------------------------------


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

    Under random assignment, ATT = ATE, but we compute it explicitly
    for correctness when treatment arm sizes differ.
    """
    y, t = np.asarray(y), np.asarray(t)
    # Under RCT: ATT ≈ ATE; use treated mean minus imputed control mean
    ctrl_rate = y[t == 0].mean()
    return float(y[t == 1].mean() - ctrl_rate)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


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
