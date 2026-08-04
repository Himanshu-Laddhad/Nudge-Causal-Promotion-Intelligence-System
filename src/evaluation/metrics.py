"""Uplift evaluation metrics: Qini, AUUC, deadweight loss, ATE/ATT."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils import check_consistent_length

from src.config import PROMO_COST


def qini_curve(
    y_true: np.ndarray,
    uplift_score: np.ndarray,
    treatment: np.ndarray,
    *,
    normalize: bool = False,
    normalize_y: bool = False,
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
    normalize    : if True, divide x-axis by total population size (for plotting
                   on a 0-1 "population targeted" axis).
    normalize_y  : if True, divide the y-axis (cumulative incremental
                   conversions) by total population size too, turning it into
                   an incremental *rate*. Required for ``qini_auc`` to be a
                   scale-invariant coefficient comparable across dataset
                   sizes — without it, the curve's y-axis (and hence its AUC)
                   scales linearly with n, which silently produces numbers
                   like 1000+ on a 2.8M-row test set for a model that scores
                   ~0.03 on a 13K-row one, even though targeting quality is
                   unchanged. Defaults to False to preserve existing plotting
                   behaviour (y-axis in raw incremental-conversion counts).

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
    if normalize_y:
        qini = qini / n

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
    Qini coefficient = area under the (population-normalized) Qini curve
    minus the random baseline.

    Higher is better; positive value means the model beats random targeting.
    Scale-invariant across dataset sizes (see ``qini_curve``'s
    ``normalize_y`` docstring) — safe to compare across test sets of very
    different sizes.
    """
    fracs, qini = qini_curve(
        y_true, uplift_score, treatment, normalize=True, normalize_y=True
    )
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
    Fraction of a model's OWN predicted scores that are ≤ threshold.

    NOTE: this is a score-distribution diagnostic, not a comparable
    cross-model deadweight-loss metric. It only means "wasted spend" when
    ``uplift_scores`` is itself a CATE estimate centred at 0 (T/S/X-Learner,
    DR-Learner). Applied to a raw propensity/conversion
    probability (e.g. the naive baseline's P(convert | X, T=1), which is
    bounded in [0, 1] and essentially never ≤ 0), it trivially returns ~0%
    regardless of how many sleeping dogs the model actually targets — that
    is a property of the score's support, not of targeting quality. Use
    ``topk_deadweight_loss`` for an apples-to-apples comparison across
    models with different score scales.

    Parameters
    ----------
    uplift_scores : predicted individual treatment effects (or any score).
    threshold     : score value below which a customer is considered wasted.

    Returns
    -------
    float in [0, 1]. Lower is better. NOT comparable across models with
    different score supports.
    """
    scores = np.asarray(uplift_scores)
    return float((scores <= threshold).mean())


def topk_deadweight_loss(
    y_true: np.ndarray,
    uplift_score: np.ndarray,
    treatment: np.ndarray,
    *,
    top_frac: float = 0.2,
    cost_per_treatment: float = PROMO_COST,
) -> dict:
    """
    Realized-outcome deadweight loss in a model's top-``top_frac`` target list.

    Model-agnostic and comparable across naive propensity scores, CATE
    estimates, or any other ranking score, because it never looks at the
    score's own sign or scale — only at what the RCT's control arm actually
    did. Mirrors the Phase-1 "Query C" sleeping-dog audit
    (notebooks/phase1_naive_baseline.ipynb): rank everyone by
    ``uplift_score``, take the top ``top_frac`` of the population, then use
    the CONTROL-arm conversion rate within that group as an unbiased
    RCT-based estimate of P(Y(0)=1 | targeted) — i.e. how many of the
    people this model would target actually convert with no promotion at
    all. That fraction, applied to the full targeted list, is the estimated
    number of customers who receive a promotion for zero incremental
    return (sleeping dogs), and is the number that should be compared
    across models — not the sign of each model's own predicted score.

    Parameters
    ----------
    y_true             : binary outcome (0/1).
    uplift_score       : any ranking score, higher = more targeted first.
    treatment           : binary treatment indicator (1 = treated).
    top_frac            : fraction of the population targeted (default top 20%).
    cost_per_treatment  : $ cost of promoting one customer.

    Returns
    -------
    dict with n_targeted, pct_sleeping_dogs (of the targeted list),
    n_sleeping_dogs, wasted_budget_usd, and control_conv_rate_in_topk /
    treated_conv_rate_in_topk for context.
    """
    check_consistent_length(y_true, uplift_score, treatment)
    y = np.asarray(y_true)
    u = np.asarray(uplift_score)
    t = np.asarray(treatment)

    n = len(y)
    k = max(int(round(n * top_frac)), 1)
    order = np.argsort(-u)
    top_idx = order[:k]

    y_top, t_top = y[top_idx], t[top_idx]
    ctrl_mask = t_top == 0
    treat_mask = t_top == 1

    ctrl_conv_rate = float(y_top[ctrl_mask].mean()) if ctrl_mask.any() else float("nan")
    treat_conv_rate = float(y_top[treat_mask].mean()) if treat_mask.any() else float("nan")

    pct_sleeping_dogs = ctrl_conv_rate if not np.isnan(ctrl_conv_rate) else 0.0
    n_sleeping_dogs = pct_sleeping_dogs * k

    return {
        "n_targeted": int(k),
        "pct_sleeping_dogs": round(pct_sleeping_dogs * 100, 2),
        "n_sleeping_dogs_est": round(n_sleeping_dogs, 1),
        "wasted_budget_usd_est": round(n_sleeping_dogs * cost_per_treatment, 2),
        "control_conv_rate_in_topk": round(ctrl_conv_rate, 4) if not np.isnan(ctrl_conv_rate) else None,
        "treated_conv_rate_in_topk": round(treat_conv_rate, 4) if not np.isnan(treat_conv_rate) else None,
    }


def bootstrap_qini_ci(
    y: np.ndarray,
    cate: np.ndarray,
    t: np.ndarray,
    qini_fn,
    n_boot: int = 500,
    seed: int = 42,
    outlier_mad_k: float = 15.0,
) -> dict:
    """
    Bootstrap confidence interval for a Qini AUC statistic, robust to
    outlier resamples.

    Ratio-based Qini implementations (e.g. ``sklift.metrics.qini_auc_score``,
    which divides by the "perfect Qini" AUC) can spike to extreme values on
    individual bootstrap resamples where that denominator happens to be
    small — this is most visible on rare-outcome datasets, where a resample
    can by chance draw very few positives in one arm. A single such draw is
    enough to drag the mean and
    std far from the true sampling distribution while leaving the
    percentile-based bounds nearly untouched (percentiles only move if
    outliers exceed ~2.5% of resamples). Rather than silently trusting a
    corrupted mean, this filters resamples more than ``outlier_mad_k``
    median-absolute-deviations from the median before computing summary
    stats, and reports how many were dropped.

    qini_fn(y_sample, cate_sample, t_sample) -> float
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            score = qini_fn(y[idx], cate[idx], t[idx])
        except Exception:
            continue
        if np.isfinite(score):
            boot_scores.append(score)
    boot_scores = np.array(boot_scores)
    n_successful = len(boot_scores)

    median = float(np.median(boot_scores))
    mad = float(np.median(np.abs(boot_scores - median))) or 1e-12
    keep = np.abs(boot_scores - median) <= outlier_mad_k * mad * 1.4826
    clean = boot_scores[keep]
    n_outliers = int((~keep).sum())

    return {
        "qini_auc_mean": float(clean.mean()),
        "qini_auc_median": median,
        "qini_auc_ci_lower": float(np.percentile(boot_scores, 2.5)),
        "qini_auc_ci_upper": float(np.percentile(boot_scores, 97.5)),
        "qini_auc_std": float(clean.std()),
        "n_boot_successful": n_successful,
        "n_outliers_dropped": n_outliers,
    }


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
    *,
    top_frac: float = 0.2,
    cost_per_treatment: float = PROMO_COST,
) -> pd.DataFrame:
    """
    Compute all uplift metrics and return a one-row DataFrame.

    Suitable for accumulating results across multiple models in a loop.
    ``pct_sleeping_dogs_topk`` / ``wasted_budget_usd_topk`` (from
    ``topk_deadweight_loss``) are the metrics comparable across models;
    ``pct_score_nonpositive`` (from ``deadweight_loss_fraction``) is kept
    only as a score-distribution diagnostic — see that function's docstring
    for why it is NOT cross-model comparable.
    """
    topk = topk_deadweight_loss(
        y_true, uplift_score, treatment,
        top_frac=top_frac, cost_per_treatment=cost_per_treatment,
    )
    return pd.DataFrame([{
        "model": model_name,
        "qini_auc": qini_auc(y_true, uplift_score, treatment),
        "auuc": auuc(y_true, uplift_score, treatment),
        "pct_sleeping_dogs_topk": topk["pct_sleeping_dogs"],
        "wasted_budget_usd_topk": topk["wasted_budget_usd_est"],
        "n_targeted_topk": topk["n_targeted"],
        "pct_score_nonpositive": deadweight_loss_fraction(uplift_score) * 100,
        "ate_rct": ate_from_rct(y_true, treatment),
        "att_rct": att_from_rct(y_true, treatment),
        "mean_pred_cate": float(np.mean(uplift_score)),
        "std_pred_cate": float(np.std(uplift_score)),
    }])
