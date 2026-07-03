"""
Budget-constrained promotion targeting optimizer.

Causal framing
--------------
Given individual CATE estimates τ̂(xᵢ), the optimal targeting rule under a
budget constraint B (number of promotions or $ spend) is:

    Target the top-K customers ranked by τ̂(xᵢ), where K = B / cost_per_promo.

This is the Neyman-Pearson uplift rule.  It maximises expected incremental
conversions subject to a budget constraint *without* requiring a specific
decision threshold.

Why this beats a flat discount policy
--------------------------------------
A flat "send coupon to everyone" policy has a zero marginal cost of
identifying sleeping dogs — it simply mails them anyway.  Under a budget
constraint, every coupon sent to a sleeping dog is a coupon not sent to a
persuadable customer.  The optimizer quantifies this opportunity cost.

Value metrics
-------------
incremental_conversions(K) = Σᵢ∈top_K  τ̂(xᵢ)
saved_promotions(K)         = N - K  (vs. flat policy)
roi_lift(K)                 = incremental_conversions(K) / (K × cost_per_promo)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BudgetOptimizer:
    """
    Rank-and-threshold promotion targeting under a budget constraint.

    Parameters
    ----------
    cost_per_promo : cost (£/$) of sending one promotion.
    revenue_per_conversion : expected revenue per incremental conversion.
    """

    def __init__(
        self,
        cost_per_promo: float = 1.0,
        revenue_per_conversion: float = 10.0,
    ) -> None:
        self.cost_per_promo = cost_per_promo
        self.revenue_per_conversion = revenue_per_conversion

    def optimize(
        self,
        cate_scores: np.ndarray,
        budget: float,
    ) -> dict:
        """
        Select the optimal targeting set given a total budget.

        Returns
        -------
        dict with keys:
            target_mask      : bool array, True = customer should be targeted.
            n_targeted        : number of customers targeted.
            n_saved           : customers NOT targeted (vs. flat policy).
            expected_lift     : Σ τ̂(xᵢ) over targeted set.
            expected_roi      : revenue_per_conversion * lift / budget.
            deadweight_excluded : customers excluded with τ̂ ≤ 0.
        """
        scores = np.asarray(cate_scores)
        n = len(scores)
        k = min(int(budget / self.cost_per_promo), n)

        order = np.argsort(-scores)
        target_mask = np.zeros(n, dtype=bool)
        target_mask[order[:k]] = True

        lift = scores[target_mask].sum()
        roi = (self.revenue_per_conversion * lift) / max(k * self.cost_per_promo, 1e-9)

        return {
            "target_mask": target_mask,
            "n_targeted": int(k),
            "n_saved": int(n - k),
            "expected_lift": float(lift),
            "expected_roi": float(roi),
            "deadweight_excluded": int((scores[~target_mask] <= 0).sum()),
        }

    def budget_curve(
        self,
        cate_scores: np.ndarray,
        n_points: int = 100,
    ) -> pd.DataFrame:
        """
        Compute expected lift and ROI across a range of budget levels.

        Useful for the "elbow plot" that shows diminishing returns as
        targeting expands into the sleeping-dog segment.

        Returns
        -------
        DataFrame with columns: budget, n_targeted, expected_lift, expected_roi.
        """
        scores = np.asarray(cate_scores)
        n = len(scores)
        max_budget = n * self.cost_per_promo

        budgets = np.linspace(0, max_budget, n_points + 1)[1:]
        rows = []
        for b in budgets:
            res = self.optimize(scores, b)
            rows.append({
                "budget": b,
                "budget_pct": b / max_budget * 100,
                "n_targeted": res["n_targeted"],
                "n_targeted_pct": res["n_targeted"] / n * 100,
                "expected_lift": res["expected_lift"],
                "expected_roi": res["expected_roi"],
            })
        return pd.DataFrame(rows)

    def targeting_comparison(
        self,
        cate_scores: np.ndarray,
        y_true: np.ndarray,
        treatment: np.ndarray,
        budget_fractions: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Compare CATE-targeted vs. random vs. flat policy at several budget levels.

        Parameters
        ----------
        budget_fractions : list of floats in (0, 1] representing fraction of
                           total population to target.  Defaults to [0.1, 0.2,
                           0.3, 0.5, 1.0].

        Returns
        -------
        DataFrame with policy × budget_fraction comparison.
        """
        if budget_fractions is None:
            budget_fractions = [0.1, 0.2, 0.3, 0.5, 1.0]

        scores = np.asarray(cate_scores)
        y = np.asarray(y_true)
        t = np.asarray(treatment)
        n = len(scores)

        # Observed conversion rate in treatment arm (proxy for actual uplift)
        ctrl_rate = y[t == 0].mean()

        rows = []
        for frac in budget_fractions:
            k = max(1, int(frac * n))
            budget = k * self.cost_per_promo

            # CATE-guided targeting
            order_cate = np.argsort(-scores)[:k]
            cate_lift = scores[order_cate].sum()

            # Random targeting (expected value = flat ATE × k)
            ate = y[t == 1].mean() - ctrl_rate
            random_lift = ate * k

            rows.append({
                "budget_fraction": frac,
                "budget": budget,
                "n_targeted": k,
                "cate_lift": cate_lift,
                "random_lift": random_lift,
                "lift_vs_random": cate_lift - random_lift,
            })
        return pd.DataFrame(rows)
