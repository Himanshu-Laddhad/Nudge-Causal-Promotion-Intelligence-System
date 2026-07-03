"""
Budget-constrained promotion targeting via Linear Programming.

Causal framing
--------------
Given individual CATE estimates τ̂(xᵢ), the promotion targeting problem is:

    max   Σᵢ τ̂(xᵢ) · xᵢ
    s.t.  Σᵢ c · xᵢ ≤ B        (budget constraint)
          xᵢ ∈ {0, 1}            (binary targeting decision)

where c = cost per promotion, B = total budget.

This is a 0-1 knapsack problem. When τ̂(xᵢ) > 0 for all items and costs
are uniform (cᵢ = c for all i), the LP relaxation is tight and the optimal
integer solution is simply: sort by τ̂(xᵢ) descending, take top K = ⌊B/c⌋.
This is the Neyman-Pearson uplift rule.

When costs are heterogeneous (e.g., different discount depths per segment),
we use scipy.optimize.linprog on the LP relaxation and round, or PuLP for
exact integer programming.

ROI calculation
---------------
Expected incremental revenue = Σᵢ∈S τ̂(xᵢ) × revenue_per_conversion
Expected cost                = |S| × cost_per_promo
ROI = (incremental_revenue - cost) / cost
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linprog


class LPBudgetOptimizer:
    """
    LP-based budget optimizer for promotion targeting.

    For uniform costs (the common case), uses the closed-form Neyman-Pearson
    solution (sort by CATE, take top-K). For heterogeneous costs, falls back
    to the LP relaxation with randomised rounding.

    Parameters
    ----------
    cost_per_promo : float — cost of sending one promotion (default $10)
    revenue_per_conversion : float — expected revenue per incremental conversion (default $50)
    """

    def __init__(self, cost_per_promo: float = 10.0, revenue_per_conversion: float = 50.0):
        self.cost_per_promo = cost_per_promo
        self.revenue_per_conversion = revenue_per_conversion

    def optimize(self, cate_scores: np.ndarray, budget: float,
                 costs: np.ndarray | None = None) -> dict:
        """
        Select optimal targeting set for a given budget.

        For uniform costs, uses sort-based Neyman-Pearson (exact optimal).
        For heterogeneous costs, uses LP relaxation + greedy rounding.

        Returns dict with keys:
            target_mask, n_targeted, expected_lift, expected_revenue,
            expected_cost, expected_roi, pct_positive_cate_targeted
        """
        scores = np.asarray(cate_scores, dtype=float)
        n = len(scores)
        if costs is None:
            costs = np.full(n, self.cost_per_promo)
        costs = np.asarray(costs, dtype=float)

        # Neyman-Pearson closed form for uniform costs
        if np.allclose(costs, costs[0]):
            k = min(int(budget / costs[0]), n)
            order = np.argsort(-scores)
            mask = np.zeros(n, dtype=bool)
            mask[order[:k]] = True
        else:
            # LP relaxation: minimise -τ·x  s.t. c·x ≤ B, 0 ≤ x ≤ 1
            res = linprog(
                -scores,
                A_ub=costs[np.newaxis, :],
                b_ub=[budget],
                bounds=[(0, 1)] * n,
                method='highs',
            )
            x_relax = np.clip(res.x, 0, 1)
            # Greedy rounding: sort by fractional value × score / cost
            priority = x_relax * scores / np.where(costs == 0, 1e-9, costs)
            order = np.argsort(-priority)
            mask = np.zeros(n, dtype=bool)
            cum_cost = 0.0
            for i in order:
                if cum_cost + costs[i] <= budget:
                    mask[i] = True
                    cum_cost += costs[i]

        lift = float(scores[mask].sum())
        revenue = lift * self.revenue_per_conversion
        actual_cost = float(costs[mask].sum())
        roi = (revenue - actual_cost) / max(actual_cost, 1e-9)

        return {
            'target_mask': mask,
            'n_targeted': int(mask.sum()),
            'expected_lift': lift,
            'expected_revenue': revenue,
            'expected_cost': actual_cost,
            'expected_roi': roi,
            'pct_positive_cate_targeted': float((scores[mask] > 0).mean() * 100) if mask.any() else 0.0,
        }

    def budget_sweep(self, cate_scores: np.ndarray, budgets: list[float],
                     naive_scores: np.ndarray | None = None,
                     costs: np.ndarray | None = None) -> pd.DataFrame:
        """
        Run optimize() across multiple budget levels and return a comparison DataFrame.

        If naive_scores provided, also computes naive-targeting metrics for comparison.

        Returns DataFrame with one row per budget with columns:
            budget, n_targeted, pct_population, expected_lift,
            expected_revenue, expected_cost, expected_roi,
            naive_lift (if naive_scores given), lift_vs_naive,
            pct_positive_cate_targeted
        """
        scores = np.asarray(cate_scores, dtype=float)
        n = len(scores)
        rows = []
        for b in budgets:
            res = self.optimize(scores, b, costs=costs)
            row = {
                'budget': b,
                'n_targeted': res['n_targeted'],
                'pct_population': round(res['n_targeted'] / n * 100, 2),
                'expected_lift': round(res['expected_lift'], 4),
                'expected_revenue': round(res['expected_revenue'], 2),
                'expected_cost': round(res['expected_cost'], 2),
                'expected_roi': round(res['expected_roi'], 4),
                'pct_positive_cate_targeted': round(res['pct_positive_cate_targeted'], 2),
            }
            if naive_scores is not None:
                naive_res = self.optimize(np.asarray(naive_scores), b, costs=costs)
                row['naive_lift'] = round(naive_res['expected_lift'], 4)
                row['lift_vs_naive'] = round(res['expected_lift'] - naive_res['expected_lift'], 4)
                row['naive_roi'] = round(naive_res['expected_roi'], 4)
                row['roi_vs_naive'] = round(res['expected_roi'] - naive_res['expected_roi'], 4)
            rows.append(row)
        return pd.DataFrame(rows)
