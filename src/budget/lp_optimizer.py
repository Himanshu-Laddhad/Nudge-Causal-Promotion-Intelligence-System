"""
Budget-constrained promotion targeting via Linear Programming.

For uniform costs, uses the closed-form Neyman-Pearson solution (sort by CATE,
take top-K). For heterogeneous costs, falls back to the LP relaxation with
greedy rounding (0-1 knapsack).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.config import MARGIN_PER_CONVERSION, PROMO_COST


class LPBudgetOptimizer:
    """
    LP-based budget optimizer for promotion targeting.

    Parameters
    ----------
    cost_per_promo : float — expected cost of promoting one customer
    revenue_per_conversion : float — value of one incremental conversion

    Defaults come from src.config so every phase reports comparable dollars.
    The default value is *contribution margin*, not gross order value: crediting
    a campaign with revenue it has to spend on COGS overstates ROI roughly 3x.
    """

    def __init__(self, cost_per_promo: float = PROMO_COST,
                 revenue_per_conversion: float = MARGIN_PER_CONVERSION):
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

        If naive_scores is provided, the naive strategy is used only to *select*
        customers; the resulting set is then scored on the same CATE yardstick as
        the optimal set. Scoring naive selections on P(convert) instead would
        compare two different quantities and inflate the naive strategy.

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
                naive_mask = self.optimize(np.asarray(naive_scores), b, costs=costs)['target_mask']
                naive_lift = float(scores[naive_mask].sum())
                naive_cost = (float(np.asarray(costs)[naive_mask].sum())
                              if costs is not None
                              else naive_mask.sum() * self.cost_per_promo)
                naive_revenue = naive_lift * self.revenue_per_conversion
                naive_roi = (naive_revenue - naive_cost) / max(naive_cost, 1e-9)
                row['naive_lift'] = round(naive_lift, 4)
                row['lift_vs_naive'] = round(res['expected_lift'] - naive_lift, 4)
                row['naive_roi'] = round(naive_roi, 4)
                row['roi_vs_naive'] = round(res['expected_roi'] - naive_roi, 4)
            rows.append(row)
        return pd.DataFrame(rows)
