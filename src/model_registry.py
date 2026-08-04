"""
Single source of truth for which model the pipeline promotes.

Every stage that needs "the best model" resolves it from
outputs/master_comparison_table.csv rather than hardcoding a name. Hardcoding is
how the repo previously ended up claiming three different winners in three
different places after a re-run shifted the ranking.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Display name in master_comparison_table.csv -> score column in the CATE parquets.
MODEL_COLUMNS: dict[str, str] = {
    'Naive XGB': 'naive_score',
    'T-Learner': 'cate_t_learner',
    'S-Learner': 'cate_s_learner',
    'X-Learner': 'cate_x_learner',
    'DR-Learner': 'cate_dr_clean',
}

# The naive classifier is the baseline we argue against, not a targeting policy,
# so it is never eligible to be selected as the CATE model.
BASELINE_MODELS: tuple[str, ...] = ('Naive XGB',)

# Used only when the comparison table is missing entirely.
FALLBACK_ORDER: tuple[str, ...] = (
    'cate_t_learner', 'cate_s_learner', 'cate_dr_clean',
    'cate_x_learner', 'naive_score', 'p_convert',
)


def best_cate_model(
    master_csv: str | Path,
    available_columns: list[str] | None = None,
) -> tuple[str, str]:
    """
    Resolve the highest-Qini causal model to (display_name, score_column).

    `available_columns` restricts the choice to score columns actually present in
    the scored dataset, so a model that ranks well but was not exported is skipped
    rather than causing a KeyError downstream.
    """
    cols = available_columns

    master_csv = Path(master_csv)
    if master_csv.exists():
        table = pd.read_csv(master_csv)
        if {'model', 'qini_auc'}.issubset(table.columns):
            ranked = table.sort_values('qini_auc', ascending=False)
            for _, row in ranked.iterrows():
                name = str(row['model'])
                if name in BASELINE_MODELS:
                    continue
                col = MODEL_COLUMNS.get(name)
                if col and (cols is None or col in cols):
                    return name, col

    for col in FALLBACK_ORDER:
        if cols is None or col in cols:
            name = next((k for k, v in MODEL_COLUMNS.items() if v == col), col)
            return name, col

    raise ValueError(f"No usable CATE column found among {cols}")


def display_name(score_column: str) -> str:
    """Human-readable model name for a score column."""
    return next(
        (name for name, col in MODEL_COLUMNS.items() if col == score_column),
        score_column.replace('_', ' ').title(),
    )
