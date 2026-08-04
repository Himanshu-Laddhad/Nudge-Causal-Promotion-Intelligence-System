"""Cached data loaders for the Nudge dashboard. All data is pre-computed."""
import sys
import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.model_registry import best_cate_model, display_name  # noqa: E402

OUTPUTS = ROOT / 'outputs'
MASTER_CSV = OUTPUTS / 'master_comparison_table.csv'
FEATURE_COLS = [
    'recency', 'history', 'mens', 'womens', 'newbie',
    'recency_bucket', 'spend_tier',
    'zip_urban', 'zip_suburban', 'zip_rural',
    'ch_phone', 'ch_web', 'ch_multi',
    'newbie_x_spend_tier', 'recency_x_spend_tier',
]


@st.cache_data
def load_cate_scores() -> pd.DataFrame:
    """
    Load best available CATE scores, merging across phase outputs.

    Falls back gracefully: phase4 → phase2 → phase1. The chosen model is resolved
    from the measured Qini ranking, so the dashboard never disagrees with the
    comparison table about which model is best.
    """
    for fname in [
        'phase4_final_cate.parquet',
        'phase2_cate_predictions.parquet',
    ]:
        p = OUTPUTS / fname
        if p.exists():
            df = pd.read_parquet(p)
            try:
                _, col = best_cate_model(MASTER_CSV, list(df.columns))
            except ValueError:
                continue
            df['_best_cate'] = df[col]
            df.attrs['best_cate_col'] = col
            return df

    p = OUTPUTS / 'phase1_naive_scores.parquet'
    if p.exists():
        df = pd.read_parquet(p)
        if 'p_convert' in df.columns and 'naive_score' not in df.columns:
            df['naive_score'] = df['p_convert']
        df['_best_cate'] = df.get('naive_score', df.get('p_convert'))
        df.attrs['best_cate_col'] = 'naive_score'
        return df

    return pd.DataFrame()


@st.cache_data
def load_features() -> pd.DataFrame:
    p = OUTPUTS / 'hillstrom_features.parquet'
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_master_comparison() -> pd.DataFrame:
    for fname in ['master_comparison_table.csv', 'phase2_comparison_table.csv']:
        p = OUTPUTS / fname
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data
def load_budget_results() -> pd.DataFrame:
    p = OUTPUTS / 'budget_optimizer_results.csv'
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_robustness_results() -> pd.DataFrame:
    p = OUTPUTS / 'robustness_experiment_results.csv'
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_decile_uplift() -> pd.DataFrame:
    p = OUTPUTS / 'phase2_decile_uplift.csv'
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_breakeven() -> pd.DataFrame:
    p = OUTPUTS / 'phase5a_breakeven.csv'
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def get_best_cate_col(df: pd.DataFrame) -> str | None:
    """Name of the score column the dashboard is treating as the CATE model."""
    if 'best_cate_col' in df.attrs:
        return df.attrs['best_cate_col']
    try:
        return best_cate_model(MASTER_CSV, list(df.columns))[1]
    except ValueError:
        return None


def held_out(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows safe for outcome-based metrics.

    The scored parquets cover the full customer base so the budget optimiser can
    allocate across everyone, but Qini curves and observed conversion rates must
    only use the held-out split — training rows carry in-sample scores and would
    overstate performance.
    """
    if 'in_test' in df.columns:
        return df[df['in_test'].astype(bool)]
    return df


def best_model_label(df: pd.DataFrame | None = None) -> str:
    """Human-readable name of the promoted model, e.g. 'T-Learner'."""
    col = get_best_cate_col(df) if df is not None else None
    if col:
        return display_name(col)
    try:
        return best_cate_model(MASTER_CSV, None)[0]
    except ValueError:
        return 'CATE model'


def phases_available() -> dict[str, bool]:
    """Return a dict of which phase output files exist."""
    return {
        'phase1': (OUTPUTS / 'phase1_naive_scores.parquet').exists(),
        'phase2': (OUTPUTS / 'phase2_cate_predictions.parquet').exists(),
        'phase4': (OUTPUTS / 'phase4_final_cate.parquet').exists(),
        'features': (OUTPUTS / 'hillstrom_features.parquet').exists(),
        'master': (OUTPUTS / 'master_comparison_table.csv').exists(),
        'budget': (OUTPUTS / 'budget_optimizer_results.csv').exists(),
        'robustness': (OUTPUTS / 'robustness_experiment_results.csv').exists(),
        'decile': (OUTPUTS / 'phase2_decile_uplift.csv').exists(),
        'breakeven': (OUTPUTS / 'phase5a_breakeven.csv').exists(),
    }
