"""Cached data loaders for the Nudge dashboard. All data is pre-computed."""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

OUTPUTS = Path(__file__).parent.parent.parent / 'outputs'

# Priority order: best model first
CATE_PRIORITY = ['cate_dr_clean', 'cate_cf', 'cate_x_learner', 'cate_t_learner', 'naive_score', 'p_convert']
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

    Falls back gracefully: phase4 → phase3 → phase2 → phase1.
    Normalises the best available score column to _best_cate.
    """
    for fname in [
        'phase4_final_cate.parquet',
        'phase3_cate_predictions.parquet',
        'phase2_cate_predictions.parquet',
    ]:
        p = OUTPUTS / fname
        if p.exists():
            df = pd.read_parquet(p)
            available = [c for c in CATE_PRIORITY if c in df.columns]
            if available:
                df['_best_cate'] = df[available[0]]
                return df

    p = OUTPUTS / 'phase1_naive_scores.parquet'
    if p.exists():
        df = pd.read_parquet(p)
        if 'p_convert' in df.columns and 'naive_score' not in df.columns:
            df['naive_score'] = df['p_convert']
        df['_best_cate'] = df.get('naive_score', df.get('p_convert'))
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
def load_phase1_scores() -> pd.DataFrame:
    """Load Phase 1 naive scores (p_convert proxy)."""
    p = OUTPUTS / 'phase1_naive_scores.parquet'
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if 'p_convert' in df.columns and 'naive_score' not in df.columns:
        df['naive_score'] = df['p_convert']
    return df


def get_best_cate_col(df: pd.DataFrame) -> str | None:
    """Return the name of the best available CATE column present in df."""
    return next((c for c in CATE_PRIORITY if c in df.columns), None)


def phases_available() -> dict[str, bool]:
    """Return a dict of which phase output files exist."""
    return {
        'phase1': (OUTPUTS / 'phase1_naive_scores.parquet').exists(),
        'phase2': (OUTPUTS / 'phase2_cate_predictions.parquet').exists(),
        'phase3': (OUTPUTS / 'phase3_cate_predictions.parquet').exists(),
        'phase4': (OUTPUTS / 'phase4_final_cate.parquet').exists(),
        'features': (OUTPUTS / 'hillstrom_features.parquet').exists(),
        'master': (OUTPUTS / 'master_comparison_table.csv').exists(),
        'budget': (OUTPUTS / 'budget_optimizer_results.csv').exists(),
        'robustness': (OUTPUTS / 'robustness_experiment_results.csv').exists(),
        'decile': (OUTPUTS / 'phase2_decile_uplift.csv').exists(),
    }
