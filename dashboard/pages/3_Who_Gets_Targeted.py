"""Page 3 — Who Gets Targeted? Customer segment analysis."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dashboard.utils.data_loader import (
    load_cate_scores,
    load_features,
    get_best_cate_col,
    best_model_label,
    FEATURE_COLS,
)
from dashboard.utils.charts import (
    apply_dark_theme,
    cate_histogram,
    segment_bar,
    TEXT,
)


st.title("👥 Who Gets Targeted?")
st.markdown("### Customer segment analysis — who responds to promotions?")

df       = load_cate_scores()
feat_df  = load_features()

if df.empty:
    st.info("Run Phase 1 notebook (`phase1_naive_baseline.ipynb`) to generate customer score data.")
    st.stop()

# Normalise p_convert → naive_score if needed
if 'p_convert' in df.columns and 'naive_score' not in df.columns:
    df['naive_score'] = df['p_convert']

best_cate = get_best_cate_col(df)

# Merge features if available and customer_id present in both
if not feat_df.empty and 'customer_id' in df.columns and 'customer_id' in feat_df.columns:
    merge_cols = ['customer_id'] + [c for c in FEATURE_COLS if c in feat_df.columns]
    df = df.merge(feat_df[merge_cols], on='customer_id', how='left')

# ── Section A: CATE Distribution by Segment ──────────────────────────────────
st.markdown("---")
st.markdown("## CATE Distribution by Customer Segment")

# Segment selector
segment_options = []
segment_col_map = {}

if 'recency_bucket' in df.columns:
    segment_options.append("Recency Bucket")
    segment_col_map["Recency Bucket"] = 'recency_bucket'
if 'spend_tier' in df.columns:
    segment_options.append("Spend Tier")
    segment_col_map["Spend Tier"] = 'spend_tier'
# Channel flags
for ch_col, ch_label in [('ch_phone', 'Phone'), ('ch_web', 'Web'), ('ch_multi', 'Multi')]:
    if ch_col in df.columns:
        segment_options.append(f"Channel ({ch_label})")
        segment_col_map[f"Channel ({ch_label})"] = ch_col
# Zip type
for z_col, z_label in [('zip_urban', 'Urban'), ('zip_suburban', 'Suburban'), ('zip_rural', 'Rural')]:
    if z_col in df.columns:
        segment_options.append(f"Zip ({z_label})")
        segment_col_map[f"Zip ({z_label})"] = z_col

# Fallback: use raw column from phase1 data
if not segment_options:
    if 'recency' in df.columns:
        df['recency_bucket'] = pd.qcut(df['recency'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
        segment_options.append("Recency Bucket")
        segment_col_map["Recency Bucket"] = 'recency_bucket'
    if 'history' in df.columns:
        df['spend_tier'] = pd.qcut(df['history'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
        segment_options.append("Spend Tier")
        segment_col_map["Spend Tier"] = 'spend_tier'

score_col = '_best_cate' if '_best_cate' in df.columns else best_cate

if segment_options and score_col:
    seg_choice = st.selectbox("Segment by:", segment_options)
    seg_col    = segment_col_map[seg_choice]

    seg_label_map = {
        'recency_bucket': {0: 'Active (0)', 1: 'Warm (1)', 2: 'Cooling (2)', 3: 'Lapsed (3)'},
        'spend_tier':     {0: 'Low (0)', 1: 'Mid-Low (1)', 2: 'Mid-High (2)', 3: 'High (3)'},
    }

    grouped = df.groupby(seg_col)[score_col].mean().reset_index()
    grouped.columns = [seg_col, 'mean_cate']

    if seg_col in seg_label_map:
        grouped[seg_col] = grouped[seg_col].map(seg_label_map[seg_col]).fillna(grouped[seg_col].astype(str))
    else:
        grouped[seg_col] = grouped[seg_col].astype(str)

    colors_seg = ['#9b59b6' if v >= 0 else '#e74c3c' for v in grouped['mean_cate']]
    model_label = best_model_label(df)

    fig_seg = segment_bar(
        labels=grouped[seg_col].tolist(),
        values=grouped['mean_cate'].tolist(),
        colors=colors_seg,
        title=f'Mean {model_label} by {seg_choice}',
        x_title=seg_choice,
        y_title='Mean CATE (Incremental Lift)',
        height=400,
    )
    st.plotly_chart(fig_seg, width='stretch')

    # Second chart: always show spend tier if available
    if 'spend_tier' in df.columns and seg_col != 'spend_tier':
        st.markdown("#### Mean CATE by Spend Tier")
        sp_grouped = df.groupby('spend_tier')[score_col].mean().reset_index()
        sp_grouped.columns = ['spend_tier', 'mean_cate']
        sp_grouped['spend_tier'] = sp_grouped['spend_tier'].map(
            seg_label_map.get('spend_tier', {})
        ).fillna(sp_grouped['spend_tier'].astype(str))
        sp_colors = ['#9b59b6' if v >= 0 else '#e74c3c' for v in sp_grouped['mean_cate']]
        fig_sp = segment_bar(
            labels=sp_grouped['spend_tier'].tolist(),
            values=sp_grouped['mean_cate'].tolist(),
            colors=sp_colors,
            title='Mean CATE by Spend Tier',
            x_title='Spend Tier',
            y_title='Mean CATE',
            height=360,
        )
        st.plotly_chart(fig_sp, width='stretch')
else:
    st.info(
        "Segment features (recency_bucket, spend_tier) not found in current data. "
        "Run `phase1_naive_baseline.ipynb` or feature engineering notebooks."
    )

# ── Section B: CATE Distribution Histogram ───────────────────────────────────
st.markdown("---")
st.markdown("## CATE Score Distribution")

if score_col and score_col in df.columns:
    scores = df[score_col].dropna().values
    model_label = best_model_label(df)

    fig_hist = cate_histogram(scores, model_label, color='#9b59b6')

    # Overlay percentile lines
    p10, p50, p90 = np.percentile(scores, [10, 50, 90])
    for pval, label, color in [
        (p10, 'P10', '#e74c3c'),
        (p50, 'Median', '#f39c12'),
        (p90, 'P90', '#2ecc71'),
    ]:
        fig_hist.add_vline(x=pval, line_color=color, line_dash='dot', line_width=1.2,
                           annotation_text=label, annotation_font_color=color)

    st.plotly_chart(fig_hist, width='stretch')

    # Stats table
    stats = {
        'Statistic': ['Mean', 'Std Dev', 'Median (P50)', 'P10', 'P90', '% Positive', '% Negative'],
        'Value': [
            f"{scores.mean():.4f}",
            f"{scores.std():.4f}",
            f"{np.median(scores):.4f}",
            f"{p10:.4f}",
            f"{p90:.4f}",
            f"{(scores > 0).mean():.1%}",
            f"{(scores <= 0).mean():.1%}",
        ],
    }
    col_stats, _ = st.columns([1, 2])
    with col_stats:
        st.dataframe(pd.DataFrame(stats), width='stretch', hide_index=True)

# ── Section C: Top 10 Most Persuadable Customer Profiles ─────────────────────
st.markdown("---")
st.markdown("## Top 10 Most Persuadable Customer Profiles")
st.caption(
    "These customers have the highest predicted incremental conversion probability if targeted "
    "with a promotion."
)

if score_col and score_col in df.columns:
    display_cols_raw = ['customer_id', score_col, 'recency', 'history', 'treatment', 'conversion']
    rename_map = {
        'customer_id': 'Customer ID',
        score_col:     'Predicted CATE',
        'recency':     'Days Since Last Purchase',
        'history':     'History ($)',
        'treatment':   'Was Treated',
        'conversion':  'Converted',
    }

    # Add segment cols if available
    for c in ['recency_bucket', 'spend_tier', 'ch_web', 'ch_phone', 'zip_urban']:
        if c in df.columns:
            display_cols_raw.append(c)
            rename_map[c] = c.replace('_', ' ').title()

    available_cols = [c for c in display_cols_raw if c in df.columns]
    top10 = (
        df.nlargest(10, score_col)[available_cols]
        .rename(columns=rename_map)
        .reset_index(drop=True)
    )
    top10.index = top10.index + 1
    top10.index.name = 'Rank'

    # Round CATE column
    cate_display = rename_map[score_col]
    top10[cate_display] = top10[cate_display].round(4)

    st.dataframe(top10, width='stretch')

# ── Section D: Feature Heterogeneity ─────────────────────────────────────────
st.markdown("---")
st.markdown("## Feature Heterogeneity")
st.markdown("Which customer characteristics predict responsiveness to promotions?")

if score_col and score_col in df.columns:
    numeric_features = [c for c in FEATURE_COLS if c in df.columns and df[c].dtype in [np.float64, np.int64, float, int]]

    # Also try raw columns from phase1 data
    fallback_features = [c for c in ['recency', 'history', 'mens', 'womens', 'newbie'] if c in df.columns]
    candidate_features = list(dict.fromkeys(numeric_features + fallback_features))

    if candidate_features:
        corrs = {
            f: df[[score_col, f]].dropna().corr().iloc[0, 1]
            for f in candidate_features
            if df[f].nunique() > 1
        }
        corr_df = (
            pd.DataFrame.from_dict(corrs, orient='index', columns=['correlation'])
            .abs()
            .sort_values('correlation', ascending=False)
            .reset_index()
            .rename(columns={'index': 'feature'})
        )

        colors_corr = ['#9b59b6' if v >= 0.05 else '#7f8c8d' for v in corr_df['correlation']]
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Bar(
            x=corr_df['feature'],
            y=corr_df['correlation'],
            marker_color=colors_corr,
            text=[f"{v:.3f}" for v in corr_df['correlation']],
            textposition='outside',
            textfont=dict(color=TEXT),
        ))
        fig_corr.update_xaxes(title_text='Feature')
        fig_corr.update_yaxes(title_text='|Correlation with CATE|')
        fig_corr = apply_dark_theme(
            fig_corr,
            'Which Features Drive Treatment Effect Heterogeneity?',
            height=420,
        )
        st.plotly_chart(fig_corr, width='stretch')

        st.caption(
            "Features with high |correlation| are stronger predictors of individual treatment "
            "responsiveness. Note: correlation is a linear proxy — it will miss non-linear "
            "heterogeneity that a tree-based model like X-Learner or DR-Learner can capture."
        )

        # CI heterogeneity from DR-Learner if available
        if 'dr_ci_upper' in df.columns and 'dr_ci_lower' in df.columns:
            df['ci_width'] = df['dr_ci_upper'] - df['dr_ci_lower']
            ci_zero_pct = df.get('dr_ci_includes_zero', pd.Series()).mean() * 100 if 'dr_ci_includes_zero' in df.columns else None

            st.markdown("#### DR-Learner Confidence Interval Width")
            col_ci1, col_ci2 = st.columns(2)
            col_ci1.metric("Mean CI Width", f"{df['ci_width'].mean():.4f}")
            if ci_zero_pct is not None:
                col_ci2.metric("CATEs Including Zero", f"{ci_zero_pct:.1f}%")
    else:
        st.info(
            "Feature heterogeneity analysis requires numeric feature columns. "
            "Run the feature engineering notebook to generate `hillstrom_features.parquet`."
        )
else:
    st.info("Load CATE scores to see feature heterogeneity analysis.")
