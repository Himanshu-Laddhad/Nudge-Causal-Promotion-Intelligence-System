"""Page 1 — Business Problem: why standard ML wastes promotion budget."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dashboard.utils.data_loader import (
    load_cate_scores,
    load_master_comparison,
    load_budget_results,
    best_model_label,
    held_out,
)
from dashboard.utils.charts import (
    apply_dark_theme,
    persuadability_heatmap,
    pie_segments,
)


st.title("📊 The Business Problem")
st.markdown("### Why standard ML wastes your promotion budget")

# ── Section A: The Core Insight ───────────────────────────────────────────────
st.markdown("---")
st.markdown("## The Sleeping Dog Problem")

col_text, col_metric = st.columns([3, 2])

with col_text:
    st.markdown("""
**Standard ML models rank customers by P(convert | X)** — conversion probability.
Under a budget constraint this optimises the *wrong objective*: it targets loyal customers
who convert regardless of the promotion, cannibalising margin and wasting budget.

Nudge estimates the **Conditional Average Treatment Effect (CATE)**:

> τ(x) = E[Y(1) − Y(0) | X = x]

Customers with τ(x) ≤ 0 are **sleeping dogs** — targeting them is pure deadweight loss.
Customers with τ(x) >> 0 are **persuadables** — the only segment where a promotion
creates net-new revenue.
    """)

with col_metric:
    comparison = load_master_comparison()
    DW_COL = 'pct_sleeping_dogs_topk'
    if not comparison.empty and DW_COL in comparison.columns:
        naive_row = comparison[comparison['model'].str.contains('Naive|naive|XGB|xgb', na=False, regex=True)]
        if not naive_row.empty:
            dw = naive_row[DW_COL].iloc[0]
            st.metric(
                label="Naive XGB — sleeping dogs in top 20%",
                value=f"{dw:.2f}%",
                delta="of targeted spend is deadweight",
                delta_color="inverse",
            )
        best_label = best_model_label()
        best_row = comparison[comparison['model'] == best_label]
        if not best_row.empty:
            best_dw = best_row[DW_COL].iloc[0]
            st.metric(
                label=f"{best_label} — sleeping dogs in top 20%",
                value=f"{best_dw:.2f}%",
                delta=f"{best_dw - dw:+.2f} pp vs Naive XGB",
                delta_color="inverse",
            )
    else:
        st.info(
            "Run the Phase 2–4 notebooks to generate `master_comparison_table.csv` "
            "with full model comparison metrics."
        )

    st.markdown("""
    <div style="background:#1a1a2e; border:1px solid #2c3e50; border-radius:8px; padding:1rem; margin-top:1rem;">
    <b style="color:#9b59b6;">Key Insight</b><br>
    <span style="color:#ecf0f1; font-size:0.9rem;">
    The naive model's top-scoring customers convert at similar rates regardless of
    whether they received the promotion — revealing the model targets loyalty,
    not incremental persuadability.
    </span>
    </div>
    """, unsafe_allow_html=True)

# ── Section B: Persuadability Matrix ─────────────────────────────────────────
st.markdown("---")
st.markdown("## Persuadability Matrix")
st.markdown(
    "The heatmap below reveals the core problem: **high-score customers convert at nearly the "
    "same rate with or without the promotion.** The model learns loyalty, not incremental lift."
)

df = load_cate_scores()

if df.empty:
    st.info("Run Phase 1 notebook (`phase1_naive_baseline.ipynb`) to generate score data.")
else:
    score_col = '_best_cate'

    if score_col in df.columns:
        # Observed conversion rates must come from held-out rows only.
        quad = held_out(df).copy()
        threshold = np.percentile(quad[score_col], 80)
        quad['score_band'] = np.where(quad[score_col] >= threshold, 'High Score', 'Low Score')
        quad['arm'] = np.where(quad['treatment'] == 1, 'Treated', 'Control')

        pivot = quad.groupby(['score_band', 'arm'])['conversion'].mean().unstack()

        # Ensure both columns exist
        for col in ['Control', 'Treated']:
            if col not in pivot.columns:
                pivot[col] = 0.0

        pivot = pivot[['Control', 'Treated']]
        pivot = pivot.loc[['High Score', 'Low Score']] if 'High Score' in pivot.index else pivot

        z_vals = pivot.values.tolist()
        heatmap_data = {
            'z': z_vals,
            'x_labels': ['Control (No Promo)', 'Treated (Promo Sent)'],
            'y_labels': list(pivot.index),
        }

        col_hm, col_note = st.columns([2, 1])
        with col_hm:
            st.plotly_chart(persuadability_heatmap(heatmap_data), width='stretch')
        with col_note:
            if len(z_vals) >= 2:
                high_control   = z_vals[0][0] if z_vals[0] else 0
                high_treated   = z_vals[0][1] if len(z_vals[0]) > 1 else 0
                delta_high     = abs(high_treated - high_control)
                low_control    = z_vals[1][0] if len(z_vals) > 1 else 0
                low_treated    = z_vals[1][1] if len(z_vals) > 1 and len(z_vals[1]) > 1 else 0
                delta_low      = abs(low_treated - low_control)

                st.markdown(f"""
<div style="background:#1a1a2e; border:1px solid #2c3e50; border-radius:8px; padding:1rem;">
<b style="color:#e74c3c;">High-Score Customers</b><br>
<span style="color:#ecf0f1; font-size:0.9rem;">
Control: <b>{high_control:.2%}</b> → Treated: <b>{high_treated:.2%}</b><br>
Incremental lift: <b>{delta_high:.2%}</b>
</span><br><br>
<b style="color:#2ecc71;">Low-Score Customers</b><br>
<span style="color:#ecf0f1; font-size:0.9rem;">
Control: <b>{low_control:.2%}</b> → Treated: <b>{low_treated:.2%}</b><br>
Incremental lift: <b>{delta_low:.2%}</b>
</span><br><br>
<span style="color:#95a5a6; font-size:0.85rem;">
⚠️ High-score customers show nearly identical conversion rates with and without the
promotion. The naive model targets loyalty, not persuadability.
</span>
</div>
                """, unsafe_allow_html=True)

# ── Section C: Budget Comparison ($25K scenario) ──────────────────────────────
st.markdown("---")
st.markdown("## Budget Comparison — $25,000 Campaign")
st.markdown("Side-by-side: what happens when you spend the same budget with naive vs CATE targeting.")

budget_df = load_budget_results()

col_naive, col_cate = st.columns(2)

with col_naive:
    st.markdown("""
    <div style="background:#2d1515; border:1px solid #e74c3c; border-radius:10px; padding:1.5rem;">
    <h4 style="color:#e74c3c; margin:0 0 1rem 0;">❌ Naive XGB Targeting</h4>
    """, unsafe_allow_html=True)

    if not budget_df.empty and not budget_df[budget_df['budget'] == 25000].empty:
        r = budget_df[budget_df['budget'] == 25000].iloc[0]
        st.metric("Customers Targeted", f"{int(r['n_targeted']):,}")
        st.metric("Expected Lift", f"+{r['naive_lift']:.1f} conversions")
        st.metric("ROI", f"{r['naive_roi'] * 100:+.1f}%")
    else:
        st.info("Run the Phase 5A notebook to populate this comparison.")

    st.markdown("""
    <p style="color:#95a5a6; font-size:0.85rem; margin-top:0.75rem;">
    Targets highest P(convert) customers — many would convert without the promotion.
    </p>
    </div>
    """, unsafe_allow_html=True)

with col_cate:
    st.markdown(f"""
    <div style="background:#152d15; border:1px solid #2ecc71; border-radius:10px; padding:1.5rem;">
    <h4 style="color:#2ecc71; margin:0 0 1rem 0;">✅ {best_model_label()} CATE-Optimal Targeting</h4>
    """, unsafe_allow_html=True)

    if not budget_df.empty and not budget_df[budget_df['budget'] == 25000].empty:
        r = budget_df[budget_df['budget'] == 25000].iloc[0]
        st.metric("Customers Targeted", f"{int(r['n_targeted']):,}")
        st.metric(
            "Expected Lift",
            f"+{r['expected_lift']:.1f} conversions",
            delta=f"{r['lift_vs_naive']:+.1f} vs naive",
        )
        st.metric(
            "ROI",
            f"{r['expected_roi'] * 100:+.1f}%",
            delta=f"{r['roi_vs_naive'] * 100:+.1f} pp vs naive",
        )
    else:
        st.info("Run the Phase 5A notebook to populate this comparison.")

    st.markdown("""
    <p style="color:#95a5a6; font-size:0.85rem; margin-top:0.75rem;">
    Targets highest CATE customers — focuses budget on the persuadable segment only.
    </p>
    </div>
    """, unsafe_allow_html=True)

if not budget_df.empty:
    st.caption(
        "Both columns are scored on the same yardstick: the naive model only "
        "*selects* customers, and the resulting set is then valued using CATE. "
        "Scoring naive selections on P(convert) would compare two different "
        "quantities and flatter the baseline."
    )

# ── Section D: Deadweight Loss Breakdown ──────────────────────────────────────
st.markdown("---")
st.markdown("## Deadweight Loss Breakdown")
st.markdown("Which segment of customers is the model targeting — and who are they actually?")

if not df.empty and '_best_cate' in df.columns:
    cate = df['_best_cate'].values

    # Only show CATE segments if scores look like actual CATEs (can be negative)
    if cate.min() < 0:
        persuadable_n  = int((cate > 0.01).sum())
        neutral_n      = int(((cate >= -0.01) & (cate <= 0.01)).sum())
        sleeping_dog_n = int((cate < -0.01).sum())

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Persuadable (CATE > 0.01)", f"{persuadable_n:,}", f"{persuadable_n/len(cate):.1%}")
        col_m2.metric("Neutral (|CATE| ≤ 0.01)", f"{neutral_n:,}", f"{neutral_n/len(cate):.1%}")
        col_m3.metric("Sleeping Dog (CATE < -0.01)", f"{sleeping_dog_n:,}", f"{sleeping_dog_n/len(cate):.1%}")

        # Top 20% analysis
        top20_mask = cate >= np.percentile(cate, 80)
        top20_sleeping = (cate[top20_mask] < 0).mean() * 100
        st.metric(
            "Sleeping Dogs in Top 20% (Naive Targets)",
            f"{top20_sleeping:.1f}%",
            delta="of naive top-targets are non-persuadable",
            delta_color="inverse",
        )

        col_pie, col_explain = st.columns([2, 1])
        with col_pie:
            fig_pie = pie_segments(
                labels=['Persuadable (CATE > 0.01)', 'Neutral (|CATE| ≤ 0.01)', 'Sleeping Dog (CATE < -0.01)'],
                values=[persuadable_n, neutral_n, sleeping_dog_n],
                colors=['#2ecc71', '#f39c12', '#e74c3c'],
                title='Customer Population by Persuadability',
            )
            st.plotly_chart(fig_pie, width='stretch')

        with col_explain:
            st.markdown(f"""
<div style="background:#1a1a2e; border:1px solid #2c3e50; border-radius:8px; padding:1rem;">
<b style="color:#2ecc71;">Persuadable</b>
<p style="color:#ecf0f1; font-size:0.9rem;">CATE > 0.01 — promotion increases their probability of converting.
These are your true targets.</p>

<b style="color:#f39c12;">Neutral</b>
<p style="color:#ecf0f1; font-size:0.9rem;">|CATE| ≤ 0.01 — promotion has negligible effect.
Promotion is wasted, no harm done.</p>

<b style="color:#e74c3c;">Sleeping Dog</b>
<p style="color:#ecf0f1; font-size:0.9rem;">CATE < -0.01 — promotion may actually reduce conversion
(reactance, price anchoring). Active harm.</p>
</div>
            """, unsafe_allow_html=True)
    else:
        # Phase 1 naive scores are probabilities (0–1), not CATEs — show different breakdown
        st.info(
            "CATE-based segment breakdown requires Phase 2+ outputs. "
            "Currently showing Phase 1 naive scores (P(convert) scale)."
        )
        eval_df = held_out(df)
        eval_cate = eval_df['_best_cate'].values
        top20_mask = eval_cate >= np.percentile(eval_cate, 80)
        in_top20   = top20_mask.sum()
        ctrl_conv_top20     = eval_df.loc[top20_mask & (eval_df['treatment'] == 0), 'conversion'].mean() if 'treatment' in eval_df.columns else 0
        treated_conv_top20  = eval_df.loc[top20_mask & (eval_df['treatment'] == 1), 'conversion'].mean() if 'treatment' in eval_df.columns else 0

        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Customers in Top 20%", f"{in_top20:,}")
        col_s2.metric("Control Conv. Rate (Top 20%)", f"{ctrl_conv_top20:.2%}")
        col_s3.metric("Treated Conv. Rate (Top 20%)", f"{treated_conv_top20:.2%}")

        if ctrl_conv_top20 > 0:
            incremental = (treated_conv_top20 - ctrl_conv_top20) / ctrl_conv_top20 * 100
            st.metric(
                "Incremental Lift in Top 20%",
                f"{incremental:+.1f}%",
                delta="relative lift from promotion",
                delta_color="normal" if incremental > 0 else "inverse",
            )
else:
    st.info(
        "Run any Phase notebook to populate score data for the deadweight analysis. "
        "At minimum, `phase1_naive_baseline.ipynb` is needed."
    )
