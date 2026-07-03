"""
Page 4 — Budget Optimizer
Set your campaign budget and see the optimal targeting strategy.
All computation is done in-memory from pre-loaded CATE scores — no model inference.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dashboard.utils.data_loader import (
    load_cate_scores,
    load_budget_results,
    get_best_cate_col,
)
from dashboard.utils.charts import (
    apply_dark_theme,
    budget_curve_plot,
    TEXT,
    BG_PAPER,
    BG_PLOT,
    GRID,
)

st.set_page_config(page_title='Budget Optimizer — CPIS', layout='wide')

st.title("💰 Budget Optimizer")
st.markdown("### Set your campaign budget and see the optimal targeting strategy")

df        = load_cate_scores()
budget_df = load_budget_results()

if df.empty:
    st.info("Run Phase 1 notebook (`phase1_naive_baseline.ipynb`) to generate customer score data.")
    st.stop()

# Normalise p_convert → naive_score
if 'p_convert' in df.columns and 'naive_score' not in df.columns:
    df['naive_score'] = df['p_convert']

best_cate_col = get_best_cate_col(df)
score_col     = '_best_cate' if '_best_cate' in df.columns else best_cate_col

if not score_col or score_col not in df.columns:
    st.error("No score column found in loaded data.")
    st.stop()

cate_scores = df[score_col].fillna(0).values
n           = len(cate_scores)
order       = np.argsort(-cate_scores)   # descending by score — computed once

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Campaign Parameters")
    cost_per_promo   = st.number_input("Discount cost per customer ($)", value=10, min_value=1, max_value=100)
    revenue_per_conv = st.number_input("Revenue per conversion ($)", value=50, min_value=10, max_value=500)

# ── Section A: Interactive Budget Slider ─────────────────────────────────────
st.markdown("---")

max_budget = int(min(n * cost_per_promo, 150_000))
budget = st.slider(
    "Campaign Budget ($)",
    min_value=1_000,
    max_value=max_budget,
    value=min(25_000, max_budget),
    step=1_000,
    format="$%d",
)

# Live computation from CATE scores (numpy sort only — no model inference)
k              = min(int(budget / cost_per_promo), n)
targeted_mask  = np.zeros(n, dtype=bool)
targeted_mask[order[:k]] = True

targeted_scores = cate_scores[targeted_mask]
lift            = targeted_scores[targeted_scores > 0].sum()   # only positive CATE contributes
revenue         = lift * revenue_per_conv
roi             = (revenue - budget) / budget * 100 if budget > 0 else 0
deadweight_k    = (targeted_scores <= 0).sum()
deadweight_pct  = deadweight_k / k * 100 if k > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Customers Targeted",
    f"{k:,}",
    f"{k/n:.1%} of population",
)
col2.metric(
    "Expected Incremental Lift",
    f"+{lift:.1f}",
    "conversions",
)
col3.metric(
    "Expected Revenue",
    f"${revenue:,.0f}",
    f"at ${revenue_per_conv}/conversion",
)
col4.metric(
    "Expected ROI",
    f"{roi:+.1f}%",
    delta_color="normal" if roi > 0 else "inverse",
)

st.markdown(f"""
<div style="background:#1a1a2e; border:1px solid #2c3e50; border-radius:8px; padding:0.8rem; margin:0.5rem 0;">
<span style="color:#95a5a6;">At <b style="color:#ecf0f1;">${budget:,}</b> budget you are targeting the top
<b style="color:#9b59b6;">{k/n:.1%}</b> of customers by persuadability.
<b style="color:#e74c3c;">{deadweight_pct:.1f}%</b> of targeted customers have CATE ≤ 0 (deadweight).</span>
</div>
""", unsafe_allow_html=True)

# ── Section B: Allocation Table ───────────────────────────────────────────────
st.markdown("---")
st.markdown("## Target Allocation — Top 50 Preview")

display_cols = ['customer_id', score_col]
for c in ['recency', 'history', 'treatment', 'conversion']:
    if c in df.columns:
        display_cols.append(c)

allocation = df.iloc[order[:k]][display_cols].copy()
rename_map = {
    'customer_id': 'Customer ID',
    score_col:     'Predicted CATE',
    'recency':     'Recency (months)',
    'history':     'History ($)',
    'treatment':   'Was Treated',
    'conversion':  'Converted',
}
allocation = allocation.rename(columns={k_: v for k_, v in rename_map.items() if k_ in allocation.columns})
allocation = allocation.sort_values('Predicted CATE', ascending=False).head(50)

try:
    styled = allocation.style.background_gradient(subset=['Predicted CATE'], cmap='Purples')
    st.dataframe(styled, use_container_width=True)
except Exception:
    st.dataframe(allocation, use_container_width=True)

# ── Section C: Pre-computed Budget Curve ──────────────────────────────────────
st.markdown("---")
st.markdown("## Budget Curve (Pre-computed)")

if not budget_df.empty and 'expected_lift' in budget_df.columns:
    fig_curve = budget_curve_plot(budget_df)

    # Annotate current slider budget
    min_b = budget_df['budget'].min()
    max_b = budget_df['budget'].max()
    if min_b <= budget <= max_b:
        fig_curve.add_vline(
            x=budget / 1000,
            line_color='#f39c12',
            line_dash='dot',
            line_width=2,
            annotation_text=f'Current: ${budget/1000:.0f}K',
            annotation_font_color='#f39c12',
        )

    st.plotly_chart(fig_curve, use_container_width=True)
else:
    st.info(
        "Pre-computed budget curve not found. "
        "Run `phase5a_budget_optimizer.ipynb` to generate `budget_optimizer_results.csv`."
    )

# ── Section D: Live Elbow Plot ────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Elbow Plot — Diminishing Returns")
st.markdown(
    "Cumulative incremental lift as budget grows. The elbow marks where additional spend "
    "yields diminishing returns."
)

fine_budgets = np.arange(1000, min(max_budget + 1000, 152_000), 2000)
sorted_scores = cate_scores[order]  # pre-sorted descending

fine_lifts = []
for b in fine_budgets:
    k_i = min(int(b / cost_per_promo), n)
    top_scores = sorted_scores[:k_i]
    fine_lifts.append(float(top_scores[top_scores > 0].sum()))

fine_lifts = np.array(fine_lifts)

# Detect elbow (max second derivative)
if len(fine_lifts) > 4:
    marginal     = np.diff(fine_lifts)
    second_deriv = np.diff(marginal)
    elbow_idx    = int(np.argmin(second_deriv)) + 1
    elbow_budget = fine_budgets[elbow_idx] / 1000
else:
    elbow_idx    = len(fine_budgets) // 2
    elbow_budget = fine_budgets[elbow_idx] / 1000

fig_elbow = go.Figure()

# Efficient zone (before elbow)
fig_elbow.add_trace(go.Scatter(
    x=fine_budgets[:elbow_idx + 1] / 1000,
    y=fine_lifts[:elbow_idx + 1],
    fill='tozeroy',
    fillcolor='rgba(46, 204, 113, 0.15)',
    line=dict(color='#2ecc71', width=2.5),
    name='Efficient Zone',
    mode='lines',
))

# Diminishing returns zone (after elbow)
fig_elbow.add_trace(go.Scatter(
    x=fine_budgets[elbow_idx:] / 1000,
    y=fine_lifts[elbow_idx:],
    fill='tozeroy',
    fillcolor='rgba(231, 76, 60, 0.10)',
    line=dict(color='#e74c3c', width=2, dash='dash'),
    name='Diminishing Returns',
    mode='lines',
))

# Current budget marker
current_idx  = np.searchsorted(fine_budgets, budget)
if 0 <= current_idx < len(fine_budgets):
    fig_elbow.add_trace(go.Scatter(
        x=[budget / 1000],
        y=[fine_lifts[min(current_idx, len(fine_lifts) - 1)]],
        mode='markers',
        name=f'Current Budget (${budget/1000:.0f}K)',
        marker=dict(color='#f39c12', size=12, symbol='diamond'),
    ))

fig_elbow.add_vline(
    x=elbow_budget,
    line_color='#f39c12', line_dash='dot', line_width=1.5,
    annotation_text=f'Elbow ≈ ${elbow_budget:.0f}K',
    annotation_font_color='#f39c12',
)

fig_elbow.update_xaxes(title_text='Budget ($K)')
fig_elbow.update_yaxes(title_text='Expected Incremental Conversions')
fig_elbow = apply_dark_theme(
    fig_elbow,
    'Cumulative Lift vs Budget — Efficient Zone vs Diminishing Returns',
    height=430,
)
st.plotly_chart(fig_elbow, use_container_width=True)

st.caption(
    f"Elbow detected at approximately **${elbow_budget:.0f}K** — budgets beyond this point yield "
    f"rapidly diminishing incremental conversions as the model exhausts the persuadable segment."
)
