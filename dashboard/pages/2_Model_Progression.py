"""
Page 2 — Model Progression
How causal ML improves over standard propensity models.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dashboard.utils.data_loader import (
    load_cate_scores,
    load_master_comparison,
    get_best_cate_col,
)
from dashboard.utils.charts import (
    apply_dark_theme,
    qini_curve_plot,
    bar_comparison,
    cate_histogram,
    PALETTE,
    TEXT,
    BG_PAPER,
    BG_PLOT,
    GRID,
)
import plotly.graph_objects as go

st.set_page_config(page_title='Model Progression — CPIS', layout='wide')

st.title("📈 Model Progression")
st.markdown("### How causal ML improves over standard propensity models")

df       = load_cate_scores()
comp_df  = load_master_comparison()

# ── Section A: Qini Curve Comparison ─────────────────────────────────────────
st.markdown("---")
st.markdown("## Qini Curves — Model Comparison")
st.markdown(
    "Qini curves measure *incremental* targeting efficiency. A steeper early rise means the model "
    "correctly ranks the most persuadable customers at the top."
)

if df.empty:
    st.info("Run Phase 1–4 notebooks to generate model score data.")
else:
    # Build models_data list from available columns
    model_col_map = {
        'Naive XGB':     'naive_score',
        'T-Learner':     'cate_t_learner',
        'S-Learner':     'cate_s_learner',
        'X-Learner':     'cate_x_learner',
        'Causal Forest': 'cate_cf',
        'DR-Learner':    'cate_dr_clean',
    }
    # p_convert is phase-1 proxy for naive_score
    if 'naive_score' not in df.columns and 'p_convert' in df.columns:
        df['naive_score'] = df['p_convert']

    models_data = []
    for name, col in model_col_map.items():
        if col in df.columns:
            valid = df[col].notna()
            models_data.append({
                'name':      name,
                'scores':    df.loc[valid, col].values,
                'y_true':    df.loc[valid, 'conversion'].values,
                'treatment': df.loc[valid, 'treatment'].values,
            })

    if models_data:
        st.plotly_chart(qini_curve_plot(models_data), use_container_width=True)
    else:
        st.warning("No compatible score columns found in loaded data.")

    with st.expander("What is a Qini Curve?"):
        st.markdown("""
**Qini curves** measure the *incremental* benefit of targeting customers in rank order by model score.

- **X-axis**: % of population targeted (ordered by descending score)
- **Y-axis**: Cumulative incremental conversions = (treated converts) − (control converts × treatment rate)
- **Area above random diagonal (Qini AUC)**: Higher is better — more persuadables ranked at the top

**Why not AUC-ROC?** AUC-ROC measures discrimination between buyers and non-buyers regardless of treatment.
A model with perfect AUC-ROC can have *zero* incremental lift if it just learns who buys unconditionally.
Qini directly rewards models that rank *persuadables* first.
        """)

# ── Section B: AUUC / Qini AUC Bar Chart ─────────────────────────────────────
st.markdown("---")
st.markdown("## Model Performance — Qini AUC Ranking")

if not comp_df.empty and 'qini_auc' in comp_df.columns:
    comp_sorted = comp_df.sort_values('qini_auc', ascending=False)

    fig_bar = bar_comparison(
        comp_sorted, 'model', 'qini_auc',
        title='Qini AUC by Model (Higher = Better Uplift Targeting)',
        color_map=PALETTE,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Model Descriptions")
    desc_data = {
        'Model': ['Naive XGB', 'T-Learner', 'S-Learner', 'X-Learner', 'Causal Forest', 'DR-Learner'],
        'Objective': [
            'P(convert | X, T=1)',
            'μ₁(X) − μ₀(X)',
            'μ(X,1) − μ(X,0)',
            'Cross pseudo-outcomes + propensity blend',
            'Honest splits on τ heterogeneity',
            'Doubly robust semiparametric',
        ],
        'Key Risk': [
            'Wrong objective — conflates loyalty with lift',
            'High variance with imbalanced arms',
            'Treatment shrinkage (T feature dominated)',
            'Requires reasonable propensity estimation',
            'Computationally intensive',
            'Requires correct nuisance model spec',
        ],
    }
    st.dataframe(pd.DataFrame(desc_data), use_container_width=True, hide_index=True)
else:
    st.info(
        "Run Phase 2–4 notebooks to generate `master_comparison_table.csv` with Qini AUC scores. "
        "Showing model descriptions only."
    )
    desc_data = {
        'Model': ['Naive XGB', 'T-Learner', 'S-Learner', 'X-Learner', 'Causal Forest', 'DR-Learner'],
        'Type': ['Propensity', 'Meta-Learner', 'Meta-Learner', 'Meta-Learner', 'Non-parametric', 'Semiparametric'],
        'Description': [
            'Trains on T=1 only, scores all. Wrong objective — conflates P(buy) with lift.',
            'Separate μ₁(X), μ₀(X). CATE = μ₁ − μ₀. Unbiased under RCT, high variance with small arms.',
            'Single model with T as feature. CATE = μ(X,1) − μ(X,0). Treatment shrinkage risk.',
            'Cross-fitted pseudo-outcomes + propensity blending. Best when treatment is rare.',
            'Honest trees, splits maximise τ heterogeneity, provides valid CIs via bootstrap.',
            'Doubly robust: survives one misspecified nuisance model (propensity or outcome).',
        ],
    }
    st.dataframe(pd.DataFrame(desc_data), use_container_width=True, hide_index=True)

# ── Section C: Deadweight Loss Comparison ────────────────────────────────────
st.markdown("---")
st.markdown("## Deadweight Loss per Model")
st.markdown("Percentage of targeted population with CATE ≤ 0 — lower is better.")

if not comp_df.empty and 'deadweight_pct' in comp_df.columns:
    dw_sorted = comp_df.sort_values('deadweight_pct', ascending=True)

    dw_colors = [PALETTE.get(m, '#3498db') for m in dw_sorted['model']]
    fig_dw = go.Figure()
    fig_dw.add_trace(go.Bar(
        x=dw_sorted['model'],
        y=dw_sorted['deadweight_pct'],
        marker_color=dw_colors,
        text=[f"{v:.1f}%" for v in dw_sorted['deadweight_pct']],
        textposition='outside',
        textfont=dict(color=TEXT),
    ))
    fig_dw.update_yaxes(title_text='% of Targeted Population with CATE ≤ 0')
    fig_dw = apply_dark_theme(
        fig_dw,
        'Deadweight Loss: % of Promotions Wasted on Non-Persuadables',
        height=400,
    )
    st.plotly_chart(fig_dw, use_container_width=True)
else:
    st.info(
        "Deadweight loss breakdown requires `master_comparison_table.csv` with a "
        "`deadweight_pct` column. Run Phase 2–4 notebooks."
    )

# ── Section D: Model Cards (expandable) ──────────────────────────────────────
st.markdown("---")
st.markdown("## Model Cards")

model_cards = [
    {
        'name': 'Naive XGB (Baseline)',
        'color': '#e74c3c',
        'formula': r'\hat{p}(x) = P(\text{convert} \mid X = x, T = 1)',
        'description': (
            'Standard propensity model. Trained on treated customers only, then scored across all customers. '
            'This is the **wrong objective** for uplift — it estimates who *will* buy, not who will buy '
            '*because of* the promotion.'
        ),
        'pros': ['Simple, fast', 'Widely understood', 'Strong AUC-ROC baseline'],
        'cons': [
            'Wrong target variable (P(buy) not CATE)',
            'Systematically targets loyal customers',
            'High deadweight loss under budget constraint',
        ],
    },
    {
        'name': 'T-Learner',
        'color': '#3498db',
        'formula': r'\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)',
        'description': (
            'Two separate outcome models: one trained on treated, one on control. '
            'CATE = difference in predictions. Unbiased under RCT (no confounding), '
            'but high variance when treatment/control arms are imbalanced.'
        ),
        'pros': ['Unbiased under RCT', 'Flexible — any base learner', 'Intuitive'],
        'cons': [
            'High variance with small control arm',
            'No information sharing between μ₁ and μ₀',
        ],
    },
    {
        'name': 'S-Learner',
        'color': '#2ecc71',
        'formula': r'\hat{\tau}(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)',
        'description': (
            'Single outcome model with T as a feature. CATE estimated as counterfactual difference. '
            'Risk: tree-based models may shrink the treatment feature if it has low marginal importance '
            'relative to X.'
        ),
        'pros': ['Uses all data jointly', 'Regularises toward zero (conservative)'],
        'cons': [
            'Treatment shrinkage in tree-based learners',
            'May underestimate true heterogeneity',
        ],
    },
    {
        'name': 'X-Learner',
        'color': '#f39c12',
        'formula': (
            r'\tilde{D}_i^1 = Y_i - \hat{\mu}_0(X_i),\quad '
            r'\tilde{D}_i^0 = \hat{\mu}_1(X_i) - Y_i,\quad '
            r'\hat{\tau}(x) = g(x)\hat{\tau}_0 + (1-g(x))\hat{\tau}_1'
        ),
        'description': (
            'Cross-fitted pseudo-outcomes blended via propensity weighting. '
            'Particularly efficient when treatment arms are highly imbalanced (e.g. 95/5 split). '
            'In balanced RCTs, improvement over T-Learner is modest.'
        ),
        'pros': [
            'Efficient with imbalanced treatment arms',
            'Flexible propensity blending',
            'Lower variance than T-Learner when arms differ',
        ],
        'cons': ['More complex pipeline', 'Requires propensity estimation'],
    },
    {
        'name': 'Causal Forest',
        'color': '#1abc9c',
        'formula': (
            r'\hat{\tau}(x) = \text{honest forest splits maximising } '
            r'\text{Var}(\tau(x))\text{, with kernel CIs}'
        ),
        'description': (
            'Non-parametric CATE estimator using honest causal trees. '
            'Splits are chosen to maximise treatment effect heterogeneity rather than outcome prediction. '
            'Provides asymptotically valid confidence intervals via the infinitesimal jackknife.'
        ),
        'pros': [
            'Asymptotically valid CIs',
            'No functional form assumptions',
            'Captures high-dimensional heterogeneity',
        ],
        'cons': ['Computationally expensive', 'No extrapolation outside support'],
    },
    {
        'name': 'DR-Learner',
        'color': '#9b59b6',
        'formula': (
            r'\tilde{Y}_i = \hat{\tau}(X_i) + \frac{T_i - \hat{e}(X_i)}{\hat{e}(X_i)(1-\hat{e}(X_i))}'
            r'(Y_i - \hat{\mu}_{T_i}(X_i))'
        ),
        'description': (
            'Doubly robust estimator: consistent if *either* the propensity model e(X) or the outcome model '
            'μ(X,T) is correctly specified (but not both need to be). '
            'Achieves semiparametric efficiency bound, making it the most robust and efficient choice.'
        ),
        'pros': [
            'Doubly robust — survives one misspecification',
            'Semiparametric efficiency bound',
            'Best Qini AUC in practice',
        ],
        'cons': [
            'Requires fitting two nuisance models',
            'Cross-fitting adds complexity',
        ],
    },
]

for card in model_cards:
    with st.expander(f"**{card['name']}**", expanded=False):
        col_desc, col_stats = st.columns([2, 1])
        with col_desc:
            st.markdown(
                f"<div style='border-left:3px solid {card['color']}; padding-left:1rem;'>",
                unsafe_allow_html=True,
            )
            st.markdown(card['description'])
            st.latex(card['formula'])
            st.markdown("</div>", unsafe_allow_html=True)
        with col_stats:
            st.markdown("**Pros**")
            for p in card['pros']:
                st.markdown(f"✅ {p}")
            st.markdown("**Cons**")
            for c in card['cons']:
                st.markdown(f"⚠️ {c}")

            if not comp_df.empty and 'model' in comp_df.columns and 'qini_auc' in comp_df.columns:
                row = comp_df[comp_df['model'].str.contains(
                    card['name'].split(' ')[0], na=False, case=False
                )]
                if not row.empty:
                    st.metric("Qini AUC", f"{row['qini_auc'].iloc[0]:.4f}")
