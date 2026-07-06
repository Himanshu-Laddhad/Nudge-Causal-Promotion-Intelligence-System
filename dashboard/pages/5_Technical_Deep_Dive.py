"""Page 5 — Technical Deep Dive: robustness experiment and model validation."""
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
    load_robustness_results,
    load_decile_uplift,
    get_best_cate_col,
)
from dashboard.utils.charts import (
    apply_dark_theme,
    bar_comparison,
    PALETTE,
    TEXT,
    BG_PAPER,
    BG_PLOT,
    GRID,
)

st.set_page_config(page_title='Technical Deep Dive — Nudge', layout='wide')

st.title("🔬 Technical Deep Dive")
st.markdown("### Causal identification, robustness, and model validation")

df         = load_cate_scores()
comp_df    = load_master_comparison()
robust_df  = load_robustness_results()
decile_df  = load_decile_uplift()

# ── Section A: Robustness Experiment ─────────────────────────────────────────
st.markdown("---")
st.markdown("## Robustness Experiment")
st.markdown(
    "What happens to model performance when we artificially introduce confounding? "
    "A doubly robust estimator should degrade less than a naive meta-learner."
)

if robust_df.empty:
    st.info(
        "Run `phase4_dr_learner_robustness.ipynb` to generate `robustness_experiment_results.csv`. "
        "This notebook introduces synthetic confounding and re-evaluates all models."
    )
else:
    col_tbl, col_chart = st.columns([1, 2])

    with col_tbl:
        st.markdown("#### Robustness Results Table")
        display_df = robust_df.copy()
        if 'Pct_Drop' in display_df.columns:
            display_df['Pct_Drop'] = display_df['Pct_Drop'].map(lambda x: f"{x:.1f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col_chart:
        fig_rob = go.Figure()
        for model_name in robust_df['Model'].unique():
            row   = robust_df[robust_df['Model'] == model_name].iloc[0]
            color = PALETTE.get(model_name, '#3498db')

            fig_rob.add_trace(go.Bar(
                name=f'{model_name} — Clean',
                x=[model_name],
                y=[row['Qini_Clean']],
                marker_color=color,
                opacity=0.9,
                text=[f"{row['Qini_Clean']:.4f}"],
                textposition='outside',
                textfont=dict(color=TEXT),
            ))
            fig_rob.add_trace(go.Bar(
                name=f'{model_name} — Confounded',
                x=[model_name],
                y=[row['Qini_Confounded']],
                marker_color=color,
                opacity=0.45,
                text=[f"{row['Qini_Confounded']:.4f}"],
                textposition='outside',
                textfont=dict(color=TEXT),
            ))

        fig_rob.update_layout(barmode='group')
        fig_rob = apply_dark_theme(fig_rob, 'Qini AUC: Clean vs Confounded Data', height=420)
        st.plotly_chart(fig_rob, use_container_width=True)

    with st.expander("What confounding was introduced?"):
        st.markdown("""
**Synthetic confounding experiment** (from Phase 4 notebook):

1. A hidden variable `Z` (e.g. "is a VIP customer") is constructed from existing features.
2. `Z` is made to correlate with **both** treatment assignment and the outcome.
3. All models are re-evaluated without access to `Z` — simulating a real-world observational setting.

**Expected result:**
- **T-Learner / S-Learner**: Performance degrades significantly — these models assume unconfoundedness
  and have no mechanism to correct for selection bias.
- **DR-Learner**: Smaller degradation — doubly robust property means it remains consistent
  if *either* the propensity model e(X) or the outcome model μ(X,T) is correctly specified
  (even when neither sees Z directly).

The `Pct_Drop` column shows the % reduction in Qini AUC from clean to confounded data.
        """)

# ── Section B: CATE Confidence Intervals ─────────────────────────────────────
st.markdown("---")
st.markdown("## Causal Forest Confidence Intervals")

has_ci = (
    not df.empty
    and 'cf_ci_lower' in df.columns
    and 'cf_ci_upper' in df.columns
)

if not has_ci:
    st.info(
        "Causal Forest confidence intervals require `phase3_cate_predictions.parquet` with "
        "columns `cf_ci_lower`, `cf_ci_upper`, `cf_ci_includes_zero`. "
        "Run `phase3_causal_forest.ipynb`."
    )
else:
    df['ci_width'] = df['cf_ci_upper'] - df['cf_ci_lower']
    ci_zero_col    = 'cf_ci_includes_zero'

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Mean CATE", f"{df['cate_cf'].mean():.4f}")
    col_m2.metric("Mean CI Width", f"{df['ci_width'].mean():.4f}")
    if ci_zero_col in df.columns:
        ci_zero_pct = df[ci_zero_col].mean() * 100
        col_m3.metric("Statistically Insignificant CATEs", f"{ci_zero_pct:.1f}%",
                      delta="CI includes 0", delta_color="off")

    # Scatter: CATE vs CI width, coloured by significance
    if ci_zero_col in df.columns:
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig_scatter = go.Figure()

        for sig, color, name in [
            (True,  '#e74c3c', 'CI includes 0 (not significant)'),
            (False, '#2ecc71', 'Significant (CI excludes 0)'),
        ]:
            mask = sample[ci_zero_col] == sig
            if mask.any():
                fig_scatter.add_trace(go.Scatter(
                    x=sample.loc[mask, 'cate_cf'],
                    y=sample.loc[mask, 'ci_width'],
                    mode='markers',
                    name=name,
                    marker=dict(color=color, opacity=0.4, size=4),
                ))

        fig_scatter.update_xaxes(title_text='CATE Estimate')
        fig_scatter.update_yaxes(title_text='CI Width')
        fig_scatter = apply_dark_theme(
            fig_scatter,
            'CATE Estimates vs Confidence Interval Width',
            height=430,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Wider CIs indicate higher uncertainty. Red points = CI crosses zero (not statistically significant).")

# ── Section C: Decile Validation ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## Decile Validation")
st.markdown(
    "If the model correctly ranks persuadability, top deciles should show higher actual uplift "
    "(treated conversion rate − control conversion rate)."
)

if decile_df.empty:
    st.info(
        "Run `phase2_meta_learners.ipynb` to generate `phase2_decile_uplift.csv`. "
        "This validation checks whether model rankings correspond to actual treatment effects."
    )
else:
    col_decile, col_decile_table = st.columns([2, 1])

    with col_decile:
        fig_decile = go.Figure()

        if 'actual_uplift' in decile_df.columns:
            fig_decile.add_trace(go.Scatter(
                x=decile_df['decile'],
                y=decile_df['actual_uplift'],
                mode='lines+markers',
                name='Actual Uplift (Treat − Control)',
                line=dict(color='#2ecc71', width=2.5),
                marker=dict(size=8),
            ))

        if 'avg_cate_x' in decile_df.columns:
            fig_decile.add_trace(go.Scatter(
                x=decile_df['decile'],
                y=decile_df['avg_cate_x'],
                mode='lines+markers',
                name='Predicted CATE (X-Learner)',
                line=dict(color='#f39c12', width=2, dash='dash'),
                marker=dict(size=8, symbol='square'),
            ))

        fig_decile.update_xaxes(title_text='Decile (1 = Highest Predicted CATE)')
        fig_decile.update_yaxes(title_text='Conversion Rate Uplift')
        fig_decile = apply_dark_theme(
            fig_decile,
            'Decile Validation: Does the Model Correctly Rank Persuadability?',
            height=430,
        )
        st.plotly_chart(fig_decile, use_container_width=True)

    with col_decile_table:
        st.dataframe(decile_df, use_container_width=True, hide_index=True)

    st.caption(
        "A well-calibrated model shows monotonically decreasing actual uplift from decile 1 to 10. "
        "Non-monotonicity in lower deciles is expected — the signal is weakest for low-CATE customers."
    )

# ── Section D: Model Cards (tabbed) ──────────────────────────────────────────
st.markdown("---")
st.markdown("## Model Reference Cards")

tab_names = ['Naive XGB', 'T-Learner', 'S-Learner', 'X-Learner', 'Causal Forest', 'DR-Learner']
tabs = st.tabs(tab_names)

model_info = {
    'Naive XGB': {
        'type': 'Propensity Model',
        'assumption': 'Stable Unit Treatment Value (SUTVA); no causal identification goal',
        'formula': r'P(\text{convert} \mid X, T=1)',
        'formula_display': 'P(convert | X, T=1)',
        'pros': 'Simple, fast, widely understood',
        'cons': 'Wrong objective — conflates loyalty with lift. Systematic deadweight loss.',
        'summary': (
            'Trains a gradient boosted classifier on treated customers only, then scores all customers. '
            'Despite strong AUC-ROC, this model optimises the wrong target: it identifies *who buys*, '
            'not *who buys because of the promotion*.'
        ),
    },
    'T-Learner': {
        'type': 'Meta-Learner',
        'assumption': 'Unconfoundedness: Y(0), Y(1) ⊥ T | X',
        'formula': r'\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)',
        'formula_display': 'τ̂(x) = μ̂₁(x) − μ̂₀(x)',
        'pros': 'Unbiased under RCT; flexible base learner; interpretable',
        'cons': 'High variance with imbalanced arms; no cross-arm information sharing',
        'summary': (
            'Two separate outcome models: μ₁ on treated, μ₀ on control. '
            'CATE = difference in predictions. In the Hillstrom RCT (balanced), '
            'this is straightforward but can be noisy on small control arms.'
        ),
    },
    'S-Learner': {
        'type': 'Meta-Learner',
        'assumption': 'Unconfoundedness; T is modelled as just another feature',
        'formula': r'\hat{\tau}(x) = \hat{\mu}(x, T{=}1) - \hat{\mu}(x, T{=}0)',
        'formula_display': 'τ̂(x) = μ̂(x,1) − μ̂(x,0)',
        'pros': 'Uses all data jointly; regularises toward zero',
        'cons': 'Treatment shrinkage — tree splitters may ignore T if it has low marginal importance',
        'summary': (
            'Single outcome model with T as an input feature. '
            'CATE estimated as counterfactual prediction difference. '
            'Risk: XGBoost/RF may assign low importance to T relative to rich X features, '
            'leading to under-estimated treatment effects.'
        ),
    },
    'X-Learner': {
        'type': 'Meta-Learner',
        'assumption': 'Unconfoundedness; requires propensity model e(X)',
        'formula': (
            r'\tilde{D}^1_i = Y_i - \hat{\mu}_0(X_i),\quad '
            r'\tilde{D}^0_i = \hat{\mu}_1(X_i) - Y_i'
        ),
        'formula_display': 'D̃¹ᵢ = Yᵢ − μ̂₀(Xᵢ), then blended via propensity ê(x)',
        'pros': 'Efficient with imbalanced treatment arms; lower variance than T-Learner in that regime',
        'cons': 'Requires propensity estimation; modest gain over T-Learner in balanced RCTs',
        'summary': (
            'Cross-fits pseudo-outcomes: treated units use μ₀ counterfactual, '
            'control units use μ₁ counterfactual. '
            'Final CATE blended by propensity: τ̂(x) = ê(x)τ̂₀ + (1−ê(x))τ̂₁. '
            'Best advantage emerges when one arm is much smaller.'
        ),
    },
    'Causal Forest': {
        'type': 'Non-parametric',
        'assumption': 'Unconfoundedness; overlap (0 < e(x) < 1)',
        'formula': (
            r'\text{Honest splits on } \hat{\text{Var}}(\tau(x)); '
            r'\text{ IJ-bootstrap CIs}'
        ),
        'formula_display': 'Honest splits maximising Var(τ(x)); infinitesimal jackknife CIs',
        'pros': 'Valid asymptotic CIs; no functional form assumptions; captures high-dim heterogeneity',
        'cons': 'Computationally expensive; no extrapolation',
        'summary': (
            'Random forest variant where each tree uses "honest" estimation: '
            'one half of the sample determines splits, the other estimates leaf means. '
            'Splits are chosen to maximise treatment effect *variance* (heterogeneity), not outcome fit. '
            'Infinitesimal jackknife provides pointwise confidence intervals.'
        ),
    },
    'DR-Learner': {
        'type': 'Semiparametric',
        'assumption': 'Unconfoundedness + overlap; correct spec of e(X) OR μ(X,T)',
        'formula': (
            r'\tilde{Y}_i = \hat{\tau}(X_i) + '
            r'\frac{T_i - \hat{e}(X_i)}{\hat{e}(X_i)(1-\hat{e}(X_i))}'
            r'(Y_i - \hat{\mu}_{T_i}(X_i))'
        ),
        'formula_display': 'Augmented IPW pseudo-outcome → final CATE regression',
        'pros': (
            'Doubly robust: consistent if either e(X) or μ(X,T) is correct. '
            'Achieves semiparametric efficiency bound.'
        ),
        'cons': 'Requires two nuisance models; cross-fitting adds pipeline complexity',
        'summary': (
            'Constructs doubly robust pseudo-outcomes, then regresses them on X to obtain CATE. '
            'The IPW correction term debiases the outcome model; the outcome model stabilises IPW. '
            'Under cross-fitting (sample splitting), the final estimator is consistent even if '
            'one nuisance model is misspecified.'
        ),
    },
}

for tab, name in zip(tabs, tab_names):
    info = model_info[name]
    with tab:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown(f"**Type:** {info['type']}")
            st.markdown(f"**Causal Assumption:** {info['assumption']}")
            st.markdown(f"**Summary:** {info['summary']}")
            st.markdown("**Key Formula:**")
            st.latex(info['formula'])

        with col_b:
            st.markdown("**Pros:**")
            st.markdown(info['pros'])
            st.markdown("**Cons:**")
            st.markdown(info['cons'])

            if not comp_df.empty and 'model' in comp_df.columns:
                key = name.split(' ')[0]
                row = comp_df[comp_df['model'].str.contains(key, na=False, case=False)]
                if not row.empty:
                    if 'qini_auc' in row.columns:
                        st.metric("Qini AUC", f"{row['qini_auc'].iloc[0]:.4f}")
                    if 'deadweight_pct' in row.columns:
                        st.metric("Deadweight Loss", f"{row['deadweight_pct'].iloc[0]:.1f}%")
                    if 'mean_cate' in row.columns:
                        st.metric("Mean CATE", f"{row['mean_cate'].iloc[0]:.4f}")
