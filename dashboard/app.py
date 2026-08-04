"""Nudge — Streamlit multi-page dashboard. Reads pre-computed outputs from outputs/."""
import streamlit as st

st.set_page_config(
    page_title='Nudge — Causal Promotion Targeting',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Global dark CSS
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f0f23; }
    .block-container { background-color: #0f0f23; padding-top: 2rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #16213e;
        border-right: 1px solid #2c3e50;
    }
    section[data-testid="stSidebar"] * { color: #ecf0f1 !important; }

    /* Text */
    h1, h2, h3, h4, p, li { color: #ecf0f1 !important; }
    .stMarkdown { color: #ecf0f1; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #16213e;
        border: 1px solid #2c3e50;
        border-radius: 8px;
        padding: 1rem;
    }
    div[data-testid="metric-container"] label { color: #95a5a6 !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #ecf0f1 !important;
        font-size: 1.6rem !important;
    }

    /* Dataframes */
    .stDataFrame { background-color: #16213e; }

    /* Sliders */
    .stSlider > div > div > div { background-color: #9b59b6 !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        color: #95a5a6;
        background-color: #16213e;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #9b59b6 !important;
        border-bottom-color: #9b59b6 !important;
    }

    /* Info/warning boxes */
    .stAlert { background-color: #16213e !important; border-color: #2c3e50 !important; }

    /* Select boxes and inputs */
    .stSelectbox > div > div { background-color: #16213e !important; color: #ecf0f1 !important; }
    .stNumberInput > div > div > input { background-color: #16213e !important; color: #ecf0f1 !important; }

    /* Expanders */
    .streamlit-expanderHeader { color: #ecf0f1 !important; background-color: #16213e !important; }
    .streamlit-expanderContent { background-color: #1a1a2e !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎯 Nudge")
    st.markdown("**Causal Promotion Targeting**")
    st.markdown("---")
    st.markdown("""
**Navigate:**
1. 📊 Business Problem
2. 📈 Model Progression
3. 👥 Who Gets Targeted?
4. 💰 Budget Optimizer
5. 🔬 Technical Deep Dive
    """)
    st.markdown("---")
    st.markdown("""
**Dataset:** Hillstrom Email Marketing RCT
**64,000** customers · **3-arm** RCT
**Outcome:** 2-week conversion
    """)
    st.markdown("---")
    st.caption("Phase 5B · Portfolio Project")

st.title("Nudge — Causal Promotion Targeting")
st.markdown("""
### Identify who is genuinely persuaded by a promotion — and who would have bought anyway.

Nudge uses causal ML to estimate the **Conditional Average Treatment Effect (CATE)** — the incremental
lift a promotion *causes* for each individual — and allocates budget only to the genuinely persuadable
segment, eliminating deadweight spend on loyal customers who convert regardless.
""")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dashboard.utils.data_loader import phases_available, best_model_label

col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset", "64K customers")
col2.metric("Design", "3-arm RCT")
col3.metric("Models", "5 (Naive → DR)")
col4.metric("Best Model", best_model_label(), help="Highest Qini AUC in master_comparison_table.csv")

st.markdown("---")

avail = phases_available()
phase_labels = {
    'phase1': 'Phase 1 — Naive XGBoost',
    'phase2': 'Phase 2 — Meta-Learners',
    'phase4': 'Phase 4 — DR-Learner & Robustness',
    'features': 'Feature Parquet',
    'master': 'Master Comparison Table',
    'budget': 'Budget Optimizer Results',
    'robustness': 'Robustness Experiment',
    'decile': 'Decile Validation',
    'breakeven': 'Break-Even Analysis',
}

st.markdown("### Data Status")
status_cols = st.columns(3)
items = list(phase_labels.items())
for i, (key, label) in enumerate(items):
    icon = "✅" if avail.get(key) else "⏳"
    status_cols[i % 3].markdown(f"{icon} {label}")

st.markdown("---")
st.markdown("""
**Select a page from the sidebar** to explore the analysis.
*Problem* → *Solution* → *Who responds?* → *Optimal spend* → *Technical validation*
""")
