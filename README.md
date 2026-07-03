# Nudge

> Identify who is genuinely persuaded by a promotion — and stop wasting budget on customers who would have bought anyway.

## What it does

Nudge estimates individual-level causal treatment effects (CATE) for a promotional email campaign using the Hillstrom Email Marketing RCT. Standard propensity models rank customers by P(buy | X), which systematically targets loyal customers who convert regardless of the promotion — pure deadweight loss. Nudge replaces that with τ̂(x) = E[Y(1) − Y(0) | X = x], then allocates a fixed budget to the top persuadable segment. A progression of five estimators (Naive XGB → T/S/X-Learner → Causal Forest → DR-Learner) demonstrates measurable Qini AUC improvements at each stage, with the DR-Learner achieving the lowest deadweight loss and highest robustness to confounding.

## Stack

- **Python 3.11**, XGBoost, LightGBM, scikit-learn
- **EconML** — CausalForestDML, LinearDRLearner
- **DuckDB** — all feature engineering and SQL analytics
- **Streamlit** — interactive results dashboard
- **scipy** — LP-based budget optimiser

## Quickstart

```bash
# conda (recommended)
conda env create -f environment.yml
conda activate nudge

# or pip
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Download data:

```bash
# Hillstrom (~3 MB)
curl -L "https://raw.githubusercontent.com/mshenfield/hillstrom-email-marketing/master/data/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv" \
     -o data/raw/hillstrom.csv

# Criteo Uplift v2 (~2 GB compressed) — optional, Phase 3+
# https://criteo-uplift-dataset.s3-us-west-2.amazonaws.com/criteo-uplift-v2.1.csv.gz
```

Run notebooks in order, then launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Project structure

```
nudge/
├── data/raw/               # hillstrom.csv, criteo-uplift-v2.csv.gz
├── notebooks/
│   ├── phase1_naive_baseline.ipynb
│   ├── phase2_meta_learners.ipynb
│   ├── phase3_causal_forest.ipynb
│   ├── phase4_dr_learner_robustness.ipynb
│   └── phase5a_budget_optimizer.ipynb
├── src/
│   ├── features/           # DuckDB feature engineering (hillstrom, criteo)
│   ├── models/             # NaiveXGBBaseline, TLearner, SLearner, XLearner
│   ├── evaluation/         # Qini, AUUC, deadweight loss, plots
│   └── budget/             # BudgetOptimizer, LPBudgetOptimizer
├── queries/                # Standalone DuckDB SQL files
├── outputs/                # Pre-computed parquets/CSVs for dashboard
├── dashboard/              # Streamlit multi-page app
├── requirements.txt
└── environment.yml
```

## Dataset

**Hillstrom Email Marketing (2008)** — 64,000 customers, genuine 3-arm RCT (no email / men's email / women's email). Named features: recency, history, channel, zip_code, newbie, mens/womens flags. Primary outcome: 2-week conversion. Source: [MineThatData blog](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html).

**Criteo Uplift v2** — 14M-row academic benchmark (Diemert et al., 2021). Binary treatment, 12 anonymised features. Used for out-of-distribution validation in Phase 3.

---

References: Künzel et al. (2019) · Wager & Athey (2018) · Chernozhukov et al. (2018) · Diemert et al. (2021) · Radcliffe (2007)
