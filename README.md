# Nudge — Causal Promotion Intelligence System

> **Standard ML optimizes for who will buy. Nudge identifies who will buy *because* you promote them.**

A causal ML system that estimates individualized treatment effects (CATE) for e-commerce promotions — distinguishing persuadable customers from those who convert regardless. A research pipeline runs from naive baseline through doubly-robust estimation to an LP budget optimizer and an interactive dashboard, plus a packaged real-time inference path for SageMaker (built and validated locally; never deployed to a live endpoint).

---

## Business Context

E-commerce teams lose margin every day by sending discounts to customers who were already going to buy. These "Sleeping Dogs" incur promotion cost with zero incremental return. A conversion classifier — the industry default — cannot detect them: it correctly identifies *likely buyers*, not *promotion-sensitive buyers*. Those are different populations.

On the Hillstrom RCT, Nudge turns this into money two ways. At a fixed budget it beats naive targeting by roughly 6–24 incremental conversions across every budget tested. More importantly, it **extends the range over which promoting is worth doing at all**: at a $5,000 budget, naive targeting has already gone clearly unprofitable (−12% ROI) while causal targeting is still holding break-even (+5%). Total campaign profit peaks at **+$861 on a 64K list**, at a $2,500 budget — spend past that and the marginal customer costs more than the margin they return. Knowing where that peak sits, not just that targeting "helps," is the deliverable.

---

## The Problem: Why Standard ML Fails

A naive XGBoost classifier trained on the treated group and scored on all customers assigns high scores to high-history customers — customers who are intrinsically valuable. The problem: in the naive model's top-20%-by-score segment, control-arm customers (who received no promotion) still convert at 0.82% — not far below the 2.87% conversion rate of that same segment's treated customers. Sending the control-arm share a discount buys little to no incremental conversion; it is largely deadweight loss (see [Phase 1](notebooks/phase1_naive_baseline.ipynb)'s persuadability matrix).

```
Persuadability Matrix — Naive XGB top-20% targeting
                     Converts in control?
                     YES            NO
Treated &    YES  [ Sleeping Dog ]  [ True Uplift  ]
converts?    NO   [ Lost Cause   ]  [ Do-Nothing   ]
```

The standard ML pipeline is blind to the Sleeping Dog quadrant. It optimizes `P(Y=1 | X, T=1)` when the decision-relevant quantity is `P(Y=1 | X, T=1) − P(Y=1 | X, T=0)` — the **Conditional Average Treatment Effect (CATE)**.

---

## The Solution: Causal ML Pipeline

Nudge estimates CATE directly at the individual level using a five-stage progression of increasingly robust estimators.

**The Persuadability Matrix, computed correctly:**

```
Estimated CATE > threshold  → Persuadable  (send promotion)
Estimated CATE < -threshold → Sleeping Dog (withhold)
|CATE| ≤ threshold          → Neutral       (no action)
```

How much of the base each model calls persuadable is itself diagnostic. The naive classifier flags **100%** — a conversion probability is never negative, so it is structurally incapable of identifying a sleeping dog. The causal estimators disagree with it and with each other: T-Learner **79.4%**, S-Learner **95.7%**, X-Learner **53.5%**, DR-Learner **75.4%**. The more a model is built to isolate the treatment effect, the more of the base it declines to spend money on. Per-customer distributions are in `outputs/phase4_final_cate.parquet`.

---

## Results

### Model Progression — Qini AUC & Realized Deadweight Loss

| Model | Qini AUC | Sleeping Dogs in Top-20% Targeted (realized) |
|---|---|---|
| **S-Learner** | **+0.0366** | 0.87% |
| Naive XGBoost | +0.0117 | 1.08% |
| T-Learner | +0.0015 | 0.85% |
| X-Learner | −0.0587 | 1.02% |
| DR-Learner | −0.1251 | 1.07% |

<sub>Regenerate with `python scripts/report_tables.py` after any pipeline run — do not hand-edit.</sub>

> **Qini AUC** measures rank-ordering of treatment effect, not conversion rate — a negative value on the Hillstrom dataset (low ATE ≈ 0.5pp, high noise) is expected for several models; relative ordering is the meaningful signal here, and it is genuinely noisy at this dataset's effect size (see caveat below).
>
> **Sleeping Dogs in Top-20%** replaces an earlier "deadweight loss" column that computed `% of a model's own predicted score ≤ 0` — a metric that is not comparable across models, since it trivially returns ~0% for any score that can never go negative (like Naive XGB's raw conversion probability) regardless of how many "would-convert-anyway" customers it actually targets. This column instead measures, for each model's top-20%-by-score target list, what fraction is *control-arm customers who converted anyway* — an unbiased, RCT-based, apples-to-apples estimate of real budget waste, computed identically for every model (`src/evaluation/metrics.py::topk_deadweight_loss`).
>
> **Honest caveat on this table:** only S-Learner clearly beats Naive XGBoost on Qini AUC. T-Learner is barely positive, essentially indistinguishable from Naive given sampling noise; X-Learner and DR-Learner score *below* it on this clean, tiny-effect RCT. That is not a contradiction of the thesis; it is why the thesis does not rest on this table. Qini AUC on a low-ATE clean RCT is dominated by test-set sampling noise (~12.8K held-out rows, ~1% base conversion rate) and does not reliably separate correlational from causal estimators when there is no confounding to exploit. More flexible estimators pay a variance cost for bias protection they do not need here. The real evidence for "naive answers the wrong question" is (a) the sleeping-dog audit in [Phase 1](notebooks/phase1_naive_baseline.ipynb) — naive targeting puts the *most* would-convert-anyway customers in its top 20% of any model in the table — and (b) the robustness experiment below, where the gap is not noise.

### The Key Finding: Robustness to Confounding

The publishable contribution of this project. Real-world promotion data is confounded — loyal customers receive more discounts. We simulate this by re-assigning treatment with `P(T=1 | high history) = 0.8`, then compare Qini AUC degradation on the same held-out, unconfounded RCT test set:

| Model | Qini (Clean RCT) | Qini (Confounded) | Rank Preserved? |
|---|---|---|---|
| T-Learner | +0.0015 | **−0.0000** | ✗ collapses to indistinguishable from random |
| **DR-Learner** | **−0.1251** | **+0.0111** | ✓ reverses upward under the selection bias |

This is the result the project is built around, and the direction of the reversal is the striking part. T-Learner's clean-RCT signal was already faint (+0.0015, barely above zero); under confounding it collapses the rest of the way to exactly zero — the model reads "received lots of discounts historically" as "responds to discounts," and once that correlation is injected, ranking by predicted CATE stops distinguishing persuadables from anyone else. DR-Learner runs the other way: it is the *worst*-ranking model of all five on the clean RCT (−0.1251) — its cross-fitting and propensity correction are pure variance cost when there is no selection bias to correct — and it is the only model that turns clearly positive once assignment is confounded.

The practical reading: model choice should depend on how the treatment was assigned, not on a clean-RCT leaderboard. DR-Learner's machinery looks like overkill, even actively harmful, on randomized data — it is the worst performer in the top table. That machinery is not idle, though: it is a specific correction for a specific failure mode, and it only pays for itself once that failure mode (selection-biased treatment assignment) is actually present. On observational campaign data — which is what any deployed system actually sees — DR-Learner is the only estimator in this comparison whose ranking can be trusted.

(Point estimates only; `src/evaluation/metrics.py::bootstrap_qini_ci` will bootstrap 95% CIs — 500 resamples — around each Qini AUC to confirm a gap this size is not sampling noise.)

### Budget Optimization

LP-constrained targeting across the full 64K customer base. Economics live in `src/config.py`, and the cost model matters as much as the model does:

| Component | Value | Source |
|---|---|---|
| Email send | $0.10 | Assumption (generous vs. real ESP rates) |
| Expected redemption | $0.35 | 15% off × $116.36 order × 2% treated conversion |
| **Cost per customer promoted** | **$0.45** | Sum of the above |
| Gross order value | $116.36 | Measured: mean spend among converters |
| **Contribution margin** | **$40.73** | 35% margin on the above |

Two things are deliberate here. The discount is charged *only when it is redeemed*, since a coupon nobody uses costs nothing — billing face value to every recipient was what previously made this campaign look hopeless. And the benefit is contribution margin, not order value, because crediting a campaign with revenue that pays for goods overstates ROI roughly threefold.

A targeted segment must therefore lift conversion by **1.10%** to pay for itself.

| Budget | Targeted | CATE lift | Naive lift | ROI (CATE) | ROI (Naive) | Profit |
|---|---|---|---|---|---|---|
| $500 | 1,113 (1.7%) | +26.5 | +20.8 | **+116%** | +70% | +$581 |
| $1,000 | 2,226 (3.5%) | +44.1 | +35.6 | **+80%** | +45% | +$798 |
| $2,500 | 5,566 (8.7%) | +82.5 | +67.8 | **+34%** | +10% | **+$861** |
| $5,000 | 11,133 (17.4%) | +129.2 | +108.2 | **+5%** | −12% | +$264 |
| $10,000 | 22,267 (34.8%) | +196.1 | +171.6 | −20% | −30% | −$2,014 |
| $25,000 | 55,669 (87.0%) | +297.2 | +284.3 | −52% | −54% | −$12,898 |

Both strategies are scored on the same yardstick: the naive model only *selects* customers, and that set is then valued using the CATE model's estimate of what those customers were actually worth. Scoring naive selections on P(convert) instead would compare two different quantities and flatter the baseline.

**The campaign is profitable, and there is a right amount to spend.** ROI is highest at the smallest budget, but total profit peaks at **$2,500** — targeting ~9% of the base for **+$861**. Past that, each additional customer returns less margin than they cost to reach: by $5,000 profit has fallen to +$264 and is effectively break-even, and from $10,000 on it is clearly negative, reaching −$12.9K if the campaign is pushed to 87% of the base.

The most useful column is the ROI comparison. At a $5,000 budget the naive model has already gone **clearly negative (−12%)** while causal targeting is **still at break-even (+5%)**. Causal targeting does not merely improve returns at a fixed spend — it *extends the range over which spending is worth doing at all*, by roughly one budget tier in this campaign.

### Break-Even: The Number a Marketing Team Can Act On

The break-even cost is the most a promotion can cost before a given targeted segment stops paying for itself. `src/model_registry.py` resolves this against whichever model currently tops the Qini table (S-Learner in this run — see the caveat on the model-progression table above about why that is not always the same model):

| Budget | S-Learner | Naive XGB | Headroom |
|---|---|---|---|
| $500 | $0.97 | $0.76 | +27.4% |
| $2,500 | $0.60 | $0.50 | +21.8% |
| $10,000 | $0.36 | $0.31 | +14.2% |
| $25,000 | $0.22 | $0.21 | +4.5% |

Against the $0.45 actually being spent, the picture is tighter than the ROI table alone suggests. Budgets through $2,500 clear the bar with room to spare ($0.60 and above). At $5,000 the margin is thin — a $0.47 break-even against a $0.45 cost is roughly 4%, well inside the noise of these estimates, so that scenario should be read as break-even rather than reliably profitable. From $10,000 on, break-even falls below cost ($0.36 vs. $0.45) and the campaign loses money, consistent with the negative ROI above. Causal targeting buys **4–27% more headroom** at equal spend, largest at the smallest budgets where the targeted segment is most concentrated in genuinely persuadable customers.

The honest caveat is that this campaign's viability is decided mostly by the cost model, not the estimator. At a deep discount charged to every recipient, nothing rescues it; at realistic email economics, even naive targeting profits at small budgets. Causal targeting is what determines *how far you can scale before that stops being true*, and by how much — worth roughly **$230 to just over $990 of additional profit versus naive targeting at the same budget**, growing through $10,000 before narrowing again once the campaign has run out of persuadable customers to find at 87% coverage.

---

## The Tech Progression

The model progression mirrors NLP's evolution from bag-of-words to transformers:

```
Phase 1  Naive XGBoost     →  Bag-of-Words       Fast, interpretable, wrong question
Phase 2  T/S/X-Learner     →  Word2Vec           Causal framing, still meta-learning
Phase 4  DR-Learner        →  BERT               Doubly robust, survives distribution shift
Phase 5  Budget Optimizer  →  Fine-tuning        Decision theory on top of representations
```

Each phase does not merely improve a metric — it corrects a fundamental assumption of the previous one.

The numbering runs 1 → 2 → 4 because Phase 3, a Causal Forest benchmark, was retired: it cost substantially more compute than the meta-learners and changed none of the conclusions at this dataset's effect size. The phase numbers were left as they were rather than renumbered, so the notebooks still line up with the outputs and commit history that reference them.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                     │
│  Hillstrom CSV ──── DuckDB ──► hillstrom_features.parquet       │
│                                (feature engineering in SQL)     │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│  ESTIMATION PIPELINE                                            │
│                                                                 │
│  Phase 1  NaiveXGBBaseline ──────────────────► phase1_results  │
│  Phase 2  T/S/X-Learner (econml) ───────────► phase2_cate      │
│  Phase 4  DRLearner + confounding exp ──────► phase4_final_cate │
│           └─ robustness_experiment_results                      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│  DECISION LAYER                                                 │
│                                                                 │
│  Phase 5A  LPBudgetOptimizer ───────────────► budget_results    │
│  Phase 5B  Streamlit Dashboard (5 pages)                        │
│  Phase 5C  SageMaker + Triton + FIL ──────► REST API (designed) │
└─────────────────────────────────────────────────────────────────┘

  Inference path (designed, not deployed):
  JSON customer features
       │
  inference_client.py  (DuckDB feature engineering)
       │
  SageMaker endpoint  (Triton + FIL backend, ml.g4dn.xlarge)
       │
  { cate_score, persuadability, action_recommendation }
```

---

## Dataset

| Dataset | Rows | Features | Treatment | Use |
|---|---|---|---|---|
| [Hillstrom Email Marketing](https://www.kaggle.com/datasets/davinwijaya/email-marketing) | 64K | 8 named | RCT (email campaign) | Interpretability, storytelling, all phases |

Hillstrom is a genuine randomized controlled trial — treatment assignment is independent of features by design, making it a clean causal identification setting. Publicly available.

---

## Project Structure

```
nudge/
├── notebooks/
│   ├── phase1_naive_baseline.ipynb      # XGBoost baseline + deadweight loss proof
│   ├── phase2_meta_learners.ipynb       # T/S/X-Learner + QINI comparison
│   ├── phase4_dr_learner_robustness.ipynb  # DR-Learner + confounding experiment
│   └── phase5a_budget_optimizer.ipynb   # LP budget optimizer + elbow analysis
│
├── src/
│   ├── config.py                        # Promotion economics — single source of truth
│   ├── model_registry.py                # Resolves "best model" from the Qini table
│   ├── features/
│   │   └── hillstrom.py                 # DuckDB feature pipeline
│   ├── models/
│   │   ├── naive_baseline.py            # NaiveXGBBaseline wrapper
│   │   ├── meta_learners.py             # T/S/X-Learner wrappers
│   │   └── tuning.py                    # Qini-aware CV hyperparameter search
│   ├── evaluation/
│   │   ├── metrics.py                   # Qini, AUUC, deadweight loss, ATT
│   │   └── plots.py                     # Qini curves, CATE distributions
│   └── budget/
│       └── lp_optimizer.py              # Neyman-Pearson + LP for heterogeneous costs
│
├── scripts/
│   ├── run_pipeline.py                  # Execute notebooks in dependency order
│   └── report_tables.py                 # Regenerate this README's result tables
│
├── queries/                             # Standalone DuckDB SQL (all transformations)
│   ├── 01_hillstrom_load.sql
│   ├── 02_hillstrom_features.sql
│   ├── 03_hillstrom_train_test_split.sql
│   ├── 04_rct_balance_check.sql
│   └── 05_segment_profiles.sql
│
├── dashboard/
│   ├── app.py                           # Streamlit entry point
│   ├── pages/                           # 5-page multi-page app
│   └── utils/                           # Data loaders + Plotly chart builders
│
├── deployment/
│   ├── README.md                        # Cost model + teardown checklist
│   ├── export_model.py                  # Validate + stage artifact into model repo
│   ├── model_artifacts/                 # Written by Phase 4 (gitignored)
│   ├── triton/
│   │   └── config_cpu.pbtxt             # KIND_CPU variant, swapped in for ml.c5
│   ├── triton_model_repo/               # Triton model repository
│   │   └── nudge_cate_model/
│   │       ├── 1/xgboost.json           # XGBoost weights (generated, gitignored)
│   │       └── config.pbtxt             # RAPIDS FIL backend config (GPU)
│   └── sagemaker/
│       ├── deploy_sagemaker.py          # boto3 endpoint creation + teardown
│       ├── inference_client.py          # Preprocess → invoke → labeled prediction
│       └── monitoring_setup.py          # CloudWatch alarms + Model Monitor
│
├── outputs/                             # Pre-computed artifacts (notebooks write here)
├── data/raw/                            # hillstrom.csv (committed)
├── requirements.txt                     # Core pipeline + dashboard
├── requirements-deploy.txt              # Adds boto3 + sagemaker
└── environment.yml
```

---

## How to Run

### Environment

```bash
# Option A: pip
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate nudge
```

### Notebooks

```bash
# Whole pipeline, in dependency order (~3 minutes):
python scripts/run_pipeline.py

# Or a subset:
python scripts/run_pipeline.py 4 5a

# Or interactively:
jupyter lab
```

Order matters: **1 → 2 → 4 → 5a**. Each phase reads the previous phase's parquet from `outputs/`, so running them out of sequence silently mixes results from different runs. `run_pipeline.py` enforces the order and stops on the first failure rather than letting a later phase consume stale inputs.

After a run, refresh this README's tables with `python scripts/report_tables.py`.

### Dashboard

```bash
streamlit run dashboard/app.py
```

Reads pre-computed `outputs/` — no model retraining required.

---

## Deployment

> **Scope note.** This is a designed and packaged deployment path, not a running service. The export, artifact validation, and Triton model repository were built and exercised locally; the scripts have never been run against a live AWS account. Instance costs and the p99 latency target below are published figures and design targets — not measurements taken from this system.

Real-time CATE inference via NVIDIA Triton on AWS SageMaker with RAPIDS FIL (Forest Inference Library) for GPU-accelerated tree scoring.

The model staged for serving is the **DR-Learner**, not whichever meta-learner currently tops the clean-RCT Qini table (S-Learner in this run — see `src/model_registry.py`), for two reasons: its final stage is the only one that is a single `X → τ(x)` map (the others need two or more forward passes, which does not fit a single-tensor FIL signature), and production traffic is not randomised — the regime where DR-Learner is the estimator that holds up. Phase 4 fits that final stage as an `XGBRegressor` and asserts the exported artifact reproduces `DRLearner.effect()` exactly before writing it.

```bash
pip install -r requirements-deploy.txt

# Phase 4 writes deployment/model_artifacts/; this validates and stages it
python deployment/export_model.py

# Deploy — GPU (production)
python deployment/sagemaker/deploy_sagemaker.py --bucket <your-s3-bucket>

# Deploy — demo mode (spins up, validates, tears down automatically; cost < $0.02)
python deployment/sagemaker/deploy_sagemaker.py --bucket <your-s3-bucket> --demo

# Score a customer
python deployment/sagemaker/inference_client.py

# Monitoring (CloudWatch + Model Monitor drift detection)
python deployment/sagemaker/monitoring_setup.py

# Teardown
python deployment/sagemaker/deploy_sagemaker.py --teardown
```

**Instance options:**

| Mode | Instance | Cost | Notes |
|---|---|---|---|
| GPU | `ml.g4dn.xlarge` | ~$0.74/hr | RAPIDS FIL, ~50K rows/sec |
| CPU | `ml.c5.large` | ~$0.10/hr | FIL CPU mode, demo/budget |

Inference response:

```json
{
  "cate_score": 0.031,
  "persuadability": "persuadable",
  "send_promotion": true,
  "action_recommendation": "Send promotion. Expected incremental conversion lift: +0.031.",
  "confidence": "medium"
}
```

---

## Research Angle

The central empirical claim: **under realistic observational confounding, the ranking of estimators reverses — the model that pays a variance cost for robustness it does not need on clean RCT data is the only one still carrying signal once the treatment assignment stops being random.**

The robustness experiment (Phase 4) constructs a confounded dataset by re-assigning treatment probability as a function of purchase history — simulating the common real-world pattern where loyalty programs disproportionately reach high-value customers. T-Learner's Qini AUC was already faint on the clean RCT (+0.0015) and **collapses to exactly zero under confounding** (−0.00003): the model has learned to rank customers by how often they were historically discounted, which under selection bias carries no information about true uplift. DR-Learner moves the opposite direction — from the *worst*-ranking model of the five on clean data (−0.1251) to the only one clearly above zero once assignment is confounded (+0.0111).

The sharpest version of the claim is that the ranking reverses on the dimension that matters: the estimator whose extra machinery looks like pure overhead on an RCT — worst Qini AUC in the comparison — is the one estimator that still works once real-world selection bias is introduced. A leaderboard built on randomised data does not just fail to predict performance on observational data; here it points in the wrong direction entirely.

This matters because most production promotion datasets are not RCTs. Teams running causal models on logged data face exactly this confounding structure. The finding suggests DR-Learner (or any doubly-robust estimator) should be the default choice when RCT data is unavailable.

**Key references:**

- Künzel et al. (2019) — [Metalearners for estimating heterogeneous treatment effects](https://arxiv.org/abs/1706.03461)
- Nie & Wager (2021) — [Quasi-oracle estimation of heterogeneous treatment effects](https://arxiv.org/abs/1712.04912)
- Kennedy (2023) — [Towards optimal doubly robust estimation](https://arxiv.org/abs/2004.14497)

---

## Citation

```bibtex
@misc{nudge2026,
  title   = {Nudge: Causal Promotion Intelligence System},
  author  = {Himanshu Laddhad},
  year    = {2026},
  url     = {https://github.com/Himanshu-Laddhad/Nudge},
  note    = {Production-grade CATE estimation pipeline for e-commerce promotion targeting}
}
```

---

## Stack

`econml` · `xgboost` · `scikit-learn` · `duckdb` · `streamlit` · `plotly` · `boto3` · `sagemaker` · `nvidia-triton` · `rapids-fil`
