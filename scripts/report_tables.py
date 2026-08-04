"""
Print the README result tables straight from outputs/.

The README's numbers previously drifted from the CSVs that produced them. Run
this after the pipeline and paste the output, rather than editing the tables by
hand.

Usage:
    python scripts/report_tables.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
OUTPUTS = ROOT / 'outputs'


def section(title: str) -> None:
    print(f'\n### {title}\n')


def main() -> None:
    master = pd.read_csv(OUTPUTS / 'master_comparison_table.csv')
    master = master.sort_values('qini_auc', ascending=False)
    section('Model Progression')
    print('| Model | Qini AUC | Sleeping Dogs in Top-20% | Positive CATE |')
    print('|---|---|---|---|')
    for _, r in master.iterrows():
        print(f"| {r['model']} | {r['qini_auc']:+.4f} | "
              f"{r['pct_sleeping_dogs_topk']:.2f}% | {r['pct_positive_cate']:.1f}% |")

    rob = pd.read_csv(OUTPUTS / 'robustness_experiment_results.csv')
    section('Robustness to Confounding')
    print('| Model | Qini (Clean RCT) | Qini (Confounded) |')
    print('|---|---|---|')
    for _, r in rob.iterrows():
        print(f"| {r['Model']} | {r['Qini_Clean']:+.4f} | {r['Qini_Confounded']:+.4f} |")

    budget = pd.read_csv(OUTPUTS / 'budget_optimizer_results.csv')
    section('Budget Optimization')
    print('| Budget | Targeted | CATE-optimal lift | Naive lift | Advantage |')
    print('|---|---|---|---|---|')
    for _, r in budget.iterrows():
        print(f"| ${r['budget']:,.0f} | {r['n_targeted']:,.0f} ({r['pct_population']:.1f}%) | "
              f"+{r['expected_lift']:.1f} | +{r['naive_lift']:.1f} | "
              f"{r['lift_vs_naive']:+.1f} |")

    be_path = OUTPUTS / 'phase5a_breakeven.csv'
    if be_path.exists():
        be = pd.read_csv(be_path)
        cate_col = next(c for c in be.columns
                        if c.startswith('break_even_cost_') and c != 'break_even_cost_naive')
        model = cate_col.replace('break_even_cost_', '')
        section('Break-Even Promotion Cost')
        print(f'| Budget | {model} | Naive XGB | Headroom |')
        print('|---|---|---|---|')
        for _, r in be.iterrows():
            print(f"| ${r['budget']:,.0f} | ${r[cate_col]:.2f} | "
                  f"${r['break_even_cost_naive']:.2f} | {r['headroom_vs_naive_pct']:+.1f}% |")


if __name__ == '__main__':
    main()
