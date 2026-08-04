"""Qini-aware hyperparameter selection for meta-learner base models (Phase 2).

The meta-learner base XGBoost models (``_DEFAULT_CLF``/``_DEFAULT_REG`` in
``src/models/meta_learners.py``) previously used one fixed, hand-picked
hyperparameter set (400 trees, depth 5) that was never validated against
the metric that actually matters for uplift — Qini AUC — only eyeballed
against logloss/AUC-ROC on the outcome-prediction task. That mismatch
matters more for meta-learners than for a single classifier: T-Learner's
CATE is a *difference* of two independently-fit models (mu1 - mu0), so any
excess variance in either arm's model shows up directly in the estimated
treatment effect, even where the true effect is near zero. An
under-regularized model with more capacity than the noisy, low-signal
outcome (Hillstrom's true ATE is ~0.5pp) can fit noise differently in each
arm, inflating that difference into what looks like heterogeneous CATE but
is actually just estimation noise -- which directly hurts Qini AUC on a
held-out test set.

This module selects hyperparameters by cross-validated Qini AUC directly,
using only the training split (never the held-out test set the final
model is scored on), so the selection is a legitimate model-selection step
and not test-set peeking.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.evaluation.metrics import qini_auc


DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 200, 400],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [20, 50, 100],
}


def _fit_t_learner_fold(X_tr, y_tr, t_tr, params, seed):
    """Fit an arm-split (T-Learner-style) pair of classifiers for one CV fold."""
    mask_1 = t_tr == 1
    mask_0 = t_tr == 0
    clf_kwargs = dict(
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    model_1 = XGBClassifier(**clf_kwargs)
    model_0 = XGBClassifier(**clf_kwargs)
    model_1.fit(X_tr[mask_1], y_tr[mask_1])
    model_0.fit(X_tr[mask_0], y_tr[mask_0])
    return model_1, model_0


def tune_uplift_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    param_grid: dict | None = None,
    n_splits: int = 3,
    seed: int = 42,
    max_combinations: int | None = 27,
) -> dict:
    """
    Select XGBoost hyperparameters for meta-learner base models by
    cross-validated Qini AUC, evaluated with a T-Learner-style fit
    (the cheapest meta-learner to fit repeatedly, and the one whose
    variance is most directly hyperparameter-sensitive).

    Uses ``X``/``y``/``t`` from the TRAINING split only -- callers must not
    pass the held-out test set here, or the final reported Qini AUC on
    that test set would no longer be an honest out-of-sample estimate.

    Parameters
    ----------
    X, y, t       : training features, binary outcome, treatment indicator.
    param_grid    : dict of hyperparameter name -> list of candidate values.
                    Defaults to ``DEFAULT_PARAM_GRID``.
    n_splits      : number of stratified CV folds (stratified on treatment
                    so each fold preserves arm balance).
    max_combinations : if the full grid exceeds this many combinations,
                    randomly subsample down to it (keeps runtime bounded
                    on larger grids). ``None`` disables subsampling.

    Returns
    -------
    dict with ``best_params``, ``best_qini_auc``, and ``results`` (a
    DataFrame of every combination tried, sorted best-first).
    """
    grid = param_grid or DEFAULT_PARAM_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    rng = np.random.RandomState(seed)
    if max_combinations is not None and len(combos) > max_combinations:
        idx = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[i] for i in idx]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []

    for combo in combos:
        params = dict(zip(keys, combo))
        fold_scores = []
        for train_idx, val_idx in skf.split(X, t):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            t_tr, t_val = t[train_idx], t[val_idx]

            model_1, model_0 = _fit_t_learner_fold(X_tr, y_tr, t_tr, params, seed)
            cate_val = (
                model_1.predict_proba(X_val)[:, 1] - model_0.predict_proba(X_val)[:, 1]
            )
            fold_scores.append(qini_auc(y_val, cate_val, t_val))

        rows.append({**params, "qini_auc_cv_mean": float(np.mean(fold_scores)),
                     "qini_auc_cv_std": float(np.std(fold_scores))})

    results = pd.DataFrame(rows).sort_values("qini_auc_cv_mean", ascending=False, ignore_index=True)
    best_row = results.iloc[0]
    # Cast back using the ORIGINAL grid value types (not the DataFrame's
    # inferred dtype, which can upcast an all-int column to float64 once
    # mixed into a row with float columns) so XGBoost gets ints where it
    # expects ints (n_estimators, max_depth, min_child_weight).
    best_params = {
        k: (type(grid[k][0])(best_row[k]))
        for k in keys
    }

    return {
        "best_params": best_params,
        "best_qini_auc": float(best_row["qini_auc_cv_mean"]),
        "results": results,
    }
