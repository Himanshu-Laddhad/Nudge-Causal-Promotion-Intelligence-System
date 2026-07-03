"""
Naive XGBoost baseline (Phase 1).

Causal framing – why this is wrong
------------------------------------
A naïve approach trains a propensity-to-convert model on *all* customers
(treated + control) using treatment as just another feature.  The model
learns P(Y=1 | X, T) — the probability of conversion given treatment.

The fatal flaw: the model ranks customers by *overall conversion probability*,
not by *incremental lift*.  High-baseline customers (people who'd buy
regardless) look attractive to the naive model but are precisely the segment
where promotion is wasted.  These "always buyers" are the sleeping-dog problem.

This module exists to **prove the problem**: show that naive propensity
targeting is strictly worse than uplift-aware targeting on Qini/AUUC, while
appearing deceptively good on AUC-ROC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class NaiveXGBBaseline:
    """
    Trains an XGBoost classifier to predict P(conversion | X, treatment).

    The 'uplift score' is simply the difference in predicted conversion
    probability between treatment=1 and treatment=0 for each customer —
    which is NOT a proper CATE estimator (no cross-fitting, no Neyman
    orthogonality), but mimics what a naive practitioner might do.

    Parameters
    ----------
    xgb_params : dict of XGBoost hyperparameters.
    n_splits    : number of CV folds for OOF score generation.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        xgb_params: dict | None = None,
        n_splits: int = 5,
    ) -> None:
        self.xgb_params = {**self.DEFAULT_PARAMS, **(xgb_params or {})}
        self.n_splits = n_splits
        self.model: XGBClassifier | None = None
        self._feature_names: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> "NaiveXGBBaseline":
        """
        Fit on the full training set (treatment as a feature column).

        Deliberately includes 'treatment' in X so the model sees it —
        that is the naive mistake we are illustrating.
        """
        self._feature_names = list(X.columns)
        X_aug = X.copy()
        X_aug["treatment"] = treatment
        self.model = XGBClassifier(**self.xgb_params)
        self.model.fit(X_aug, y)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute pseudo-uplift as P(Y=1|X,T=1) - P(Y=1|X,T=0).

        This is the *wrong* causal estimator (no cross-fitting, treatment
        not randomised at prediction time), but it is what naive models do.
        """
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        X1 = X.copy(); X1["treatment"] = 1
        X0 = X.copy(); X0["treatment"] = 0
        p1 = self.model.predict_proba(X1)[:, 1]
        p0 = self.model.predict_proba(X0)[:, 1]
        return p1 - p0

    def predict_proba_treated(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(Y=1|X,T=1) — used to show naive AUC looks deceptively good."""
        X1 = X.copy(); X1["treatment"] = 1
        return self.model.predict_proba(X1)[:, 1]

    def oof_uplift(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> np.ndarray:
        """
        Generate out-of-fold uplift scores via StratifiedKFold CV.

        OOF scores prevent optimistic leakage when evaluating on the same
        data used for fitting — important for fair Qini comparisons.
        """
        X_aug = X.copy()
        X_aug["treatment"] = treatment
        oof = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_aug, y), 1):
            clf = XGBClassifier(**self.xgb_params)
            clf.fit(X_aug.iloc[tr_idx], y[tr_idx])
            X_val = X.iloc[val_idx]
            X1 = X_val.copy(); X1["treatment"] = 1
            X0 = X_val.copy(); X0["treatment"] = 0
            oof[val_idx] = clf.predict_proba(X1)[:, 1] - clf.predict_proba(X0)[:, 1]

        return oof

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted descending."""
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        names = self._feature_names + ["treatment"]
        scores = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": names, "importance": scores})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
