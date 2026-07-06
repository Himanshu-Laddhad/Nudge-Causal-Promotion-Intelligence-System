"""
Naive XGBoost propensity baseline (Phase 1).

Trains P(conversion | X, T) with treatment as a feature — the naive mistake.
Used as a benchmark to demonstrate that propensity ranking conflates loyalty
with incremental lift, resulting in systematic targeting of sleeping dogs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class NaiveXGBBaseline:
    """
    XGBoost classifier predicting P(conversion | X, treatment).

    The uplift score is P(Y=1|X,T=1) - P(Y=1|X,T=0) — not a proper CATE
    estimator (no cross-fitting, no Neyman orthogonality), but representative
    of what naive practitioners deploy.

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
        that is the naive mistake this class illustrates.
        """
        self._feature_names = list(X.columns)
        X_aug = X.copy()
        X_aug["treatment"] = treatment
        self.model = XGBClassifier(**self.xgb_params)
        self.model.fit(X_aug, y)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Pseudo-uplift: P(Y=1|X,T=1) - P(Y=1|X,T=0).

        Not a valid CATE estimator — no cross-fitting, no orthogonality.
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
        Out-of-fold uplift scores via StratifiedKFold CV.

        OOF prevents optimistic leakage when evaluating on training data.
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
