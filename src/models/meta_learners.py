"""Meta-learner wrappers for CATE estimation (Phase 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor


_DEFAULT_CLF = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)

_DEFAULT_REG = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    random_state=42,
    n_jobs=-1,
)


class TLearner:
    """
    Two-model CATE estimator: τ̂(x) = μ̂₁(x) − μ̂₀(x).

    Fits independent outcome models on each treatment arm and subtracts.
    Use `outcome_type='binary'` for conversion, `'continuous'` for spend.
    """

    def __init__(self, outcome_type: str = "binary", base_learner=None) -> None:
        if outcome_type == "binary":
            self.model_1 = clone(base_learner or _DEFAULT_CLF)
            self.model_0 = clone(base_learner or _DEFAULT_CLF)
            self._proba = True
        else:
            self.model_1 = clone(base_learner or _DEFAULT_REG)
            self.model_0 = clone(base_learner or _DEFAULT_REG)
            self._proba = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, treatment: np.ndarray) -> "TLearner":
        mask_1 = treatment == 1
        mask_0 = treatment == 0
        self.model_1.fit(X[mask_1], y[mask_1])
        self.model_0.fit(X[mask_0], y[mask_0])
        return self

    def effect(self, X: pd.DataFrame) -> np.ndarray:
        if self._proba:
            return (
                self.model_1.predict_proba(X)[:, 1]
                - self.model_0.predict_proba(X)[:, 1]
            )
        return self.model_1.predict(X) - self.model_0.predict(X)


class SLearner:
    """
    Single-model CATE estimator: τ̂(x) = μ̂(x,1) − μ̂(x,0).

    Includes treatment as a feature. Susceptibility to treatment shrinkage
    makes this a useful lower-bound benchmark.
    """

    def __init__(self, outcome_type: str = "binary", base_learner=None) -> None:
        if outcome_type == "binary":
            self.model = clone(base_learner or _DEFAULT_CLF)
            self._proba = True
        else:
            self.model = clone(base_learner or _DEFAULT_REG)
            self._proba = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, treatment: np.ndarray) -> "SLearner":
        X_aug = X.copy()
        X_aug["treatment"] = treatment
        self.model.fit(X_aug, y)
        return self

    def effect(self, X: pd.DataFrame) -> np.ndarray:
        X1 = X.copy(); X1["treatment"] = 1
        X0 = X.copy(); X0["treatment"] = 0
        if self._proba:
            return (
                self.model.predict_proba(X1)[:, 1]
                - self.model.predict_proba(X0)[:, 1]
            )
        return self.model.predict(X1) - self.model.predict(X0)


class XLearner:
    """
    Cross-model CATE estimator (Künzel et al., 2019).

    Particularly effective when treatment assignment is imbalanced.
    Imputes counterfactual pseudo-outcomes for each arm and blends
    via propensity score: τ̂(x) = g(x)·τ̂₀(x) + (1−g(x))·τ̂₁(x).

    Parameters
    ----------
    outcome_type    : 'binary' or 'continuous'.
    base_learner    : outcome model (stages 1 & 3).
    propensity_model: classifier for g(x). Defaults to logistic regression.
    """

    def __init__(
        self,
        outcome_type: str = "binary",
        base_learner=None,
        propensity_model=None,
    ) -> None:
        if outcome_type == "binary":
            self.mu1 = clone(base_learner or _DEFAULT_CLF)
            self.mu0 = clone(base_learner or _DEFAULT_CLF)
            self._proba = True
        else:
            self.mu1 = clone(base_learner or _DEFAULT_REG)
            self.mu0 = clone(base_learner or _DEFAULT_REG)
            self._proba = False

        self.tau1 = clone(base_learner or _DEFAULT_REG)
        self.tau0 = clone(base_learner or _DEFAULT_REG)
        self.propensity = propensity_model or LogisticRegression(
            max_iter=300, random_state=42
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, treatment: np.ndarray) -> "XLearner":
        mask_1 = treatment == 1
        mask_0 = treatment == 0

        # Stage 1: outcome models
        self.mu1.fit(X[mask_1], y[mask_1])
        self.mu0.fit(X[mask_0], y[mask_0])

        # Stage 2: pseudo-outcomes
        if self._proba:
            mu0_on_treated = self.mu0.predict_proba(X[mask_1])[:, 1]
            mu1_on_control = self.mu1.predict_proba(X[mask_0])[:, 1]
        else:
            mu0_on_treated = self.mu0.predict(X[mask_1])
            mu1_on_control = self.mu1.predict(X[mask_0])

        d_tilde_1 = y[mask_1] - mu0_on_treated   # treated pseudo-outcome
        d_tilde_0 = mu1_on_control - y[mask_0]   # control pseudo-outcome

        # Stage 3: CATE models on pseudo-outcomes
        self.tau1.fit(X[mask_1], d_tilde_1)
        self.tau0.fit(X[mask_0], d_tilde_0)

        # Stage 4: propensity for blending
        self.propensity.fit(X, treatment)

        return self

    def effect(self, X: pd.DataFrame) -> np.ndarray:
        g = self.propensity.predict_proba(X)[:, 1]   # P(T=1|X)
        tau0_pred = self.tau0.predict(X)
        tau1_pred = self.tau1.predict(X)
        # High-propensity regions weight tau1 more
        return g * tau0_pred + (1 - g) * tau1_pred
