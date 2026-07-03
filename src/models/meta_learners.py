"""
Meta-learner wrappers for CATE estimation (Phase 2).

Causal framing
--------------
Meta-learners decompose causal inference into supervised learning sub-tasks,
making them easy to implement with any base learner (XGBoost, LightGBM, etc.).

T-Learner (Two-model)
    Fit μ₁(x) = E[Y|X=x,T=1] and μ₀(x) = E[Y|X=x,T=0] separately.
    τ̂(x) = μ̂₁(x) - μ̂₀(x).
    Weakness: each model is fit on a subset of data; variance is high when
    treatment arms are imbalanced.

S-Learner (Single-model)
    Fit μ(x,t) = E[Y|X=x,T=t] on all data with T as a feature.
    τ̂(x) = μ̂(x,1) - μ̂(x,0).
    Weakness: the model may shrink the treatment effect toward zero by
    treating T as just another low-importance feature.

X-Learner (Cross-learner, Künzel et al. 2019)
    Step 1: fit T-Learner to get μ̂₀, μ̂₁.
    Step 2: impute pseudo-outcomes:
        D̃₁ = Y₁ - μ̂₀(X₁)   (treated residual)
        D̃₀ = μ̂₁(X₀) - Y₀   (control residual)
    Step 3: fit τ̂₁(x) on D̃₁ and τ̂₀(x) on D̃₀.
    Step 4: blend: τ̂(x) = g(x)·τ̂₀(x) + (1-g(x))·τ̂₁(x),
            where g(x) is the propensity score.
    Strength: very efficient when treatment is rare (few treated samples);
    uses control data to inform τ̂ estimation.

All wrappers accept an `econml`-style API (fit / effect) so they can be
swapped into the evaluation harness transparently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor


# ---------------------------------------------------------------------------
# Default base learners
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# T-Learner
# ---------------------------------------------------------------------------


class TLearner:
    """
    Two-model CATE estimator.

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


# ---------------------------------------------------------------------------
# S-Learner
# ---------------------------------------------------------------------------


class SLearner:
    """
    Single-model CATE estimator.

    Includes treatment as a feature.  Susceptibility to treatment shrinkage
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


# ---------------------------------------------------------------------------
# X-Learner
# ---------------------------------------------------------------------------


class XLearner:
    """
    Cross-model CATE estimator (Künzel et al., 2019).

    Particularly powerful when treatment assignment is imbalanced
    (e.g., 90% control / 10% treated), as it borrows from both arms to
    estimate pseudo-outcomes.

    Parameters
    ----------
    outcome_type    : 'binary' or 'continuous'.
    base_learner    : outcome model (stage 1 & 3).
    propensity_model: classifier for g(x).  Defaults to logistic regression.
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

        # Stage 1: fit outcome models
        self.mu1.fit(X[mask_1], y[mask_1])
        self.mu0.fit(X[mask_0], y[mask_0])

        # Stage 2: compute pseudo-outcomes
        if self._proba:
            mu0_on_treated = self.mu0.predict_proba(X[mask_1])[:, 1]
            mu1_on_control = self.mu1.predict_proba(X[mask_0])[:, 1]
        else:
            mu0_on_treated = self.mu0.predict(X[mask_1])
            mu1_on_control = self.mu1.predict(X[mask_0])

        d_tilde_1 = y[mask_1] - mu0_on_treated   # treated pseudo-outcome
        d_tilde_0 = mu1_on_control - y[mask_0]   # control pseudo-outcome

        # Stage 3: fit CATE models on pseudo-outcomes
        self.tau1.fit(X[mask_1], d_tilde_1)
        self.tau0.fit(X[mask_0], d_tilde_0)

        # Stage 4: fit propensity for blending
        self.propensity.fit(X, treatment)

        return self

    def effect(self, X: pd.DataFrame) -> np.ndarray:
        g = self.propensity.predict_proba(X)[:, 1]   # P(T=1|X)
        tau0_pred = self.tau0.predict(X)
        tau1_pred = self.tau1.predict(X)
        # Blend: high-propensity regions weight tau1 more
        return g * tau0_pred + (1 - g) * tau1_pred
