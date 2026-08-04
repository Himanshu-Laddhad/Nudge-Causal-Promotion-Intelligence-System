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


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """``n_negative / n_positive`` on a binary label array.

    Standard XGBoost recipe for correcting classifier saturation on rare
    positive rates (e.g. sub-1% conversion), which otherwise
    drives outcome-model probabilities toward zero for nearly every row and
    makes T-/S-/X-Learner CATE differences dominated by numerical noise.
    Falls back to a weight of 1.0 (no reweighting) if there are no positives
    in ``y`` to avoid a divide-by-zero.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return max(n_neg, 1) / n_pos


def _apply_scale_pos_weight(model, y_subset: np.ndarray, mode) -> None:
    """Set ``scale_pos_weight`` on ``model`` in place, if applicable.

    ``mode`` may be:
      - ``None``  : no-op (preserves prior default behaviour exactly).
      - ``"auto"``: compute ``compute_scale_pos_weight(y_subset)`` and apply it
                    (i.e. weight is local to whatever rows this particular
                    model is trained on -- e.g. just the treated arm).
      - a number  : apply that fixed value directly (e.g. a global/pooled
                    approximation computed by the caller).

    No-ops silently if ``model`` doesn't expose a ``scale_pos_weight``
    hyperparameter (e.g. it's a regressor or a non-XGBoost classifier), so
    this is safe to call unconditionally.
    """
    if mode is None or not hasattr(model, "get_params"):
        return
    if "scale_pos_weight" not in model.get_params():
        return
    weight = compute_scale_pos_weight(y_subset) if mode == "auto" else float(mode)
    model.set_params(scale_pos_weight=weight)


class TLearner:
    """
    Two-model CATE estimator: τ̂(x) = μ̂₁(x) − μ̂₀(x).

    Fits independent outcome models on each treatment arm and subtracts.
    Use `outcome_type='binary'` for conversion, `'continuous'` for spend.

    Parameters
    ----------
    scale_pos_weight : None, "auto", or float, default None
        Rare-outcome correction for the underlying XGBClassifier arms
        (ignored for ``outcome_type='continuous'``). ``None`` preserves the
        original behaviour (no reweighting, ``scale_pos_weight=1``).
        ``"auto"`` computes ``n_negative / n_positive`` independently within
        each arm's training rows (recommended for imbalanced treatment
        arms + rare outcomes). A float applies that fixed
        weight to both arms.
    """

    def __init__(
        self,
        outcome_type: str = "binary",
        base_learner=None,
        scale_pos_weight=None,
    ) -> None:
        self.scale_pos_weight = scale_pos_weight
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
        if self._proba:
            _apply_scale_pos_weight(self.model_1, y[mask_1], self.scale_pos_weight)
            _apply_scale_pos_weight(self.model_0, y[mask_0], self.scale_pos_weight)
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

    Parameters
    ----------
    scale_pos_weight : None, "auto", or float, default None
        Rare-outcome correction for the underlying XGBClassifier (ignored
        for ``outcome_type='continuous'``). ``None`` preserves the original
        behaviour. ``"auto"`` computes ``n_negative / n_positive`` on the
        full pooled training set (S-Learner fits a single model across both
        arms, so there is no separate per-arm subset to weight). A float
        applies that fixed weight directly.
    """

    def __init__(
        self,
        outcome_type: str = "binary",
        base_learner=None,
        scale_pos_weight=None,
    ) -> None:
        self.scale_pos_weight = scale_pos_weight
        if outcome_type == "binary":
            self.model = clone(base_learner or _DEFAULT_CLF)
            self._proba = True
        else:
            self.model = clone(base_learner or _DEFAULT_REG)
            self._proba = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, treatment: np.ndarray) -> "SLearner":
        if self._proba:
            _apply_scale_pos_weight(self.model, y, self.scale_pos_weight)
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
    base_learner    : stage-1 outcome model (mu1, mu0). Matches
        ``outcome_type`` -- a classifier for 'binary', a regressor for
        'continuous'.
    pseudo_outcome_learner : stage-3 model (tau1, tau0), default None.
        Pseudo-outcomes (d_tilde_1, d_tilde_0) are always continuous
        residuals regardless of ``outcome_type``, so this must always be a
        regressor -- it is NEVER built from ``base_learner`` even when one
        is supplied (a classifier passed as ``base_learner`` for the
        'binary' case cannot fit continuous residuals). ``None`` defaults
        to ``_DEFAULT_REG``.
    propensity_model: classifier for g(x). Defaults to logistic regression.
    scale_pos_weight : None, "auto", or float, default None
        Rare-outcome correction applied to the stage-1 outcome models
        (mu1, mu0) only -- ignored for `outcome_type='continuous'` and for
        the stage-3 pseudo-outcome regressors (tau1, tau0), which fit
        continuous residuals rather than binary labels. ``None`` preserves
        the original behaviour. ``"auto"`` computes the weight
        independently within each arm's training rows. A float applies
        that fixed weight to both mu1 and mu0.
    """

    def __init__(
        self,
        outcome_type: str = "binary",
        base_learner=None,
        pseudo_outcome_learner=None,
        propensity_model=None,
        scale_pos_weight=None,
    ) -> None:
        self.scale_pos_weight = scale_pos_weight
        if outcome_type == "binary":
            self.mu1 = clone(base_learner or _DEFAULT_CLF)
            self.mu0 = clone(base_learner or _DEFAULT_CLF)
            self._proba = True
        else:
            self.mu1 = clone(base_learner or _DEFAULT_REG)
            self.mu0 = clone(base_learner or _DEFAULT_REG)
            self._proba = False

        self.tau1 = clone(pseudo_outcome_learner or _DEFAULT_REG)
        self.tau0 = clone(pseudo_outcome_learner or _DEFAULT_REG)
        self.propensity = propensity_model or LogisticRegression(
            max_iter=300, random_state=42
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, treatment: np.ndarray) -> "XLearner":
        mask_1 = treatment == 1
        mask_0 = treatment == 0

        # Stage 1: outcome models
        if self._proba:
            _apply_scale_pos_weight(self.mu1, y[mask_1], self.scale_pos_weight)
            _apply_scale_pos_weight(self.mu0, y[mask_0], self.scale_pos_weight)
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
        # Künzel et al. (2019): weight each imputed effect by the propensity of
        # the *opposite* arm, so the estimate leans on whichever arm is better
        # populated in that region of X.
        return g * tau0_pred + (1 - g) * tau1_pred
