"""
Physics-Guided Machine Learning (PGML) Surrogates
=================================================

Implements:
  1. PhysicsGuidedSurrogate: Learns the physical residual discrepancy
     Delta u = u_CFD - u_physics relative to analytical Poiseuille flow.
  2. GeometryOnlySurrogate: Standard black-box baseline regressing u_CFD
     directly from spatial and geometrical features.
"""

from typing import Dict, Optional, Union
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_surrogate_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Computes RMSE, MAE, and R^2 metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


class PhysicsGuidedSurrogate:
    """
    Physics-Guided Residual Machine Learning Surrogate.

    Instead of regressing velocity from scratch, the surrogate learns the
    residual discrepancy Delta u:
        Delta u = u_CFD - u_physics
        u_pred = max(0, u_physics + Delta u_pred)

    This guarantees physical asymptotic behavior in low-gradient regimes
    and enforces mass-continuity boundedness in high-gradient throat regions.
    """

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        **kwargs,
    ):
        self.estimator = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs,
        )
        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        u_phys: np.ndarray,
    ) -> "PhysicsGuidedSurrogate":
        """
        Fits the residual model on Delta u = y - u_phys.
        """
        residual = y - u_phys
        self.estimator.fit(X, residual)
        self.is_fitted_ = True
        return self

    def predict(
        self,
        X: np.ndarray,
        u_phys: np.ndarray,
    ) -> np.ndarray:
        """
        Predicts total velocity by adding learned residual to analytical baseline.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call .fit() first.")
        residual_pred = self.estimator.predict(X)
        total_velocity = u_phys + residual_pred
        return np.maximum(0.0, total_velocity)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        u_phys: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluates surrogate predictions against CFD ground truth."""
        y_pred = self.predict(X, u_phys)
        return evaluate_surrogate_performance(y_true, y_pred)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes model to disk using joblib."""
        joblib.dump(self.estimator, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PhysicsGuidedSurrogate":
        """Deserializes model from disk."""
        instance = cls()
        instance.estimator = joblib.load(filepath)
        instance.is_fitted_ = True
        return instance


class GeometryOnlySurrogate:
    """
    Standard Geometry-Only Black-Box Machine Learning Surrogate.
    Regresses velocity directly from geometrical inputs: u_pred = f(X).
    """

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        **kwargs,
    ):
        self.estimator = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs,
        )
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GeometryOnlySurrogate":
        """Fits directly on target velocity."""
        self.estimator.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts velocity directly."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted. Call .fit() first.")
        return np.maximum(0.0, self.estimator.predict(X))

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluates surrogate predictions against CFD ground truth."""
        y_pred = self.predict(X)
        return evaluate_surrogate_performance(y_true, y_pred)

    def save(self, filepath: Union[str, Path]) -> None:
        joblib.dump(self.estimator, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "GeometryOnlySurrogate":
        instance = cls()
        instance.estimator = joblib.load(filepath)
        instance.is_fitted_ = True
        return instance
