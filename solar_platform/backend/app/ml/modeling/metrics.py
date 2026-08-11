"""
app.ml.modeling.metrics
-----------------------
Evaluation Metrics for Scientific Curve Surrogates and Scalar Prediction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CurveEvaluationMetrics:
    """Evaluation metrics for curve prediction tasks."""
    mae: float
    rmse: float
    r2: float
    nrmse: float
    max_absolute_error: float
    sample_count: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "mae": round(self.mae, 6),
            "rmse": round(self.rmse, 6),
            "r2": round(self.r2, 6),
            "nrmse": round(self.nrmse, 6),
            "max_absolute_error": round(self.max_absolute_error, 6),
            "sample_count": self.sample_count,
        }


@dataclass
class ScalarEvaluationMetrics:
    """Evaluation metrics for scalar metrics prediction (Jsc, Voc, FF, PCE)."""
    mae: float
    rmse: float
    r2: float
    sample_count: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "mae": round(self.mae, 6),
            "rmse": round(self.rmse, 6),
            "r2": round(self.r2, 6),
            "sample_count": self.sample_count,
        }


def calculate_curve_metrics(predictions: List[float], targets: List[float]) -> CurveEvaluationMetrics:
    """Compute MAE, RMSE, R2, NRMSE, and Max Absolute Error for curve prediction."""
    if not predictions or not targets or len(predictions) != len(targets):
        return CurveEvaluationMetrics(mae=0.0, rmse=0.0, r2=0.0, nrmse=0.0, max_absolute_error=0.0, sample_count=0)

    n = len(targets)
    mae = sum(abs(p - t) for p, t in zip(predictions, targets)) / n
    mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / n
    rmse = math.sqrt(mse)

    mean_t = sum(targets) / n
    ss_tot = sum((t - mean_t) ** 2 for t in targets)
    ss_res = sum((t - p) ** 2 for p, t in zip(predictions, targets))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 1.0

    target_range = max(targets) - min(targets) if targets else 1.0
    nrmse = rmse / target_range if target_range > 1e-9 else 0.0

    max_abs_err = max(abs(p - t) for p, t in zip(predictions, targets))

    return CurveEvaluationMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        nrmse=nrmse,
        max_absolute_error=max_abs_err,
        sample_count=n,
    )


def calculate_scalar_metrics(predictions: List[float], targets: List[float]) -> ScalarEvaluationMetrics:
    """Compute MAE, RMSE, R2 for scalar prediction."""
    if not predictions or not targets or len(predictions) != len(targets):
        return ScalarEvaluationMetrics(mae=0.0, rmse=0.0, r2=0.0, sample_count=0)

    n = len(targets)
    mae = sum(abs(p - t) for p, t in zip(predictions, targets)) / n
    mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / n
    rmse = math.sqrt(mse)

    mean_t = sum(targets) / n
    ss_tot = sum((t - mean_t) ** 2 for t in targets)
    ss_res = sum((t - p) ** 2 for p, t in zip(predictions, targets))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 1.0

    return ScalarEvaluationMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        sample_count=n,
    )
