"""
app.ml.modeling.trainer
-----------------------
Trainer interface contracts for model training and evaluation.

Contains turn-key execution contracts for training baseline models and PyTorch MLPs
when model training is explicitly authorized.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.ml.modeling.config import TrainingConfig
from app.ml.modeling.dataset import CurvePointDataset
from app.ml.modeling.metrics import CurveEvaluationMetrics, calculate_curve_metrics
from app.ml.modeling.models import BaselineLinearModel, BaselineMeanModel, TORCH_AVAILABLE


class SolarModelTrainer:
    """Trainer manager for baseline and neural curve surrogate models."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.baseline_mean = BaselineMeanModel()
        self.baseline_linear = BaselineLinearModel()
        self.is_trained: bool = False

    def train_baseline_models(self, train_dataset: CurvePointDataset) -> Dict[str, Any]:
        """Fit baseline mean and baseline linear models on training dataset."""
        features = train_dataset.features
        targets = train_dataset.targets

        self.baseline_mean.fit(targets)
        self.baseline_linear.fit(features, targets)

        self.is_trained = True
        return {
            "baseline_mean_val": self.baseline_mean.mean_val,
            "baseline_linear_weights_count": len(self.baseline_linear.weights),
            "status": "baseline_models_fitted",
        }

    def evaluate_baseline_models(self, test_dataset: CurvePointDataset) -> Dict[str, CurveEvaluationMetrics]:
        """Evaluate baseline models on evaluation dataset."""
        features = test_dataset.features
        targets = test_dataset.targets

        mean_preds = self.baseline_mean.predict(features)
        linear_preds = self.baseline_linear.predict(features)

        mean_metrics = calculate_curve_metrics(mean_preds, targets)
        linear_metrics = calculate_curve_metrics(linear_preds, targets)

        return {
            "baseline_mean": mean_metrics,
            "baseline_linear": linear_metrics,
        }
