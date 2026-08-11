"""
app.ml.modeling
---------------
ML Model Architecture, Preprocessing, and Experiment Design Package.
"""

from app.ml.modeling.config import ModelConfig, PreprocessorConfig, SplitRegime, TaskType, TrainingConfig
from app.ml.modeling.dataset import CurvePointDataset, ScalarRecordDataset
from app.ml.modeling.evaluation import EvaluationRunner
from app.ml.modeling.losses import ScientificHuberLoss, calculate_huber_loss_numeric, calculate_mae_numeric, calculate_rmse_numeric
from app.ml.modeling.metrics import CurveEvaluationMetrics, ScalarEvaluationMetrics, calculate_curve_metrics, calculate_scalar_metrics
from app.ml.modeling.models import BaselineLinearModel, BaselineMeanModel, SolarCurveSurrogateMLP, validate_model_forward_pass
from app.ml.modeling.preprocessing import MLPreprocessor
from app.ml.modeling.trainer import SolarModelTrainer

__all__ = [
    "TaskType",
    "SplitRegime",
    "PreprocessorConfig",
    "ModelConfig",
    "TrainingConfig",
    "MLPreprocessor",
    "CurvePointDataset",
    "ScalarRecordDataset",
    "SolarCurveSurrogateMLP",
    "BaselineMeanModel",
    "BaselineLinearModel",
    "validate_model_forward_pass",
    "ScientificHuberLoss",
    "calculate_huber_loss_numeric",
    "calculate_mae_numeric",
    "calculate_rmse_numeric",
    "CurveEvaluationMetrics",
    "ScalarEvaluationMetrics",
    "calculate_curve_metrics",
    "calculate_scalar_metrics",
    "SolarModelTrainer",
    "EvaluationRunner",
]
