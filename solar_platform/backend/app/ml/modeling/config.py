"""
app.ml.modeling.config
----------------------
Configuration data models for ML Model Architecture and Experiment Design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskType(str, Enum):
    CURVE_CONDITIONING = "CURVE_CONDITIONING"      # Primary V1: f(material, params, x) -> y (JV, QE, etc.)
    SCALAR_PREDICTION = "SCALAR_PREDICTION"        # Secondary: f(material, params) -> Jsc, Voc, FF, PCE
    COORDINATE_SURROGATE = "COORDINATE_SURROGATE"  # Shared surrogate encoder across scientific types


class SplitRegime(str, Enum):
    REGIME_A_IN_DOMAIN = "REGIME_A_IN_DOMAIN"        # 70% Train / 15% Val / 15% Test split by experiment_family_id
    REGIME_B_CROSS_MATERIAL = "REGIME_B_CROSS_MATERIAL" # Train on Ca3PI3/Ca3PBr3/DataFile, Test on Ca3AsBr3


class ParameterTransformType(str, Enum):
    IDENTITY = "IDENTITY"
    STANDARD = "STANDARD"
    LOG10 = "LOG10"
    MINMAX = "MINMAX"


@dataclass
class PreprocessorConfig:
    """Configuration for dataset normalization and encoding."""
    known_materials: List[str] = field(
        default_factory=lambda: ["Ca3PI3", "Ca3PBr3", "Ca3AsBr3", "Data File (1)", "Data file", "SolarDevice_DataFile"]
    )
    known_parameters: List[str] = field(
        default_factory=lambda: [
            "thickness",
            "temperature",
            "series_resistance",
            "shunt_resistance",
            "defect_density",
            "doping_density",
            "illumination_wavelength",
            "sweep_index",
        ]
    )
    log_transform_parameters: List[str] = field(
        default_factory=lambda: ["defect_density", "doping_density", "shunt_resistance"]
    )
    scale_parameters: bool = True
    preserve_native_x: bool = True


@dataclass
class ModelConfig:
    """Hyperparameters and architecture configuration for V1 Surrogate Model."""
    model_name: str = "SolarCurveSurrogateMLP"
    input_dim: int = 15                   # 6 materials + 8 parameters + 1 x-coordinate
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 64])
    activation: str = "GELU"
    dropout_rate: float = 0.05
    use_residual: bool = True
    use_layer_norm: bool = True
    output_dim: int = 1


@dataclass
class TrainingConfig:
    """Execution parameters for model training and evaluation."""
    task_type: TaskType = TaskType.CURVE_CONDITIONING
    split_regime: SplitRegime = SplitRegime.REGIME_A_IN_DOMAIN
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    max_epochs: int = 100
    early_stopping_patience: int = 15
    huber_delta: float = 1.0
    seed: int = 42
    model_config: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "huber_delta": self.huber_delta,
            "model_config": {
                "model_name": self.model_config.model_name,
                "input_dim": self.model_config.input_dim,
                "hidden_dims": self.model_config.hidden_dims,
                "dropout_rate": self.model_config.dropout_rate,
                "use_residual": self.model_config.use_residual,
            },
        }
