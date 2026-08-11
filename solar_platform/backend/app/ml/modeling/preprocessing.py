"""
app.ml.modeling.preprocessing
-----------------------------
ML Preprocessing and Feature Normalization layer.

Enforces strict fit() on training split only to prevent data leakage.
Encodes material categorical features and applies log10/z-score transformations
to physical parameters while preserving native x-coordinates.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from app.ml.model import MLCurveRecord, MLDatasetCollection, MLScalarRecord
from app.ml.modeling.config import PreprocessorConfig


class MLPreprocessor:
    """
    Feature preprocessor converting native ML records into normalized vector tensors.

    Input Feature Vector Layout (15 dimensions):
      - Indices 0..5 : Material One-Hot Vector (6 categories)
      - Indices 6..13: Curated Parameter Vector (8 features, scaled)
      - Index 14     : Native x-coordinate (Voltage or Wavelength, scaled)
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()
        self.param_means: Dict[str, float] = {}
        self.param_stds: Dict[str, float] = {}
        self.x_mean: float = 0.0
        self.x_std: float = 1.0
        self.is_fitted: bool = False

    def fit(self, training_records: List[MLCurveRecord]) -> MLPreprocessor:
        """Fit normalization statistics ONLY on training split curve records."""
        param_acc: Dict[str, List[float]] = {p: [] for p in self.config.known_parameters}
        x_vals: List[float] = []

        for r in training_records:
            x_vals.append(r.x)
            for p_name in self.config.known_parameters:
                val = r.parameters.get(p_name)
                if val is not None and isinstance(val, (int, float)):
                    if p_name in self.config.log_transform_parameters and val > 0:
                        param_acc[p_name].append(math.log10(val))
                    else:
                        param_acc[p_name].append(float(val))

        # Compute means and stds for parameters
        for p_name, vals in param_acc.items():
            if vals:
                mean = sum(vals) / len(vals)
                variance = sum((v - mean) ** 2 for v in vals) / len(vals)
                std = math.sqrt(variance) if variance > 1e-9 else 1.0
                self.param_means[p_name] = mean
                self.param_stds[p_name] = std
            else:
                self.param_means[p_name] = 0.0
                self.param_stds[p_name] = 1.0

        # Compute mean and std for x
        if x_vals:
            self.x_mean = sum(x_vals) / len(x_vals)
            x_var = sum((v - self.x_mean) ** 2 for v in x_vals) / len(x_vals)
            self.x_std = math.sqrt(x_var) if x_var > 1e-9 else 1.0
        else:
            self.x_mean = 0.0
            self.x_std = 1.0

        self.is_fitted = True
        return self

    def transform_material_one_hot(self, material_name: str) -> List[float]:
        """Encode material string as one-hot vector."""
        vec = [0.0] * len(self.config.known_materials)
        clean_mat = material_name.replace(".xlsx", "")
        if clean_mat in self.config.known_materials:
            idx = self.config.known_materials.index(clean_mat)
            vec[idx] = 1.0
        else:
            # Fallback for unknown materials
            vec[0] = 1.0
        return vec

    def transform_parameters(self, parameters: Dict[str, Any]) -> List[float]:
        """Transform parameters into scaled vector."""
        vec: List[float] = []
        for p_name in self.config.known_parameters:
            val = parameters.get(p_name, 0.0)
            if val is not None and isinstance(val, (int, float)):
                f_val = float(val)
                if p_name in self.config.log_transform_parameters and f_val > 0:
                    f_val = math.log10(f_val)
            else:
                f_val = self.param_means.get(p_name, 0.0)

            mean = self.param_means.get(p_name, 0.0)
            std = self.param_stds.get(p_name, 1.0)
            scaled_val = (f_val - mean) / std
            vec.append(scaled_val)
        return vec

    def transform_curve_record(self, record: MLCurveRecord) -> Tuple[List[float], float]:
        """Convert an MLCurveRecord into (input_features, y_target)."""
        if not self.is_fitted:
            raise RuntimeError("MLPreprocessor must be fitted on training split before transform.")

        mat_vec = self.transform_material_one_hot(record.material)
        param_vec = self.transform_parameters(record.parameters)
        x_scaled = (record.x - self.x_mean) / self.x_std

        features = mat_vec + param_vec + [x_scaled]
        return features, record.y
