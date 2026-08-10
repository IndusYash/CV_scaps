"""
app.ml
------
ML Dataset Preparation Layer.
Provides clean, traceable long-format records and quality reports for downstream ML systems.
Consumes ONLY canonical ScientificDataset representations.
"""

from app.ml.builder import MLDatasetBuilder
from app.ml.model import MLCurveRecord, MLDatasetCollection, MLScalarRecord
from app.ml.quality_report import (
    MLQualityReport,
    generate_ml_quality_report,
    get_experiment_grouping_diagnostics,
)

__all__ = [
    "MLCurveRecord",
    "MLScalarRecord",
    "MLDatasetCollection",
    "MLDatasetBuilder",
    "MLQualityReport",
    "generate_ml_quality_report",
    "get_experiment_grouping_diagnostics",
]
