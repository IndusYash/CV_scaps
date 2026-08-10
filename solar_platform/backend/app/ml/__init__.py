"""
app.ml
------
ML Dataset Preparation Layer.
Provides clean, traceable long-format records and quality reports for downstream ML systems.
Consumes ONLY canonical ScientificDataset representations.
"""

from app.ml.audit import (
    AuditPointLevelCheck,
    ComprehensiveMLAudit,
    DuplicateAuditResult,
    GroupSplitResult,
    audit_duplicates,
    audit_native_x_grids,
    audit_parameter_distributions,
    audit_point_level_invariants,
    perform_leakage_free_group_split,
    run_comprehensive_ml_audit,
)
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
    "AuditPointLevelCheck",
    "ComprehensiveMLAudit",
    "DuplicateAuditResult",
    "GroupSplitResult",
    "audit_duplicates",
    "audit_native_x_grids",
    "audit_parameter_distributions",
    "audit_point_level_invariants",
    "perform_leakage_free_group_split",
    "run_comprehensive_ml_audit",
]
