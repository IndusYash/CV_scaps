"""parser_v2.evaluation package."""
from app.parser_v2.evaluation.fidelity_evaluator import (
    evaluate_dataset_fidelity,
    evaluate_workbook_fidelity,
)
from app.parser_v2.evaluation.fidelity_models import (
    AspectComparison,
    DatasetFidelityResult,
    MatchCategory,
    MismatchDetail,
    OverallFidelityReport,
)

__all__ = [
    "evaluate_dataset_fidelity",
    "evaluate_workbook_fidelity",
    "DatasetFidelityResult",
    "OverallFidelityReport",
    "MatchCategory",
    "AspectComparison",
    "MismatchDetail",
]
