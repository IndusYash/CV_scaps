"""
parser_v2.evaluation.fidelity_models
------------------------------------
Data structures and Enums for Stage 6.5 Extraction Fidelity Evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MatchCategory(str, Enum):
    """Fidelity classification for an evaluated dataset against ground truth."""
    EXACT_MATCH                    = "EXACT_MATCH"
    NUMERIC_MATCH_WITHIN_TOLERANCE  = "NUMERIC_MATCH_WITHIN_TOLERANCE"
    STRUCTURAL_MATCH_VALUE_MISMATCH = "STRUCTURAL_MATCH_VALUE_MISMATCH"
    MISSING_DATA                   = "MISSING_DATA"
    EXTRA_DATA                     = "EXTRA_DATA"
    INTERPRETATION_MISMATCH        = "INTERPRETATION_MISMATCH"
    UNRESOLVED_BY_PARSER           = "UNRESOLVED_BY_PARSER"
    NO_GROUND_TRUTH_AVAILABLE      = "NO_GROUND_TRUTH_AVAILABLE"


@dataclass
class AspectComparison:
    """Detailed comparison of one specific aspect (out of the 15 required)."""
    aspect_name: str
    is_match: bool
    expected: Any
    actual: Any
    notes: str = ""


@dataclass
class MismatchDetail:
    """Detailed record of a specific point-level or metadata mismatch."""
    aspect_name: str
    location: str                     # e.g. "Trace 0, Point 5 (row 7, col 2)"
    expected_value: Any
    actual_value: Any
    origin: str                       # "reading", "interpretation", "extraction", "normalization", "formula/cached-value handling", or "ground-truth preparation"
    source_provenance: Optional[str] = None


@dataclass
class DatasetFidelityResult:
    """Fidelity evaluation result for one dataset."""
    sheet_name: str
    experiment_identifier: str
    match_category: MatchCategory
    fidelity_score: float             # 0.0 to 1.0 (1.0 = 100% agreement)
    aspects: Dict[str, AspectComparison] = field(default_factory=dict)
    mismatches: List[MismatchDetail] = field(default_factory=list)
    raw_data_preserved: bool = True
    notes: str = ""


@dataclass
class OverallFidelityReport:
    """Summary fidelity report across all evaluated sheets and datasets."""
    total_datasets_evaluated: int
    datasets_with_ground_truth: int
    exact_matches: int
    tolerance_matches: int
    value_mismatches: int
    missing_data_count: int
    unresolved_by_parser: int
    no_ground_truth_count: int
    overall_fidelity_percentage: float     # Percentage over ground-truth items
    dataset_results: List[DatasetFidelityResult] = field(default_factory=list)
