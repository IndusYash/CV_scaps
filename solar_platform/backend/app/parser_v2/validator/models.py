"""
parser_v2.validator.models
--------------------------
Data models for Stage 6 Validator results and issues.

The Validator is a non-mutating evaluator of ExtractedDataset IR objects.
It produces a ValidatedDataset containing structured validation findings
without modifying, repairing, deleting, or reinterpreting the source data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from app.parser_v2.extractor.ir import ExtractedDataset
from app.parser_v2.probes.evidence import CellRef


class ValidationLevel(str, Enum):
    """The four required levels of validation."""
    STRUCTURAL           = "STRUCTURAL"
    SEMANTIC             = "SEMANTIC"
    NUMERICAL            = "NUMERICAL"
    PHYSICAL_PLAUSIBILITY = "PHYSICAL_PLAUSIBILITY"


class IssueSeverity(str, Enum):
    """Severity of a validation finding."""
    CRITICAL = "CRITICAL"   # Structural/integrity breakdown (makes structure invalid)
    WARNING  = "WARNING"    # Data anomaly, non-monotonic x, or physical range warning
    INFO     = "INFO"       # Informational observation


class OverallValidationStatus(str, Enum):
    """Overall status of dataset validation."""
    ACCEPTED_VALID          = "ACCEPTED_VALID"          # ACCEPTED + 0 CRITICAL or WARNING issues
    ACCEPTED_WITH_WARNINGS  = "ACCEPTED_WITH_WARNINGS"  # ACCEPTED + WARNING issues, 0 CRITICAL
    UNCERTAIN               = "UNCERTAIN"               # Reasoner state was UNCERTAIN
    UNRESOLVED              = "UNRESOLVED"              # Reasoner state was UNRESOLVED
    INVALID_STRUCTURE       = "INVALID_STRUCTURE"       # Has CRITICAL structural issues


@dataclass
class ValidationIssue:
    """
    A single validation issue or finding.

    Includes explicit level, severity, machine-readable code, human-readable
    message, and source cell provenance pointers where available.
    """
    level: ValidationLevel
    severity: IssueSeverity
    code: str
    message: str
    source_cells: List[CellRef] = field(default_factory=list)


@dataclass
class ValidatedDataset:
    """
    Validation outcome wrapping an unmodified ExtractedDataset.

    Non-mutating guarantee:
    - ``dataset`` is the exact, un-mutated original ExtractedDataset.
    - Raw data, extracted candidates, and ambiguity metadata are 100% preserved.

    ML readiness:
    - ``is_ml_ready`` is True ONLY for ACCEPTED_VALID and ACCEPTED_WITH_WARNINGS.
    - UNCERTAIN, UNRESOLVED, and INVALID_STRUCTURE datasets have ``is_ml_ready = False``.
    """
    dataset: ExtractedDataset
    validation_status: OverallValidationStatus
    is_ml_ready: bool

    structural_valid: bool
    semantic_valid: bool
    numerical_valid: bool
    physical_plausible: bool

    issues: List[ValidationIssue] = field(default_factory=list)

    def filter_by_level(self, level: ValidationLevel) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == level]

    def filter_by_severity(self, severity: IssueSeverity) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == severity]
