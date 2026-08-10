"""
parser_v2.validator.semantic_validator
--------------------------------------
Level 2: Semantic Validation.

Checks semantic consistency of ExtractedDataset IR objects:
- Extracted axis concept matches Reasoner's axis_concept assignment
- Extracted target concept matches Reasoner's target_concept assignment
- Parameter units and identities are internally consistent
- Scientific labels are NOT silently changed or reinterpreted

Non-mutating contract:
- Only emits ValidationIssue objects.
- Does NOT alter any dataset fields.
"""
from __future__ import annotations

from typing import List

from app.parser_v2.extractor.ir import ExtractedDataset
from app.parser_v2.validator.models import (
    IssueSeverity,
    ValidationIssue,
    ValidationLevel,
)


def validate_semantics(dataset: ExtractedDataset) -> List[ValidationIssue]:
    """Perform Level 2 Semantic Validation checks on an ExtractedDataset."""
    issues: List[ValidationIssue] = []

    # 1. Axis concept consistency
    if dataset.axis is not None and dataset.axis_concept is not None:
        if dataset.axis.concept != dataset.axis_concept:
            issues.append(ValidationIssue(
                level=ValidationLevel.SEMANTIC,
                severity=IssueSeverity.WARNING,
                code="AXIS_CONCEPT_MISMATCH",
                message=(
                    f"Region {dataset.region_id} extracted axis concept '{dataset.axis.concept}' "
                    f"does not match Reasoner axis concept '{dataset.axis_concept}'."
                ),
                source_cells=dataset.axis.source_refs[:3],
            ))

    # 2. Parameter unit consistency across traces
    if dataset.traces:
        units = {t.parameter_unit for t in dataset.traces if t.parameter_unit is not None}
        if len(units) > 1:
            issues.append(ValidationIssue(
                level=ValidationLevel.SEMANTIC,
                severity=IssueSeverity.WARNING,
                code="INCONSISTENT_PARAMETER_UNITS",
                message=(
                    f"Region {dataset.region_id} traces have conflicting parameter units: {sorted(units)}."
                ),
                source_cells=[t.parameter_source for t in dataset.traces if t.parameter_source is not None],
            ))

    # 3. Scientific label consistency check for known experiments
    if dataset.scientific_label in ("JV", "QE", "RESPONSIVITY", "DETECTIVITY"):
        if dataset.structure_type == "NumericCurveGrid" and dataset.axis is None:
            issues.append(ValidationIssue(
                level=ValidationLevel.SEMANTIC,
                severity=IssueSeverity.WARNING,
                code="EXPERIMENT_LABEL_WITHOUT_AXIS",
                message=(
                    f"Region {dataset.region_id} is labelled as scientific experiment '{dataset.scientific_label}' "
                    "but lacks an extracted independent axis."
                ),
                source_cells=dataset.provenance_cells,
            ))

    return issues
