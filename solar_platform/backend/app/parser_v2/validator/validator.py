"""
parser_v2.validator.validator
-----------------------------
Stage 6 Validator top-level entry points.

Public functions:
  - ``validate_dataset(dataset: ExtractedDataset) -> ValidatedDataset``
  - ``validate_sheet_datasets(datasets: List[ExtractedDataset]) -> List[ValidatedDataset]``

Contract:
  - NON-MUTATING: The input ExtractedDataset is NEVER altered, repaired,
    deleted, or reinterpreted.
  - Four levels of validation run sequentially: Structural, Semantic, Numerical, Physical Plausibility.
  - Status classification explicitly distinguishes ACCEPTED_VALID, ACCEPTED_WITH_WARNINGS,
    UNCERTAIN, UNRESOLVED, and INVALID_STRUCTURE.
  - ML Readiness: ``is_ml_ready = True`` ONLY for ACCEPTED_VALID and ACCEPTED_WITH_WARNINGS.
    UNCERTAIN, UNRESOLVED, and INVALID_STRUCTURE are always ``is_ml_ready = False``.
  - CellRef provenance is attached to all reported issues where available.
"""
from __future__ import annotations

from typing import List

from app.parser_v2.extractor.ir import ExtractedDataset
from app.parser_v2.validator.models import (
    IssueSeverity,
    OverallValidationStatus,
    ValidatedDataset,
    ValidationIssue,
    ValidationLevel,
)
from app.parser_v2.validator.numerical_validator import validate_numerics
from app.parser_v2.validator.physical_validator import validate_physical_plausibility
from app.parser_v2.validator.semantic_validator import validate_semantics
from app.parser_v2.validator.structural_validator import validate_structure


def validate_dataset(dataset: ExtractedDataset) -> ValidatedDataset:
    """
    Validate one ExtractedDataset IR object across all 4 validation levels.

    Returns a ValidatedDataset container wrapping the unmodified ExtractedDataset.
    Does NOT mutate, clean, drop, or reinterpret any data.
    """
    struct_issues = validate_structure(dataset)
    sem_issues = validate_semantics(dataset)
    num_issues = validate_numerics(dataset)
    phys_issues = validate_physical_plausibility(dataset)

    all_issues: List[ValidationIssue] = (
        struct_issues + sem_issues + num_issues + phys_issues
    )

    structural_valid = not any(i.severity == IssueSeverity.CRITICAL for i in struct_issues)
    semantic_valid = not any(i.severity == IssueSeverity.CRITICAL for i in sem_issues)
    numerical_valid = not any(i.severity == IssueSeverity.CRITICAL for i in num_issues)
    physical_plausible = not any(
        i.level == ValidationLevel.PHYSICAL_PLAUSIBILITY and i.severity == IssueSeverity.WARNING
        for i in phys_issues
    )

    # Determine overall status and ML readiness
    acc_state = dataset.acceptance_state

    if acc_state == "UNRESOLVED":
        status = OverallValidationStatus.UNRESOLVED
        is_ml_ready = False
    elif acc_state == "UNCERTAIN":
        status = OverallValidationStatus.UNCERTAIN
        is_ml_ready = False
    elif not structural_valid or any(i.severity == IssueSeverity.CRITICAL for i in all_issues):
        status = OverallValidationStatus.INVALID_STRUCTURE
        is_ml_ready = False
    elif any(i.severity == IssueSeverity.WARNING for i in all_issues):
        status = OverallValidationStatus.ACCEPTED_WITH_WARNINGS
        is_ml_ready = True
    else:
        status = OverallValidationStatus.ACCEPTED_VALID
        is_ml_ready = True

    return ValidatedDataset(
        dataset=dataset,
        validation_status=status,
        is_ml_ready=is_ml_ready,
        structural_valid=structural_valid,
        semantic_valid=semantic_valid,
        numerical_valid=numerical_valid,
        physical_plausible=physical_plausible,
        issues=all_issues,
    )


def validate_sheet_datasets(datasets: List[ExtractedDataset]) -> List[ValidatedDataset]:
    """Validate a list of ExtractedDatasets for a sheet."""
    return [validate_dataset(ds) for ds in datasets]
