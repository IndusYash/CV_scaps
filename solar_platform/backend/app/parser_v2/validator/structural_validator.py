"""
parser_v2.validator.structural_validator
----------------------------------------
Level 1: Structural Validation.

Checks structural integrity of ExtractedDataset IR objects:
- x/y length alignment in NumericCurveGrid
- trace value / source_ref provenance length alignment
- axis value / source_ref provenance length alignment
- presence of required axis when traces exist
- parameter/provenance consistency
- scalar row key alignment
- category row key alignment
- cross-region sheet provenance consistency

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


def validate_structure(dataset: ExtractedDataset) -> List[ValidationIssue]:
    """Perform Level 1 Structural Validation checks on an ExtractedDataset."""
    issues: List[ValidationIssue] = []

    # 1. NumericCurveGrid, StackedCurveBlocks & InterleavedCurveGrid checks
    if dataset.structure_type in ("NumericCurveGrid", "StackedCurveBlocks", "InterleavedCurveGrid"):
        axis = dataset.axis
        traces = dataset.traces

        if traces and axis is None:
            issues.append(ValidationIssue(
                level=ValidationLevel.STRUCTURAL,
                severity=IssueSeverity.CRITICAL,
                code="MISSING_AXIS",
                message=f"Region {dataset.region_id} has {len(traces)} traces but no independent axis.",
                source_cells=dataset.provenance_cells,
            ))

        if axis is not None:
            # Check axis provenance length
            if len(axis.values) != len(axis.source_refs):
                issues.append(ValidationIssue(
                    level=ValidationLevel.STRUCTURAL,
                    severity=IssueSeverity.CRITICAL,
                    code="AXIS_REF_MISMATCH",
                    message=(
                        f"Region {dataset.region_id} axis length mismatch: "
                        f"{len(axis.values)} values != {len(axis.source_refs)} source_refs."
                    ),
                    source_cells=axis.source_refs,
                ))

            # Check each trace
            for trace in traces:
                # x/y length match
                if len(trace.values) != len(axis.values):
                    issues.append(ValidationIssue(
                        level=ValidationLevel.STRUCTURAL,
                        severity=IssueSeverity.CRITICAL,
                        code="X_Y_LENGTH_MISMATCH",
                        message=(
                            f"Region {dataset.region_id} trace {trace.trace_index} length mismatch: "
                            f"{len(trace.values)} y-values != {len(axis.values)} x-values."
                        ),
                        source_cells=trace.source_refs[:3],
                    ))

                # trace value / ref length match
                if len(trace.values) != len(trace.source_refs):
                    issues.append(ValidationIssue(
                        level=ValidationLevel.STRUCTURAL,
                        severity=IssueSeverity.CRITICAL,
                        code="TRACE_REF_MISMATCH",
                        message=(
                            f"Region {dataset.region_id} trace {trace.trace_index} provenance mismatch: "
                            f"{len(trace.values)} values != {len(trace.source_refs)} source_refs."
                        ),
                        source_cells=trace.source_refs[:3],
                    ))

                # Parameter provenance check
                if trace.parameter_value is not None and trace.parameter_source is None:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.STRUCTURAL,
                        severity=IssueSeverity.WARNING,
                        code="PARAMETER_PROVENANCE_MISSING",
                        message=(
                            f"Region {dataset.region_id} trace {trace.trace_index} has parameter_value "
                            f"{trace.parameter_value} but parameter_source CellRef is missing."
                        ),
                        source_cells=[],
                    ))

                # Cross-region sheet mixing check
                for ref in trace.source_refs:
                    if ref.sheet_name != dataset.sheet_name:
                        issues.append(ValidationIssue(
                            level=ValidationLevel.STRUCTURAL,
                            severity=IssueSeverity.CRITICAL,
                            code="CROSS_REGION_SHEET_MIXING",
                            message=(
                                f"Region {dataset.region_id} trace {trace.trace_index} contains CellRef "
                                f"from sheet '{ref.sheet_name}' instead of expected '{dataset.sheet_name}'."
                            ),
                            source_cells=[ref],
                        ))
                        break

    # 2. ScalarSweepTable checks
    elif dataset.structure_type == "ScalarSweepTable":
        for idx, row in enumerate(dataset.scalar_rows):
            missing_sources = set(row.metrics.keys()) - set(row.metric_sources.keys())
            if missing_sources:
                issues.append(ValidationIssue(
                    level=ValidationLevel.STRUCTURAL,
                    severity=IssueSeverity.WARNING,
                    code="SCALAR_METRIC_PROVENANCE_MISSING",
                    message=(
                        f"Region {dataset.region_id} scalar row {idx} missing provenance for metrics: "
                        f"{sorted(missing_sources)}"
                    ),
                    source_cells=[row.parameter_source] if row.parameter_source else [],
                ))

    # 3. CategoryTable checks
    elif dataset.structure_type == "CategoryTable":
        for idx, crow in enumerate(dataset.category_rows):
            missing_sources = set(crow.values.keys()) - set(crow.value_sources.keys())
            if missing_sources:
                issues.append(ValidationIssue(
                    level=ValidationLevel.STRUCTURAL,
                    severity=IssueSeverity.WARNING,
                    code="CATEGORY_VALUE_PROVENANCE_MISSING",
                    message=(
                        f"Region {dataset.region_id} category row {idx} missing provenance for columns: "
                        f"{sorted(missing_sources)}"
                    ),
                    source_cells=[crow.label_source],
                ))

    return issues
