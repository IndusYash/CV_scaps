"""
parser_v2.validator.numerical_validator
--------------------------------------
Level 3: Numerical Validation.

Checks numerical validity of ExtractedDataset IR objects:
- Values are valid floats (not NaN, None, or Infinity)
- Reports duplicate x-axis values
- Reports non-monotonic or non-strictly-ordered x-axis grids
- Reports malformed parameter values
- Detects uncached/uncomputed Excel formula cells in raw grid

Non-mutating contract:
- Only emits ValidationIssue objects.
- Does NOT alter any dataset fields.
"""
from __future__ import annotations

import math
from typing import List

from app.parser_v2.extractor.ir import ExtractedDataset
from app.parser_v2.reader.raw_model import RawCellType
from app.parser_v2.validator.models import (
    IssueSeverity,
    ValidationIssue,
    ValidationLevel,
)


def _is_invalid_float(v: Any) -> bool:
    """Return True if v is None, not a float/int, or is NaN / Infinity."""
    if v is None:
        return True
    if not isinstance(v, (int, float)):
        return True
    return math.isnan(v) or math.isinf(v)


def validate_numerics(dataset: ExtractedDataset) -> List[ValidationIssue]:
    """Perform Level 3 Numerical Validation checks on an ExtractedDataset."""
    issues: List[ValidationIssue] = []

    # 1. Axis numeric validity, duplicates, and monotonicity
    if dataset.axis is not None:
        axis = dataset.axis
        invalid_axis_indices = [i for i, v in enumerate(axis.values) if _is_invalid_float(v)]
        if invalid_axis_indices:
            bad_refs = [axis.source_refs[i] for i in invalid_axis_indices if i < len(axis.source_refs)]
            issues.append(ValidationIssue(
                level=ValidationLevel.NUMERICAL,
                severity=IssueSeverity.WARNING,
                code="INVALID_NUMERIC_AXIS",
                message=(
                    f"Region {dataset.region_id} axis contains {len(invalid_axis_indices)} "
                    f"non-numeric, NaN, or infinite values at indices: {invalid_axis_indices[:5]}."
                ),
                source_cells=bad_refs,
            ))

        # Check duplicate x-values
        if len(axis.values) > 1 and len(set(axis.values)) < len(axis.values):
            duplicates = [v for i, v in enumerate(axis.values) if axis.values.count(v) > 1]
            dup_set = sorted(set(duplicates))
            issues.append(ValidationIssue(
                level=ValidationLevel.NUMERICAL,
                severity=IssueSeverity.WARNING,
                code="DUPLICATE_X_VALUE",
                message=(
                    f"Region {dataset.region_id} axis contains duplicate x-values: {dup_set[:5]}."
                ),
                source_cells=axis.source_refs[:5],
            ))

        # Check monotonicity for known numeric grids
        if len(axis.values) >= 3 and not invalid_axis_indices:
            diffs = [axis.values[i+1] - axis.values[i] for i in range(len(axis.values) - 1)]
            is_increasing = all(d >= 0 for d in diffs)
            is_decreasing = all(d <= 0 for d in diffs)
            if not (is_increasing or is_decreasing):
                issues.append(ValidationIssue(
                    level=ValidationLevel.NUMERICAL,
                    severity=IssueSeverity.WARNING,
                    code="NON_MONOTONIC_AXIS",
                    message=(
                        f"Region {dataset.region_id} axis values are non-monotonic."
                    ),
                    source_cells=axis.source_refs[:5],
                ))

    # 2. Trace numeric validity and parameter value validity
    for trace in dataset.traces:
        invalid_y_indices = [i for i, v in enumerate(trace.values) if _is_invalid_float(v)]
        if invalid_y_indices:
            bad_refs = [trace.source_refs[i] for i in invalid_y_indices if i < len(trace.source_refs)]
            issues.append(ValidationIssue(
                level=ValidationLevel.NUMERICAL,
                severity=IssueSeverity.WARNING,
                code="INVALID_NUMERIC_TRACE",
                message=(
                    f"Region {dataset.region_id} trace {trace.trace_index} contains {len(invalid_y_indices)} "
                    f"non-numeric, NaN, or infinite values."
                ),
                source_cells=bad_refs,
            ))

        if trace.parameter_value is not None and _is_invalid_float(trace.parameter_value):
            issues.append(ValidationIssue(
                level=ValidationLevel.NUMERICAL,
                severity=IssueSeverity.WARNING,
                code="MALFORMED_PARAMETER_VALUE",
                message=(
                    f"Region {dataset.region_id} trace {trace.trace_index} has malformed parameter_value: "
                    f"{trace.parameter_value}."
                ),
                source_cells=[trace.parameter_source] if trace.parameter_source else [],
            ))

    # 3. Scalar sweep rows numeric validity
    for idx, row in enumerate(dataset.scalar_rows):
        if row.parameter_value is not None and _is_invalid_float(row.parameter_value):
            issues.append(ValidationIssue(
                level=ValidationLevel.NUMERICAL,
                severity=IssueSeverity.WARNING,
                code="MALFORMED_SCALAR_PARAMETER_VALUE",
                message=(
                    f"Region {dataset.region_id} scalar row {idx} has malformed parameter_value: "
                    f"{row.parameter_value}."
                ),
                source_cells=[row.parameter_source] if row.parameter_source else [],
            ))
        for m_key, m_val in row.metrics.items():
            if _is_invalid_float(m_val):
                src = row.metric_sources.get(m_key)
                issues.append(ValidationIssue(
                    level=ValidationLevel.NUMERICAL,
                    severity=IssueSeverity.WARNING,
                    code="INVALID_SCALAR_METRIC_VALUE",
                    message=(
                        f"Region {dataset.region_id} scalar row {idx} metric '{m_key}' has invalid value: {m_val}."
                    ),
                    source_cells=[src] if src else [],
                ))

    # 4. Raw formula / uncomputed cached-value detection
    if dataset.raw is not None:
        for row_cells in dataset.raw.cells:
            for cell in row_cells:
                if cell.data_type == RawCellType.FORMULA:
                    if cell.raw_value is None:
                        from app.parser_v2.probes.evidence import CellRef
                        issues.append(ValidationIssue(
                            level=ValidationLevel.NUMERICAL,
                            severity=IssueSeverity.WARNING,
                            code="UNCACHED_FORMULA_VALUE",
                            message=(
                                f"Cell ({cell.row},{cell.col}) contains formula '{cell.formula_str}' "
                                "but cached result is None (uncomputed Excel formula)."
                            ),
                            source_cells=[CellRef(dataset.sheet_name, cell.row, cell.col, cell.raw_value, cell.display_str)],
                        ))

    return issues
