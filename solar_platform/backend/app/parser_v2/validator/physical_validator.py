"""
parser_v2.validator.physical_validator
--------------------------------------
Level 4: Physical Plausibility Validation.

Applies conservative physical/domain sanity checks:
- Voltage range [-5V, +5V]
- Current Density range [-500, +500] mA/cm^2
- QE range [-1%, +150%]
- Wavelength range [100, 3000] nm
- Temperature > 0 K
- Thickness > 0 nm
- Defect / Donor / Acceptor / Interface density > 0 cm^-3 / cm^-2
- Resistance > 0 ohm.cm^2

CRITICAL DOMAIN RULES:
- Only emits WARNING / INFO severity issues.
- Does NOT treat unusual or extreme values as automatically invalid.
- NEVER modifies, repairs, or discards extracted data.
- NEVER forces an unresolved experiment to fail physical checks —
  if a concept is unknown, no physical check is enforced.
"""
from __future__ import annotations

from typing import List

from app.parser_v2.extractor.ir import ExtractedDataset
from app.parser_v2.validator.models import (
    IssueSeverity,
    ValidationIssue,
    ValidationLevel,
)


def validate_physical_plausibility(dataset: ExtractedDataset) -> List[ValidationIssue]:
    """Perform Level 4 Physical Plausibility checks on an ExtractedDataset."""
    issues: List[ValidationIssue] = []

    # 1. Axis concept physical range check
    if dataset.axis is not None and dataset.axis.values:
        concept = dataset.axis.concept
        vals = dataset.axis.values
        mn, mx = min(vals), max(vals)

        if concept == "Voltage":
            if mn < -5.0 or mx > 5.0:
                issues.append(ValidationIssue(
                    level=ValidationLevel.PHYSICAL_PLAUSIBILITY,
                    severity=IssueSeverity.WARNING,
                    code="UNUSUAL_VOLTAGE_RANGE",
                    message=(
                        f"Region {dataset.region_id} Voltage axis range [{mn:.2f}, {mx:.2f}] V "
                        "is outside typical solar cell operating range [-5.0, 5.0] V."
                    ),
                    source_cells=dataset.axis.source_refs[:3],
                ))

        elif concept == "Wavelength":
            if mn < 100.0 or mx > 3000.0:
                issues.append(ValidationIssue(
                    level=ValidationLevel.PHYSICAL_PLAUSIBILITY,
                    severity=IssueSeverity.WARNING,
                    code="UNUSUAL_WAVELENGTH_RANGE",
                    message=(
                        f"Region {dataset.region_id} Wavelength axis range [{mn:.1f}, {mx:.1f}] nm "
                        "is outside typical UV-Vis-NIR spectrum [100, 3000] nm."
                    ),
                    source_cells=dataset.axis.source_refs[:3],
                ))

        elif concept == "Temperature":
            if mn <= 0.0:
                issues.append(ValidationIssue(
                    level=ValidationLevel.PHYSICAL_PLAUSIBILITY,
                    severity=IssueSeverity.WARNING,
                    code="PHYSICAL_TEMPERATURE_LE_ZERO",
                    message=(
                        f"Region {dataset.region_id} Temperature axis min value {mn:.1f} K is <= 0 K."
                    ),
                    source_cells=dataset.axis.source_refs[:3],
                ))

    # 2. Target concept physical range check on trace values
    if dataset.target_concept is not None and dataset.traces:
        target = dataset.target_concept

        if target == "Current":
            for trace in dataset.traces:
                if trace.values:
                    mn, mx = min(trace.values), max(trace.values)
                    if mn < -500.0 or mx > 500.0:
                        issues.append(ValidationIssue(
                            level=ValidationLevel.PHYSICAL_PLAUSIBILITY,
                            severity=IssueSeverity.WARNING,
                            code="UNUSUAL_CURRENT_RANGE",
                            message=(
                                f"Region {dataset.region_id} trace {trace.trace_index} Current range "
                                f"[{mn:.2f}, {mx:.2f}] mA/cm^2 is outside expected bounds [-500, 500] mA/cm^2."
                            ),
                            source_cells=trace.source_refs[:3],
                        ))
                        break  # Only report once per dataset

        elif target == "QE":
            for trace in dataset.traces:
                if trace.values:
                    mn, mx = min(trace.values), max(trace.values)
                    if mn < -1.0 or mx > 150.0:
                        issues.append(ValidationIssue(
                            level=ValidationLevel.PHYSICAL_PLAUSIBILITY,
                            severity=IssueSeverity.WARNING,
                            code="UNUSUAL_QE_RANGE",
                            message=(
                                f"Region {dataset.region_id} trace {trace.trace_index} QE range "
                                f"[{mn:.2f}, {mx:.2f}] % is outside expected bounds [-1%, 150%]."
                            ),
                            source_cells=trace.source_refs[:3],
                        ))
                        break

    # 3. Swept parameter physical check
    for trace in dataset.traces:
        if trace.parameter_value is not None:
            pv = trace.parameter_value
            unit = (trace.parameter_unit or "").lower()

            if "temperature" in (trace.parameter_raw_text or "").lower() or unit == "k":
                if pv <= 0:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.PHYSICAL_PLAUSIBILITY,
                        severity=IssueSeverity.WARNING,
                        code="PHYSICAL_TEMPERATURE_LE_ZERO",
                        message=f"Region {dataset.region_id} Temperature parameter {pv} K is <= 0 K.",
                        source_cells=[trace.parameter_source] if trace.parameter_source else [],
                    ))

            elif "thickness" in (trace.parameter_raw_text or "").lower() or unit == "nm":
                if pv <= 0:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.PHYSICAL_PLAUSIBILITY,
                        severity=IssueSeverity.WARNING,
                        code="PHYSICAL_THICKNESS_LE_ZERO",
                        message=f"Region {dataset.region_id} Thickness parameter {pv} nm is <= 0 nm.",
                        source_cells=[trace.parameter_source] if trace.parameter_source else [],
                    ))

            elif "density" in (trace.parameter_raw_text or "").lower() or "cm" in unit:
                if pv <= 0:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.PHYSICAL_PLAUSIBILITY,
                        severity=IssueSeverity.WARNING,
                        code="PHYSICAL_DENSITY_LE_ZERO",
                        message=f"Region {dataset.region_id} Density parameter {pv} is <= 0.",
                        source_cells=[trace.parameter_source] if trace.parameter_source else [],
                    ))

    return issues
