"""
app.canonical.converter
-----------------------
Converter mapping ValidatedDataset objects produced by parser_v2 into
unified Canonical ScientificDataset objects.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.canonical.model import (
    CanonicalParameter,
    CanonicalScalarRecord,
    CanonicalTrace,
    CanonicalWorkbook,
    CellRefDTO,
    ScientificDataset,
)
from app.parser_v2.extractor.ir import ExtractedDataset
from app.parser_v2.validator.models import ValidatedDataset


def _derive_parameter_name(sheet_name: str, raw_text: Optional[str], unit: Optional[str]) -> str:
    """Infer descriptive parameter name from sheet context or header text."""
    s_lower = sheet_name.lower()
    if "thick" in s_lower or "pvk" in s_lower or "etl" in s_lower:
        return "thickness"
    if "temp" in s_lower or "t (" in s_lower or "t(" in s_lower or re.search(r"\b(t|k)\b", s_lower):
        return "temperature"
    if "idd" in s_lower or "defect" in s_lower:
        return "defect_density"
    if "rs" in s_lower and "rsh" not in s_lower:
        return "series_resistance"
    if "rsh" in s_lower:
        return "shunt_resistance"
    if re.search(r"\b(na|n_a|acceptor|doping|donor)\b", s_lower) or "doping" in s_lower:
        return "doping_density"
    if "illum" in s_lower or "light" in s_lower:
        return "illumination"

    if raw_text:
        r_lower = raw_text.lower()
        if "nm" in r_lower or "um" in r_lower:
            return "thickness"
        # Narrow temperature mapping: Celsius/Kelvin and boundary checks
        if (
            "kelvin" in r_lower 
            or "celsius" in r_lower 
            or "deg c" in r_lower 
            or "°c" in r_lower 
            or re.search(r"\bdeg\b", r_lower)
            or re.search(r"\b(temp|temperature)\b", r_lower)
            or re.search(r"\b\d+\s*k\b", r_lower)
            or re.search(r"\b(t|k)\b", r_lower)
        ):
            return "temperature"
        if "ohm" in r_lower:
            return "resistance"
        if "cm-3" in r_lower or "10^" in r_lower:
            return "density"

    return "parameter"


def convert_validated_dataset_to_canonical(
    val_ds: ValidatedDataset,
    workbook_name: str = "",
    sheet_name: str = "",
) -> ScientificDataset:
    """
    Convert a ValidatedDataset from Stage 6 into a Canonical ScientificDataset.

    Handles all 5 structural hypotheses (NumericCurveGrid, ScalarSweepTable,
    StackedCurveBlocks, InterleavedCurveGrid, CategoryTable) into a single,
    unified external representation.
    """
    ext: ExtractedDataset = val_ds.dataset

    wb_name = workbook_name or getattr(ext, "workbook_name", "") or "workbook"
    sh_name = sheet_name or ext.sheet_name or "sheet"

    raw_id = f"{wb_name}_{sh_name}_{ext.region_id}"
    dataset_id = re.sub(r"[^\w\-.]", "_", raw_id)

    sci_type = (ext.scientific_label or "UNKNOWN").upper()
    struct_source = ext.structure_type or "NumericCurveGrid"

    axis_name = ext.axis_concept or (ext.axis.concept if ext.axis else None)
    axis_unit = ext.axis.unit if ext.axis else None
    target_name = ext.target_concept

    target_unit = None
    if ext.traces and ext.traces[0].parameter_unit and ext.traces[0].parameter_unit in ["mA/cm2", "%", "A/W", "W^-1", "S/m"]:
        target_unit = ext.traces[0].parameter_unit
    elif sci_type == "JV":
        target_unit = "mA/cm2"
    elif sci_type == "QE":
        target_unit = "%"
    elif sci_type == "RESPONSIVITY":
        target_unit = "A/W"

    x_vals: List[float] = ext.axis.values if ext.axis else []
    x_prov: List[CellRefDTO] = (
        [CellRefDTO.from_cell_ref(c) for c in ext.axis.source_refs]
        if ext.axis and ext.axis.source_refs
        else []
    )

    canonical_traces: List[CanonicalTrace] = []
    if ext.traces:
        for t in ext.traces:
            param_obj = None
            if t.parameter_value is not None or t.parameter_raw_text:
                param_name = _derive_parameter_name(sh_name, t.parameter_raw_text, t.parameter_unit)
                param_val = t.parameter_value if t.parameter_value is not None else t.parameter_raw_text
                param_obj = CanonicalParameter(
                    name=param_name,
                    value=param_val,
                    unit=t.parameter_unit,
                    raw_text=t.parameter_raw_text,
                    provenance=CellRefDTO.from_cell_ref(t.parameter_source),
                )

            trace_id = f"{dataset_id}_trace_{t.trace_index + 1}"
            y_prov = [CellRefDTO.from_cell_ref(c) for c in t.source_refs] if t.source_refs else []

            canonical_traces.append(
                CanonicalTrace(
                    trace_index=t.trace_index,
                    trace_id=trace_id,
                    parameter=param_obj,
                    y_values=t.values,
                    y_provenance=y_prov,
                )
            )

    canonical_scalars: List[CanonicalScalarRecord] = []
    if ext.scalar_rows:
        for s_idx, s_row in enumerate(ext.scalar_rows):
            param_name = _derive_parameter_name(sh_name, s_row.parameter_raw_text, s_row.parameter_unit)
            param_val = s_row.parameter_value if s_row.parameter_value is not None else s_row.parameter_raw_text
            param_obj = CanonicalParameter(
                name=param_name,
                value=param_val,
                unit=s_row.parameter_unit,
                raw_text=s_row.parameter_raw_text,
                provenance=CellRefDTO.from_cell_ref(s_row.parameter_source),
            )

            for target_k, target_v in s_row.metrics.items():
                record_id = f"{dataset_id}_rec_{s_idx + 1}_{target_k}"
                target_prov = CellRefDTO.from_cell_ref(s_row.metric_sources.get(target_k))
                canonical_scalars.append(
                    CanonicalScalarRecord(
                        record_id=record_id,
                        parameter=param_obj,
                        target_name=target_k,
                        target_value=float(target_v),
                        target_unit=None,
                        target_provenance=target_prov,
                    )
                )

    val_issues_text = [
        f"[{i.level.value}] {i.severity.value}: {i.message}" for i in val_ds.issues
    ]

    raw_ref_dict = {
        "region_id": ext.region_id,
        "sheet_name": ext.sheet_name,
        "structure_type": ext.structure_type,
        "acceptance_state": ext.acceptance_state,
        "score": ext.winner_evidence_score,
        "all_scores": ext.all_hypothesis_scores,
        "is_ambiguous": ext.is_ambiguous,
        "ambiguity_note": ext.ambiguity_note,
    }

    return ScientificDataset(
        dataset_id=dataset_id,
        workbook_name=wb_name,
        sheet_name=sh_name,
        region_id=ext.region_id,
        parent_region_id=ext.parent_region_id,
        scientific_type=sci_type,
        structural_source=struct_source,
        axis_name=axis_name,
        axis_unit=axis_unit,
        target_name=target_name,
        target_unit=target_unit,
        global_parameters=[],
        x_values=x_vals,
        x_provenance=x_prov,
        traces=canonical_traces,
        scalar_records=canonical_scalars,
        validation_status=val_ds.validation_status.value,
        acceptance_state=ext.acceptance_state,
        is_ml_ready=val_ds.is_ml_ready,
        confidence_score=ext.winner_evidence_score,
        ambiguity_gap=ext.winner_runner_up_gap,
        validation_issues=val_issues_text,
        raw_dataset_reference=raw_ref_dict,
    )


def build_canonical_workbook(
    workbook_name: str,
    validated_datasets: List[ValidatedDataset],
) -> CanonicalWorkbook:
    """Build a CanonicalWorkbook collection holding all ScientificDataset objects for a workbook."""
    canonical_ds_list = [
        convert_validated_dataset_to_canonical(v, workbook_name=workbook_name)
        for v in validated_datasets
    ]
    return CanonicalWorkbook(workbook_name=workbook_name, datasets=canonical_ds_list)
