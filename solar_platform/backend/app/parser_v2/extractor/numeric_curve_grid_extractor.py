"""
parser_v2.extractor.numeric_curve_grid_extractor
-------------------------------------------------
Deterministic extractor for NumericCurveGrid regions.

Consumes:
  - RawSheet            (raw cell values with provenance)
  - CandidateRegion     (structural bounding box, column info, header texts)
  - RegionInterpretation (winner.axis_concept, winner.target_concept,
                          evidence_bag for parameter and repetition evidence)

Produces:
  - AxisData            (x-axis column values + CellRef provenance)
  - List[TraceData]     (one per y-trace column, with parameter identity)

Contract:
  - Does NOT re-classify or re-apply scientific keyword logic.
  - axis_concept and target_concept come exclusively from the Reasoner.
  - Parameter values come exclusively from parameter_probe evidence items.
  - Provenance invariant: len(TraceData.values) == len(TraceData.source_refs).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.probes.evidence import CellRef, EvidenceBag, EvidencePolarity
from app.parser_v2.probes.parameter_probe import parse_parameter_expression
from app.parser_v2.reader.raw_model import RawCell, RawCellType, RawSheet
from app.parser_v2.reasoner.hypothesis import RegionInterpretation, StructuralHypothesis
from app.parser_v2.scanner.region import CandidateRegion
from app.parser_v2.extractor.ir import AxisData, TraceData


def _make_cell_ref(sheet: RawSheet, r: int, c: int) -> CellRef:
    cell = sheet.get(r, c)
    if cell is None:
        return CellRef(sheet.sheet_name, r, c, None, "")
    return CellRef(sheet.sheet_name, r, c, cell.raw_value, cell.display_str)


def _is_numeric_cell(cell: Optional[RawCell]) -> bool:
    if cell is None:
        return False
    if cell.data_type in (RawCellType.NUMERIC, RawCellType.FORMULA):
        return cell.raw_value is not None and isinstance(cell.raw_value, (int, float))
    return False


def _cell_float(cell: Optional[RawCell]) -> Optional[float]:
    """Return cell raw_value as float or None."""
    if cell is None:
        return None
    if cell.data_type in (RawCellType.NUMERIC, RawCellType.FORMULA):
        try:
            return float(cell.raw_value)
        except (TypeError, ValueError):
            return None
    return None


def _identify_header_row(region: CandidateRegion) -> Optional[int]:
    """Return the first candidate header row, or None."""
    if region.candidate_header_rows:
        return region.candidate_header_rows[0]
    return None


def _identify_axis_col(
    region: CandidateRegion,
    sheet: RawSheet,
    header_row: Optional[int],
) -> int:
    """
    Identify which column is the independent axis (x column).

    Strategy:
    1. Use shared_x_col_candidates if available.
    2. Otherwise use col_start (leftmost column in the region).

    Does NOT inspect cell content for scientific keywords.
    """
    if region.shared_x_col_candidates:
        return region.shared_x_col_candidates[0]
    return region.col_start


def _identify_trace_cols(
    region: CandidateRegion,
    axis_col: int,
    header_row: Optional[int],
) -> List[int]:
    """
    Return all data columns (non-axis, non-empty, numeric-dominant).

    Pulls from repeated_col_patterns if available, otherwise falls back
    to inspecting columns_info.
    """
    # Try repeated column patterns first
    if region.repeated_col_patterns:
        trace_cols: List[int] = []
        for pat in region.repeated_col_patterns:
            for c in range(pat.start_col, pat.end_col + 1):
                if c != axis_col:
                    trace_cols.append(c)
        if trace_cols:
            return sorted(set(trace_cols))

    # Fallback: every non-axis column with >= 50% numeric density
    trace_cols = []
    for col_info in region.columns_info:
        c = col_info.col_index
        if c == axis_col:
            continue
        if col_info.numeric_density >= 0.50:
            trace_cols.append(c)
    return sorted(trace_cols)


def _extract_axis(
    sheet: RawSheet,
    region: CandidateRegion,
    axis_col: int,
    header_row: Optional[int],
    axis_concept: str,
    axis_unit: str,
) -> AxisData:
    """Extract all numeric rows in axis_col (skipping header row)."""
    values: List[float] = []
    refs: List[CellRef] = []

    data_row_start = (header_row + 1) if header_row is not None else region.row_start

    for r in range(data_row_start, region.row_end + 1):
        cell = sheet.get(r, axis_col)
        v = _cell_float(cell)
        if v is not None:
            values.append(v)
            refs.append(_make_cell_ref(sheet, r, axis_col))

    return AxisData(
        concept=axis_concept,
        unit=axis_unit,
        values=values,
        source_refs=refs,
    )


def _extract_parameter_from_header(
    header_text: str,
    header_cell_ref: CellRef,
) -> Tuple[Optional[float], Optional[str], str]:
    """
    Parse the header text to extract (parsed_value, unit_hint, raw_text).
    Delegates entirely to parameter_probe.parse_parameter_expression().
    Returns (None, None, raw_text) if no expression is recognised.
    """
    cand = parse_parameter_expression(header_text, header_cell_ref)
    if cand is not None:
        return cand.parsed_value, cand.unit_hint, cand.raw_text
    return None, None, header_text


def _extract_traces(
    sheet: RawSheet,
    region: CandidateRegion,
    trace_cols: List[int],
    header_row: Optional[int],
    n_axis_rows: int,
) -> List[TraceData]:
    """Extract one TraceData per trace column."""
    traces: List[TraceData] = []
    data_row_start = (header_row + 1) if header_row is not None else region.row_start

    for idx, col in enumerate(trace_cols):
        # --- Parameter identity from header cell ---
        param_value: Optional[float] = None
        param_unit: Optional[str] = None
        param_raw_text: Optional[str] = None
        param_source: Optional[CellRef] = None

        if header_row is not None:
            hdr_cell = sheet.get(header_row, col)
            if hdr_cell and hdr_cell.display_str and hdr_cell.display_str.strip():
                hdr_text = hdr_cell.display_str.strip()
                param_source = _make_cell_ref(sheet, header_row, col)
                param_value, param_unit, param_raw_text = _extract_parameter_from_header(
                    hdr_text, param_source
                )
                if param_raw_text is None:
                    param_raw_text = hdr_text

        # --- Y values ---
        values: List[float] = []
        refs: List[CellRef] = []

        for r in range(data_row_start, region.row_end + 1):
            cell = sheet.get(r, col)
            v = _cell_float(cell)
            if v is not None:
                values.append(v)
                refs.append(_make_cell_ref(sheet, r, col))

        traces.append(TraceData(
            trace_index=idx,
            parameter_value=param_value,
            parameter_unit=param_unit,
            parameter_raw_text=param_raw_text,
            parameter_source=param_source,
            values=values,
            source_refs=refs,
        ))

    return traces


def _unit_from_label_evidence(bag: EvidenceBag, concept_name: str) -> str:
    """
    Look up the suggested unit for a concept name in the label_probe evidence.
    Returns an empty string if not found.
    """
    for item in bag.filter_by_name("candidate_label_concepts"):
        for cand in item.value.get("concept_candidates", []):
            if cand.get("concept") == concept_name:
                unit = cand.get("unit", "")
                if unit:
                    return unit
    return ""


def extract_numeric_curve_grid(
    sheet: RawSheet,
    region: CandidateRegion,
    interpretation: RegionInterpretation,
) -> Tuple[Optional[AxisData], List[TraceData]]:
    """
    Extract axis and traces from a NumericCurveGrid region.

    Returns (axis, traces).  axis may be None if extraction failed.
    All scientific labelling (axis_concept, target_concept) comes from
    interpretation.winner — this function never re-classifies.
    """
    winner = interpretation.winner
    if winner is None:
        return None, []

    # Axis concept and unit come exclusively from the Reasoner
    axis_concept = winner.axis_concept or "Unknown"
    axis_unit = _unit_from_label_evidence(interpretation.evidence_bag, axis_concept)
    if not axis_unit:
        # Fallback: common units for known concepts
        _DEFAULT_UNITS = {
            "Voltage": "V", "Wavelength": "nm", "Position": "um", "Temperature": "K",
        }
        axis_unit = _DEFAULT_UNITS.get(axis_concept, "")

    header_row = _identify_header_row(region)
    axis_col = _identify_axis_col(region, sheet, header_row)
    trace_cols = _identify_trace_cols(region, axis_col, header_row)

    axis = _extract_axis(
        sheet, region, axis_col, header_row, axis_concept, axis_unit
    )
    n_axis_rows = len(axis.values)
    traces = _extract_traces(sheet, region, trace_cols, header_row, n_axis_rows)

    return axis, traces
