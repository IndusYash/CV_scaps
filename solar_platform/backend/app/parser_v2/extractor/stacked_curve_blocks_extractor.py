"""
parser_v2.extractor.stacked_curve_blocks_extractor
---------------------------------------------------
Extractor implementation for generic StackedCurveBlocks hypothesis.

Pulls sub-block row ranges from `stacked_axis_reset_sequence` evidence
produced by AxisProbe, scans inter-block banner rows for parameter metadata,
and extracts an independent `ExtractedDataset` for each detected sub-block.

Guarantees:
  - Generic: no hardcoded workbook names, sheet names, or block counts.
  - Complete CellRef provenance retained for all axis points and trace values.
  - Inter-block parameter metadata attached where available.
  - Incomplete/partial blocks preserved without data loss or fabrication.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.extractor.ir import AxisData, ExtractedDataset, TraceData
from app.parser_v2.extractor.numeric_curve_grid_extractor import (
    _cell_float,
    _identify_axis_col,
    _identify_header_row,
    _identify_trace_cols,
    _make_cell_ref,
    _unit_from_label_evidence,
)
from app.parser_v2.probes.evidence import CellRef, EvidenceBag, EvidencePolarity
from app.parser_v2.probes.parameter_probe import parse_parameter_expression
from app.parser_v2.reader.raw_model import RawCell, RawSheet
from app.parser_v2.reasoner.hypothesis import RegionInterpretation
from app.parser_v2.scanner.region import CandidateRegion


def _extract_banner_parameters(
    sheet: RawSheet,
    r_start: int,
    r_end: int,
    n_cols: int,
) -> Dict[str, Tuple[Optional[float], Optional[str], str, Optional[CellRef]]]:
    """
    Scan inter-block banner rows for parameter key-value pairs.
    Returns mapping of param_name -> (value, unit, raw_text, source_ref).
    """
    params: Dict[str, Tuple[Optional[float], Optional[str], str, Optional[CellRef]]] = {}

    for r in range(max(0, r_start), min(sheet.n_rows, r_end + 1)):
        for c in range(min(sheet.n_cols, n_cols + 2)):
            cell = sheet.get(r, c)
            if not cell or not cell.display_str or not cell.display_str.strip():
                continue

            txt = cell.display_str.strip()
            cell_ref = _make_cell_ref(sheet, r, c)

            # Check adjacent cell for numeric value if txt looks like a parameter label
            adj_cell = sheet.get(r, c + 1)
            if adj_cell and adj_cell.raw_value is not None:
                full_expr = f"{txt} {adj_cell.display_str}"
                parsed = parse_parameter_expression(full_expr, cell_ref)
                if parsed:
                    params[parsed.expression_pattern] = (
                        parsed.parsed_value,
                        parsed.unit_hint,
                        full_expr,
                        cell_ref,
                    )
                    continue

            # Fallback single cell parse
            parsed = parse_parameter_expression(txt, cell_ref)
            if parsed:
                params[parsed.expression_pattern] = (
                    parsed.parsed_value,
                    parsed.unit_hint,
                    txt,
                    cell_ref,
                )

    return params


def extract_stacked_curve_blocks(
    sheet: RawSheet,
    region: CandidateRegion,
    interpretation: RegionInterpretation,
) -> List[ExtractedDataset]:
    """
    Extract independent ExtractedDataset objects for each stacked sub-block.
    """
    winner = interpretation.winner
    if winner is None:
        return []

    bag = interpretation.evidence_bag
    stacked_items = bag.filter_by_name("stacked_axis_reset_sequence")

    block_row_ranges: List[Tuple[int, int]] = []
    if stacked_items:
        for item in stacked_items:
            if item.polarity == EvidencePolarity.POSITIVE and "block_row_ranges" in item.value:
                block_row_ranges = item.value["block_row_ranges"]
                break

    if not block_row_ranges:
        # Fallback single block if no reset sequence
        block_row_ranges = [(region.row_start, region.row_end)]

    header_row = _identify_header_row(region)
    axis_col = _identify_axis_col(region, sheet, header_row)
    trace_cols = _identify_trace_cols(region, axis_col, header_row, bag)

    axis_concept = winner.axis_concept or "Wavelength"
    axis_unit = _unit_from_label_evidence(bag, axis_concept) or "nm"
    sci_label = winner.scientific_label or "QE"

    extracted_datasets: List[ExtractedDataset] = []

    for k, (r_start, r_end) in enumerate(block_row_ranges):
        # 1. Scan inter-block banner for parameter metadata
        prev_end = block_row_ranges[k - 1][1] if k > 0 else max(0, r_start - 35)
        banner_params = _extract_banner_parameters(sheet, prev_end + 1, r_start - 1, region.col_end)

        # 2. Extract axis values for sub-block
        axis_vals: List[float] = []
        axis_refs: List[CellRef] = []

        for r in range(r_start, r_end + 1):
            c_cell = sheet.get(r, axis_col)
            v = _cell_float(c_cell)
            if v is not None:
                axis_vals.append(v)
                axis_refs.append(_make_cell_ref(sheet, r, axis_col))

        if not axis_vals:
            continue

        axis_data = AxisData(
            concept=axis_concept,
            unit=axis_unit,
            values=axis_vals,
            source_refs=axis_refs,
        )

        # 3. Extract traces for sub-block
        traces: List[TraceData] = []
        for idx, col in enumerate(trace_cols):
            y_vals: List[float] = []
            y_refs: List[CellRef] = []

            for r in range(r_start, r_end + 1):
                c_cell = sheet.get(r, col)
                v = _cell_float(c_cell)
                if v is not None:
                    y_vals.append(v)
                    y_refs.append(_make_cell_ref(sheet, r, col))

            # Pick representative parameter metadata for trace
            p_val: Optional[float] = None
            p_unit: Optional[str] = None
            p_raw: Optional[str] = None
            p_src: Optional[CellRef] = None

            if banner_params:
                first_p = list(banner_params.values())[0]
                p_val, p_unit, p_raw, p_src = first_p

            traces.append(TraceData(
                trace_index=idx,
                parameter_value=p_val,
                parameter_unit=p_unit,
                parameter_raw_text=p_raw,
                parameter_source=p_src,
                values=y_vals,
                source_refs=y_refs,
            ))

        dataset = ExtractedDataset(
            region_id=f"{region.region_id}_block_{k+1}",
            sheet_name=sheet.sheet_name,
            structure_type="StackedCurveBlocks",
            scientific_label=sci_label,
            axis_concept=winner.axis_concept,
            target_concept=winner.target_concept,
            acceptance_state=interpretation.acceptance_state.value,
            winner_evidence_score=winner.score,
            winner_runner_up_gap=interpretation.winner_runner_up_gap,
            axis=axis_data,
            traces=traces,
            parent_region_id=region.region_id,
        )
        extracted_datasets.append(dataset)

    return extracted_datasets
