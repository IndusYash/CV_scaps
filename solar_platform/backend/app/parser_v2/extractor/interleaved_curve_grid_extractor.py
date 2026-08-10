"""
parser_v2.extractor.interleaved_curve_grid_extractor
----------------------------------------------------
Stage 5 Extractor for InterleavedCurveGrid interpretations.

Design Principles:
  - Deterministic consumer of Reasoner's RegionInterpretation.
  - Extracts repeated independent (axis, target) column groups:
      Curve 1 = (x_1, y_1)
      Curve 2 = (x_2, y_2)
      ...
  - Every curve receives its OWN AxisData and TraceData independently.
    Curves are NEVER forced onto a single shared x-axis.
  - Accommodates identical x-grid lengths, different x-grid lengths,
    truncated curves, and missing values without data fabrication.
  - Preserves 100% CellRef provenance for all x, y, and parameter cells.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.extractor.ir import (
    AxisData,
    ExtractedDataset,
    RawRegionData,
    TraceData,
)
from app.parser_v2.probes.evidence import CellRef
from app.parser_v2.probes.parameter_probe import parse_parameter_expression
from app.parser_v2.reader.raw_model import RawCellType, RawSheet
from app.parser_v2.reasoner.hypothesis import RegionInterpretation
from app.parser_v2.scanner.region import CandidateRegion


def _suggest_axis_unit(concept: str) -> str:
    if concept == "Voltage":
        return "V"
    elif concept == "Wavelength":
        return "nm"
    elif concept == "Position":
        return "um"
    return ""


def extract_interleaved_curve_grid(
    sheet: RawSheet,
    region: CandidateRegion,
    interpretation: RegionInterpretation,
) -> List[ExtractedDataset]:
    """
    Extract independent (axis, target) curve datasets from an InterleavedCurveGrid.
    """
    winner = interpretation.winner
    sci_label = winner.scientific_label if winner else "UNKNOWN"
    axis_concept = winner.axis_concept if winner else "Voltage"
    target_concept = winner.target_concept if winner else "Current"

    axis_unit = _suggest_axis_unit(axis_concept or "")

    # Retrieve interleaved column group evidence
    interleaved_items = interpretation.evidence_bag.filter_by_name("interleaved_column_group")
    groups_info: List[Dict[str, Any]] = []

    if interleaved_items and interleaved_items[0].polarity.name == "POSITIVE":
        groups_info = interleaved_items[0].value.get("groups", [])
        pairs = interleaved_items[0].value.get("pairs", [])
        if not groups_info and pairs:
            groups_info = [{"param_col": None, "x_col": p[0], "y_cols": [p[1]]} for p in pairs]

    if not groups_info:
        # Fallback: scan columns for adjacent monotonic numeric pairs
        x_cols = [c.col_index for c in region.columns_info if c.numeric_density >= 0.40 and (c.is_monotonic_increasing or c.is_monotonic_decreasing)]
        for x_c in x_cols:
            y_cand = [c.col_index for c in region.columns_info if c.col_index > x_c and c.numeric_density >= 0.30]
            if y_cand:
                groups_info.append({"param_col": x_c - 1 if x_c > region.col_start else None, "x_col": x_c, "y_cols": [y_cand[0]]})

    # Prepare RawRegionData
    raw_cells = []
    for r in range(region.row_start, region.row_end + 1):
        row_slice = [sheet.get(r, c) for c in range(region.col_start, region.col_end + 1)]
        raw_cells.append(row_slice)

    raw_data = RawRegionData(
        region_id=region.region_id,
        sheet_name=sheet.sheet_name,
        row_start=region.row_start,
        col_start=region.col_start,
        row_end=region.row_end,
        col_end=region.col_end,
        cells=raw_cells,
    )

    extracted_datasets: List[ExtractedDataset] = []
    header_row = max(0, region.row_start - 1)

    for idx, grp in enumerate(groups_info):
        x_col = grp["x_col"]
        y_cols = grp.get("y_cols", [])
        if not y_cols:
            continue
        y_col = y_cols[0]
        param_col = grp.get("param_col")

        # 1. Parameter extraction from header
        p_val: Optional[float] = None
        p_unit: str = ""
        p_raw: str = ""
        p_src: Optional[CellRef] = None

        # Check param_col, x_col, or header cells for parameter text
        cand_p_cols = [c for c in (param_col, x_col - 1, x_col, y_col) if c is not None and region.col_start <= c <= region.col_end]
        for p_c in cand_p_cols:
            for r_hdr in (region.row_start, header_row, max(0, region.row_start - 2)):
                cell = sheet.get(r_hdr, p_c)
                if cell and cell.data_type == RawCellType.STRING and cell.display_str:
                    cell_ref = CellRef(sheet.sheet_name, r_hdr, p_c, cell.raw_value, cell.display_str)
                    cand_param = parse_parameter_expression(cell.display_str, cell_ref)
                    if cand_param is not None:
                        p_val = cand_param.parsed_value
                        p_unit = cand_param.unit_hint or ""
                        p_raw = cand_param.raw_text or cell.display_str
                        p_src = cand_param.source_cell
                        break
            if p_raw:
                break

        # 2. Extract x and y values for this curve
        x_vals: List[float] = []
        x_refs: List[CellRef] = []
        y_vals: List[float] = []
        y_refs: List[CellRef] = []

        # Data rows start after header row if header is inside region
        data_start_row = region.row_start
        # If row_start cell is a string header, advance data_start_row
        cell_x_start = sheet.get(region.row_start, x_col)
        if cell_x_start and cell_x_start.data_type == RawCellType.STRING:
            data_start_row = region.row_start + 1

        for r in range(data_start_row, region.row_end + 1):
            cx = sheet.get(r, x_col)
            cy = sheet.get(r, y_col)

            # Valid data row must contain numeric x and y
            if cx is None or cy is None:
                continue
            if cx.data_type not in (RawCellType.NUMERIC, RawCellType.FORMULA) or cy.data_type not in (RawCellType.NUMERIC, RawCellType.FORMULA):
                continue
            if not isinstance(cx.raw_value, (int, float)) or not isinstance(cy.raw_value, (int, float)):
                continue

            x_ref = CellRef(sheet.sheet_name, r, x_col, cx.raw_value, cx.display_str)
            y_ref = CellRef(sheet.sheet_name, r, y_col, cy.raw_value, cy.display_str)

            x_vals.append(float(cx.raw_value))
            x_refs.append(x_ref)
            y_vals.append(float(cy.raw_value))
            y_refs.append(y_ref)

        if not x_vals or not y_vals:
            continue

        axis_data = AxisData(
            concept=axis_concept or "Voltage",
            unit=axis_unit,
            values=x_vals,
            source_refs=x_refs,
        )

        trace = TraceData(
            trace_index=0,
            parameter_value=p_val,
            parameter_unit=p_unit,
            parameter_raw_text=p_raw,
            parameter_source=p_src,
            values=y_vals,
            source_refs=y_refs,
        )

        dataset = ExtractedDataset(
            region_id=f"{region.region_id}_curve_{idx+1}",
            sheet_name=sheet.sheet_name,
            structure_type="InterleavedCurveGrid",
            scientific_label=sci_label,
            axis_concept=axis_concept,
            target_concept=target_concept,
            acceptance_state=interpretation.acceptance_state.value,
            winner_evidence_score=winner.score if winner else 0.0,
            winner_runner_up_gap=interpretation.winner_runner_up_gap,
            axis=axis_data,
            traces=[trace],
            raw=raw_data,
            parent_region_id=region.region_id,
        )

        extracted_datasets.append(dataset)

    return extracted_datasets
