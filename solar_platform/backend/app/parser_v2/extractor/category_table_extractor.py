"""
parser_v2.extractor.category_table_extractor
---------------------------------------------
Deterministic extractor for CategoryTable regions.

Consumes:
  - RawSheet            (raw cell values with provenance)
  - CandidateRegion     (structural bounding box)
  - RegionInterpretation (winner)

Produces:
  - List[CategoryRow]   (one per data row, with verbatim label and values)

Contract:
  - Does NOT invent scientific meaning for category labels.
  - All column names come from header row verbatim.
  - Provenance invariant: every values[key] has a value_sources[key].
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.parser_v2.probes.evidence import CellRef
from app.parser_v2.reader.raw_model import RawCell, RawCellType, RawSheet
from app.parser_v2.reasoner.hypothesis import RegionInterpretation
from app.parser_v2.scanner.region import CandidateRegion
from app.parser_v2.extractor.ir import CategoryRow


def _make_cell_ref(sheet: RawSheet, r: int, c: int) -> CellRef:
    cell = sheet.get(r, c)
    if cell is None:
        return CellRef(sheet.sheet_name, r, c, None, "")
    return CellRef(sheet.sheet_name, r, c, cell.raw_value, cell.display_str)


def _cell_raw_value(cell: Optional[RawCell]) -> Any:
    """Return the raw_value from a cell, or the display_str if raw_value is None."""
    if cell is None:
        return None
    if cell.raw_value is not None:
        return cell.raw_value
    return cell.display_str if cell.display_str and cell.display_str.strip() else None


def extract_category_table(
    sheet: RawSheet,
    region: CandidateRegion,
    interpretation: RegionInterpretation,
) -> List[CategoryRow]:
    """
    Extract rows from a CategoryTable region.

    Column 0 is the label column.  Remaining columns are value columns.
    Column names are taken verbatim from the header row.

    Returns a list of CategoryRow objects.
    """
    header_row: Optional[int] = None
    if region.candidate_header_rows:
        header_row = region.candidate_header_rows[0]

    label_col = region.col_start
    value_cols: List[int] = list(range(region.col_start + 1, region.col_end + 1))

    # Build column name map from header row (verbatim, no scientific interpretation)
    col_names: Dict[int, str] = {}
    if header_row is not None:
        for c in value_cols:
            hdr = sheet.get(header_row, c)
            if hdr and hdr.display_str and hdr.display_str.strip():
                col_names[c] = hdr.display_str.strip()
            else:
                col_names[c] = f"col_{c}"
    else:
        for c in value_cols:
            col_names[c] = f"col_{c}"

    data_row_start = (header_row + 1) if header_row is not None else region.row_start

    rows: List[CategoryRow] = []

    for r in range(data_row_start, region.row_end + 1):
        label_cell = sheet.get(r, label_col)
        label_val = _cell_raw_value(label_cell)
        if label_val is None:
            continue  # skip empty rows
        label_str = str(label_val).strip()
        if not label_str:
            continue

        label_source = _make_cell_ref(sheet, r, label_col)

        values: Dict[str, Any] = {}
        value_sources: Dict[str, CellRef] = {}

        for c in value_cols:
            col_name = col_names[c]
            val_cell = sheet.get(r, c)
            val = _cell_raw_value(val_cell)
            if val is not None:
                values[col_name] = val
                value_sources[col_name] = _make_cell_ref(sheet, r, c)

        rows.append(CategoryRow(
            label=label_str,
            label_source=label_source,
            values=values,
            value_sources=value_sources,
        ))

    return rows
