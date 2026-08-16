"""
parser_v2.reader.raw_reader
---------------------------
Stage 1 reader: converts an Excel workbook (bytes) into a RawWorkbook.

Design contract (enforced by this module):
  - Reads using openpyxl only.
  - Loads the workbook TWICE:
      Pass A  data_only=False  →  captures formula strings + formatting
      Pass B  data_only=True   →  captures computed cell values
  - Preserves every cell, including empties.
  - Preserves merged-cell membership and anchor identity.
  - Preserves font-bold and background-color where available.
  - Preserves named ranges verbatim.
  - Does NOT classify experiments, axes, parameters, or regions.
  - Does NOT drop, skip, or summarise any cell.
  - Raises ValueError for non-Excel input.

Threading: not thread-safe (do not share the same bytes buffer
across threads without locking).
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import openpyxl

from app.parser_v2.reader.raw_model import (
    RawCell,
    RawCellType,
    RawSheet,
    RawWorkbook,
)


# ---------------------------------------------------------------------------
# Internal helpers — all private to this module
# ---------------------------------------------------------------------------

def _classify_value(value: Any) -> RawCellType:
    """
    Map a Python value (from openpyxl data_only=True) to a RawCellType.
    Called only for cells that do NOT contain a formula.
    """
    if value is None:
        return RawCellType.EMPTY
    if isinstance(value, bool):
        # bool must be checked before int because bool is a subclass of int
        return RawCellType.BOOLEAN
    if isinstance(value, (int, float)):
        return RawCellType.NUMERIC
    if isinstance(value, str):
        return RawCellType.EMPTY if value.strip() == "" else RawCellType.STRING
    # datetime.datetime, datetime.date, datetime.timedelta
    return RawCellType.DATE


def _make_display_str(data_type: RawCellType, raw_value: Any, formula_str: Optional[str]) -> str:
    """
    Build a best-effort display string.

    For FORMULA cells: use str(raw_value) of the computed value if available,
    otherwise fall back to the formula text.
    For empty cells: empty string.
    For all others: str(raw_value).

    Note: openpyxl does not evaluate number formats, so this is not
    identical to what Excel displays for formatted numbers.
    """
    if data_type == RawCellType.EMPTY:
        return ""
    if data_type == RawCellType.FORMULA:
        if raw_value is not None:
            return str(raw_value)
        return formula_str or ""
    if raw_value is None:
        return ""
    return str(raw_value)


def _extract_bg_color(cell) -> Optional[str]:
    """
    Return the hex ARGB background color of a cell, or None.
    openpyxl stores fill colors under cell.fill.fgColor.
    We only return theme-independent rgb values.
    """
    try:
        fill = cell.fill
        if fill is None or fill.fill_type in (None, "none"):
            return None
        fg = fill.fgColor
        if fg is None:
            return None
        if fg.type == "rgb":
            rgb = fg.rgb
            # "00000000" is the default transparent/unset value
            if rgb and rgb != "00000000":
                return rgb
        # theme-indexed colors are recorded with a prefix so callers can
        # distinguish them from RGB values
        if fg.type == "theme":
            return f"theme:{fg.theme}"
    except Exception:
        pass
    return None


def _build_merge_map(ws) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Return a dict mapping every cell in a merged region to its anchor (top-left) cell.
    Both keys and values are 0-indexed (row, col) tuples.

    Anchor cells are included: anchor → anchor.
    Non-merged cells are NOT in the dict.
    """
    merge_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for merged_range in ws.merged_cells.ranges:
        anchor = (merged_range.min_row - 1, merged_range.min_col - 1)
        for r in range(merged_range.min_row, merged_range.max_row + 1):
            for c in range(merged_range.min_col, merged_range.max_col + 1):
                merge_map[(r - 1, c - 1)] = anchor
    return merge_map


def _build_formula_map(ws_fmt) -> Dict[Tuple[int, int], str]:
    """
    From a worksheet loaded with data_only=False, return a dict of
    (row_0, col_0) → formula_string for every cell that contains a formula.
    Cells with ordinary values are not included.
    """
    formula_map: Dict[Tuple[int, int], str] = {}
    if ws_fmt.max_row is None or ws_fmt.max_column is None:
        return formula_map
    for row_cells in ws_fmt.iter_rows():
        for cell in row_cells:
            val = cell.value
            if isinstance(val, str) and val.startswith("="):
                formula_map[(cell.row - 1, cell.column - 1)] = val
    return formula_map


def _build_raw_cell(
    r0: int,
    c0: int,
    computed_value: Any,
    formula_str: Optional[str],
    fmt_cell,
    merge_map: Dict[Tuple[int, int], Tuple[int, int]],
) -> RawCell:
    """Assemble a RawCell from all collected pieces."""

    # Data type
    if formula_str is not None:
        data_type = RawCellType.FORMULA
    else:
        data_type = _classify_value(computed_value)

    display_str = _make_display_str(data_type, computed_value, formula_str)

    # Formatting — guarded: a missing or None fmt_cell is acceptable
    is_bold = False
    bg_color: Optional[str] = None
    if fmt_cell is not None:
        try:
            if fmt_cell.font and fmt_cell.font.bold:
                is_bold = True
        except Exception:
            pass
        bg_color = _extract_bg_color(fmt_cell)

    # Merge membership
    coord = (r0, c0)
    if coord in merge_map:
        anchor = merge_map[coord]
        is_merged = True
        # The anchor cell has merge_parent=None; non-anchor cells point to anchor
        merge_parent: Optional[Tuple[int, int]] = None if anchor == coord else anchor
    else:
        is_merged = False
        merge_parent = None

    return RawCell(
        row=r0,
        col=c0,
        data_type=data_type,
        raw_value=computed_value,
        display_str=display_str,
        formula_str=formula_str,
        is_bold=is_bold,
        background_color=bg_color,
        is_merged=is_merged,
        merge_parent=merge_parent,
    )


def _read_sheet(
    sheet_name: str,
    sheet_index: int,
    ws_fmt,
    ws_val,
) -> RawSheet:
    """
    Read one worksheet and return a RawSheet.
    ws_fmt is loaded with data_only=False (for formulas + formatting).
    ws_val is loaded with data_only=True  (for computed values).
    """
    # Grid dimensions — take the maximum reported by each pass
    n_rows = max(ws_fmt.max_row or 0, ws_val.max_row or 0)
    n_cols = max(ws_fmt.max_column or 0, ws_val.max_column or 0)

    if n_rows == 0 or n_cols == 0:
        return RawSheet(
            sheet_name=sheet_name,
            sheet_index=sheet_index,
            n_rows=0,
            n_cols=0,
            cells=[],
        )

    # Build per-sheet lookup tables
    merge_map   = _build_merge_map(ws_fmt)
    formula_map = _build_formula_map(ws_fmt)

    # Load both passes into flat dicts for O(1) access during grid construction
    val_map:  Dict[Tuple[int, int], Any]   = {}
    fmt_map:  Dict[Tuple[int, int], Any]   = {}

    for row_cells in ws_val.iter_rows(min_row=1, max_row=n_rows,
                                       min_col=1, max_col=n_cols):
        for cell in row_cells:
            val_map[(cell.row - 1, cell.column - 1)] = cell.value

    for row_cells in ws_fmt.iter_rows(min_row=1, max_row=n_rows,
                                       min_col=1, max_col=n_cols):
        for cell in row_cells:
            fmt_map[(cell.row - 1, cell.column - 1)] = cell

    # Build the complete rectangular grid
    grid: List[List[RawCell]] = []
    for r0 in range(n_rows):
        row_out: List[RawCell] = []
        for c0 in range(n_cols):
            coord = (r0, c0)
            raw_cell = _build_raw_cell(
                r0=r0,
                c0=c0,
                computed_value=val_map.get(coord),
                formula_str=formula_map.get(coord),
                fmt_cell=fmt_map.get(coord),
                merge_map=merge_map,
            )
            row_out.append(raw_cell)
        grid.append(row_out)

    return RawSheet(
        sheet_name=sheet_name,
        sheet_index=sheet_index,
        n_rows=n_rows,
        n_cols=n_cols,
        cells=grid,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_workbook(filename: str, contents: bytes) -> RawWorkbook:
    """
    Parse an Excel workbook from raw bytes and return a RawWorkbook.

    Parameters
    ----------
    filename : str
        Original filename (used only for provenance; not for I/O).
    contents : bytes
        Raw bytes of an .xlsx or .xls file.

    Returns
    -------
    RawWorkbook
        Complete workbook representation with no scientific interpretation.

    Raises
    ------
    ValueError
        If openpyxl cannot open the bytes as an Excel workbook.
    """
    try:
        buf_fmt = io.BytesIO(contents)
        wb_fmt = openpyxl.load_workbook(buf_fmt, data_only=False, read_only=False)

        buf_val = io.BytesIO(contents)
        wb_val = openpyxl.load_workbook(buf_val, data_only=True,  read_only=False)
    except Exception as exc:
        raise ValueError(f"Cannot read workbook '{filename}': {exc}") from exc

    # Named ranges — preserved verbatim, no interpretation
    named_ranges: Dict[str, str] = {}
    try:
        for name, defn in wb_fmt.defined_names.items():
            if hasattr(defn, "attr_text") and defn.attr_text:
                named_ranges[name] = defn.attr_text
    except Exception:
        pass  # named ranges are optional; failures must not abort reading

    # Read every sheet
    raw_sheets: List[RawSheet] = []
    for idx, sheet_name in enumerate(wb_fmt.sheetnames):
        ws_fmt = wb_fmt[sheet_name]
        ws_val = wb_val[sheet_name]
        raw_sheet = _read_sheet(
            sheet_name=sheet_name,
            sheet_index=idx,
            ws_fmt=ws_fmt,
            ws_val=ws_val,
        )
        raw_sheets.append(raw_sheet)

    return RawWorkbook(
        filename=filename,
        n_sheets=len(raw_sheets),
        sheets=raw_sheets,
        named_ranges=named_ranges,
    )
