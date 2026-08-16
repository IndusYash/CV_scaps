"""
parser_v2.reader.raw_model
--------------------------
Pure data classes that represent a workbook exactly as openpyxl observes it.

Rules:
  - No scientific interpretation of any kind.
  - No reference to experiments, axes, parameters, or curve types.
  - Every field maps 1-to-1 to information available from openpyxl.
  - Row and column indices are always 0-indexed throughout this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Cell type classification
# ---------------------------------------------------------------------------

class RawCellType(str, Enum):
    """
    What openpyxl reports about a cell's content.

    FORMULA  — cell contains an Excel formula (value may be the computed result
                or None if the workbook has never been opened in Excel).
    NUMERIC  — cell holds an integer or float (no formula).
    STRING   — cell holds a non-empty text value (no formula).
    BOOLEAN  — cell holds True or False (no formula).
    DATE     — cell holds a datetime / date / timedelta object (no formula).
    EMPTY    — cell is blank (value is None or whitespace-only string).
    """
    FORMULA = "FORMULA"
    NUMERIC = "NUMERIC"
    STRING  = "STRING"
    BOOLEAN = "BOOLEAN"
    DATE    = "DATE"
    EMPTY   = "EMPTY"


# ---------------------------------------------------------------------------
# Individual cell
# ---------------------------------------------------------------------------

@dataclass
class RawCell:
    """
    Represents one Excel cell with full provenance and formatting metadata.

    Provenance guarantee:
        (row, col) uniquely identify this cell within its sheet.
        raw_value is exactly what openpyxl reports after loading with
        data_only=True (i.e., the computed value, not the formula text).

    Formula handling:
        If the cell contains a formula:
          - formula_str holds the formula source (e.g. "=SUM(A1:A3)")
          - raw_value holds the last computed value (may be None if the
            workbook was never opened in Excel)
          - data_type is FORMULA regardless of raw_value's Python type

    Merge handling:
        is_merged=True means this cell is part of a merged region.
        merge_parent=(r, c) points to the top-left (anchor) cell.
        For the anchor cell itself:  is_merged=True, merge_parent=None.
        For non-merged cells:        is_merged=False, merge_parent=None.
    """
    row: int                              # 0-indexed row
    col: int                              # 0-indexed column

    data_type: RawCellType

    raw_value: Any                        # Python value from openpyxl (data_only=True)
    display_str: str                      # best-effort string representation

    formula_str: Optional[str]            # "=SUM(A1:A3)" or None

    is_bold: bool                         # font.bold from formatting pass
    background_color: Optional[str]       # hex ARGB e.g. "FF0070C0", or None

    is_merged: bool                       # True ↔ part of a merged region
    merge_parent: Optional[Tuple[int, int]]  # (row, col) of anchor, or None


# ---------------------------------------------------------------------------
# Sheet
# ---------------------------------------------------------------------------

@dataclass
class RawSheet:
    """
    Represents one worksheet as a complete, rectangular cell grid.

    The grid is always n_rows × n_cols and contains a RawCell at every
    position — empty cells are represented by RawCell(data_type=EMPTY)
    rather than being omitted.

    n_rows and n_cols are derived from openpyxl's max_row / max_column
    (the extents of the used range, not necessarily the entire sheet).
    """
    sheet_name: str
    sheet_index: int                      # 0-indexed position in workbook
    n_rows: int
    n_cols: int
    cells: List[List[RawCell]]            # cells[row][col], both 0-indexed

    def get(self, row: int, col: int) -> Optional[RawCell]:
        """
        Return the cell at (row, col) or None if out of bounds.
        Both indices are 0-indexed.
        """
        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return self.cells[row][col]
        return None

    def iter_cells(self):
        """Yield every RawCell in row-major order."""
        for row_cells in self.cells:
            yield from row_cells


# ---------------------------------------------------------------------------
# Workbook
# ---------------------------------------------------------------------------

@dataclass
class RawWorkbook:
    """
    Represents an entire Excel workbook as returned by the Stage 1 reader.

    named_ranges maps defined-name → address string exactly as stored in
    the workbook XML.  This is preserved verbatim with no interpretation.
    """
    filename: str
    n_sheets: int
    sheets: List[RawSheet]
    named_ranges: Dict[str, str] = field(default_factory=dict)

    def get_sheet(self, name: str) -> Optional[RawSheet]:
        """Return the sheet with the given name, or None if not found."""
        for sheet in self.sheets:
            if sheet.sheet_name == name:
                return sheet
        return None

    def sheet_names(self) -> List[str]:
        """Return sheet names in workbook order."""
        return [s.sheet_name for s in self.sheets]
