"""
parser_v2.scanner.boundary_detector
-----------------------------------
Detects structural boundaries, empty row/column separators, and bounding boxes
in a RawSheet.

Pure structural logic only. Does not inspect string contents for keywords.
"""
from __future__ import annotations

from typing import List, Tuple

from app.parser_v2.reader.raw_model import RawCellType, RawSheet


def is_row_empty(sheet: RawSheet, row: int, col_start: int = 0, col_end: Optional[int] = None) -> bool:
    """Return True if all cells in sheet[row][col_start..col_end] are RawCellType.EMPTY."""
    if col_end is None:
        col_end = sheet.n_cols - 1
    for c in range(max(0, col_start), min(sheet.n_cols, col_end + 1)):
        cell = sheet.get(row, c)
        if cell is not None and cell.data_type != RawCellType.EMPTY:
            return False
    return True


def is_col_empty(sheet: RawSheet, col: int, row_start: int = 0, row_end: Optional[int] = None) -> bool:
    """Return True if all cells in sheet[row_start..row_end][col] are RawCellType.EMPTY."""
    if row_end is None:
        row_end = sheet.n_rows - 1
    for r in range(max(0, row_start), min(sheet.n_rows, row_end + 1)):
        cell = sheet.get(r, col)
        if cell is not None and cell.data_type != RawCellType.EMPTY:
            return False
    return True


def find_empty_row_separators(sheet: RawSheet, col_start: int = 0, col_end: Optional[int] = None) -> List[int]:
    """Return 0-indexed row indices that are completely empty across the specified column span."""
    if sheet.n_rows == 0:
        return []
    return [r for r in range(sheet.n_rows) if is_row_empty(sheet, r, col_start, col_end)]


def find_empty_col_separators(sheet: RawSheet, row_start: int = 0, row_end: Optional[int] = None) -> List[int]:
    """Return 0-indexed column indices that are completely empty across the specified row span."""
    if sheet.n_cols == 0:
        return []
    return [c for c in range(sheet.n_cols) if is_col_empty(sheet, c, row_start, row_end)]


def detect_rectangular_bounds(sheet: RawSheet) -> List[Tuple[int, int, int, int]]:
    """
    Slices a RawSheet into candidate rectangular bounding boxes (row_start, row_end, col_start, col_end).

    Algorithm:
      1. First find horizontal blocks bounded by empty rows.
      2. For each horizontal block, find vertical sub-blocks separated by empty columns.
      3. Shrink each block's bounding box to eliminate any remaining leading/trailing empty edge rows or columns.
      4. Filter out completely empty blocks.

    Returns a list of 4-tuples: (row_start, row_end, col_start, col_end), all 0-indexed inclusive.
    """
    if sheet.n_rows == 0 or sheet.n_cols == 0:
        return []

    # Step 1: Divide into row slices separated by fully empty rows
    row_slices: List[Tuple[int, int]] = []
    r_start = None

    for r in range(sheet.n_rows):
        empty = is_row_empty(sheet, r)
        if not empty and r_start is None:
            r_start = r
        elif empty and r_start is not None:
            row_slices.append((r_start, r - 1))
            r_start = None

    if r_start is not None:
        row_slices.append((r_start, sheet.n_rows - 1))

    if not row_slices:
        return []

    boxes: List[Tuple[int, int, int, int]] = []

    # Step 2: For each row slice, divide into column sub-slices separated by empty columns
    for r_min, r_max in row_slices:
        c_start = None
        for c in range(sheet.n_cols):
            empty = is_col_empty(sheet, c, r_min, r_max)
            if not empty and c_start is None:
                c_start = c
            elif empty and c_start is not None:
                boxes.append((r_min, r_max, c_start, c - 1))
                c_start = None
        if c_start is not None:
            boxes.append((r_min, r_max, c_start, sheet.n_cols - 1))

    # Step 3: Trim edge empty rows and columns for each candidate box
    tight_boxes: List[Tuple[int, int, int, int]] = []
    for r_min, r_max, c_min, c_max in boxes:
        # Trim top
        while r_min <= r_max and is_row_empty(sheet, r_min, c_min, c_max):
            r_min += 1
        # Trim bottom
        while r_max >= r_min and is_row_empty(sheet, r_max, c_min, c_max):
            r_max -= 1
        # Trim left
        while c_min <= c_max and is_col_empty(sheet, c_min, r_min, r_max):
            c_min += 1
        # Trim right
        while c_max >= c_min and is_col_empty(sheet, c_max, r_min, r_max):
            c_max -= 1

        if r_min <= r_max and c_min <= c_max:
            tight_boxes.append((r_min, r_max, c_min, c_max))

    return tight_boxes
