"""
parser_v2.scanner.region
------------------------
Data structures representing CandidateRegions and pure structural evidence.

Rules:
  - Exclusively structural metadata.
  - No scientific or domain labels.
  - No keyword-based classification or semantic domain knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ColumnStructureInfo:
    """Structural properties of a single column within a CandidateRegion."""
    col_index: int                       # 0-indexed sheet column index
    region_col_index: int                # 0-indexed column offset within region
    total_cells: int
    numeric_count: int
    string_count: int
    formula_count: int
    empty_count: int
    numeric_density: float
    string_density: float
    dominant_type: str                   # "numeric", "string", "empty", "mixed"
    is_monotonic_increasing: bool
    is_monotonic_decreasing: bool
    distinct_value_count: int
    value_range: Optional[Tuple[float, float]]


@dataclass
class RepeatedColumnPattern:
    """Structural pattern describing a contiguous group of columns with matching shape/type."""
    pattern_type: str                    # e.g. "uniform_numeric_group", "alternating_pair"
    start_col: int                       # 0-indexed sheet col
    end_col: int                         # 0-indexed sheet col (inclusive)
    num_columns: int
    column_row_length: int
    numeric_density: float


@dataclass
class CandidateRegion:
    """
    Represents a rectangular region discovered by boundary analysis in a RawSheet.

    Contains purely structural properties:
      - Spatial bounding box (0-indexed inclusive: row_start, row_end, col_start, col_end)
      - Cell density metrics (numeric, string, formula, empty ratios)
      - Boundary context (empty rows/cols before and after)
      - Column structural analysis (ColumnStructureInfo per column)
      - Shared column candidates (columns that could serve as index/axis grids)
      - Repeated column group patterns
      - Verbatim header/banner text (raw strings preserved without classification)
    """
    region_id: str
    sheet_name: str

    # Bounding Box (0-indexed inclusive)
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    # Dimensions
    n_rows: int
    n_cols: int
    total_cells: int
    non_empty_cells: int

    # Densities [0.0, 1.0]
    non_empty_density: float
    numeric_density: float
    string_density: float
    formula_density: float
    empty_density: float

    # Boundary context
    empty_rows_before: int
    empty_rows_after: int
    empty_cols_before: int
    empty_cols_after: int

    # Structural row/col indicators (defaults using field)
    candidate_header_rows: List[int] = field(default_factory=list)
    candidate_header_cols: List[int] = field(default_factory=list)
    columns_info: List[ColumnStructureInfo] = field(default_factory=list)

    # Cross-column structural patterns
    shared_x_col_candidates: List[int] = field(default_factory=list)
    repeated_col_patterns: List[RepeatedColumnPattern] = field(default_factory=list)
    neighbor_region_ids: List[str] = field(default_factory=list)

    # Formatting structural signals
    merged_cell_count: int = 0
    bold_cell_count: int = 0

    # Raw preserved text (uninterpreted verbatim strings from candidate header/banner rows)
    raw_header_texts: List[str] = field(default_factory=list)
    raw_banner_texts: List[str] = field(default_factory=list)

    def contains_cell(self, row: int, col: int) -> bool:
        """Return True if (row, col) is within this region's bounding box."""
        return self.row_start <= row <= self.row_end and self.col_start <= col <= self.col_end
