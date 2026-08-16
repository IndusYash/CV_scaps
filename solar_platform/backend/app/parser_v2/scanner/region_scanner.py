"""
parser_v2.scanner.region_scanner
--------------------------------
Discovers CandidateRegion objects from a RawSheet and extracts pure structural evidence.

Contracts:
  - Operates ONLY on RawSheet / RawCell structures.
  - Zero scientific domain logic.
  - Identifies structural regions, grid geometries, column types, monotonicity,
    repeated column groups, and neighboring relationships.
  - Preserves verbatim header/banner raw strings as uninterpreted text evidence.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.reader.raw_model import RawCell, RawCellType, RawSheet
from app.parser_v2.scanner.boundary_detector import detect_rectangular_bounds
from app.parser_v2.scanner.region import (
    CandidateRegion,
    ColumnStructureInfo,
    RepeatedColumnPattern,
)


def _analyze_column_structure(
    sheet: RawSheet,
    r_min: int,
    r_max: int,
    col_idx: int,
    region_col_idx: int,
) -> ColumnStructureInfo:
    """Analyze pure cell type distributions and numerical monotonicity for one column."""
    total = r_max - r_min + 1
    num_cnt = 0
    str_cnt = 0
    fmt_cnt = 0
    emp_cnt = 0

    numeric_vals: List[float] = []

    for r in range(r_min, r_max + 1):
        cell = sheet.get(r, col_idx)
        if cell is None or cell.data_type == RawCellType.EMPTY:
            emp_cnt += 1
        elif cell.data_type == RawCellType.NUMERIC:
            num_cnt += 1
            if isinstance(cell.raw_value, (int, float)):
                numeric_vals.append(float(cell.raw_value))
        elif cell.data_type == RawCellType.FORMULA:
            fmt_cnt += 1
            if isinstance(cell.raw_value, (int, float)):
                num_cnt += 1
                numeric_vals.append(float(cell.raw_value))
        elif cell.data_type == RawCellType.STRING:
            str_cnt += 1
        else:
            str_cnt += 1

    num_density = num_cnt / total if total > 0 else 0.0
    str_density = str_cnt / total if total > 0 else 0.0

    # Dominant type
    if num_density >= 0.70:
        dom_type = "numeric"
    elif str_density >= 0.50:
        dom_type = "string"
    elif emp_cnt == total:
        dom_type = "empty"
    else:
        dom_type = "mixed"

    # Monotonicity check on numeric values
    is_inc = False
    is_dec = False
    if len(numeric_vals) >= 3:
        diffs = [numeric_vals[i + 1] - numeric_vals[i] for i in range(len(numeric_vals) - 1)]
        is_inc = all(d >= 0 for d in diffs) and any(d > 0 for d in diffs)
        is_dec = all(d <= 0 for d in diffs) and any(d < 0 for d in diffs)

    distinct_count = len(set(numeric_vals)) if numeric_vals else len({sheet.get(r, col_idx).display_str for r in range(r_min, r_max + 1) if sheet.get(r, col_idx)})
    val_range = (min(numeric_vals), max(numeric_vals)) if numeric_vals else None

    return ColumnStructureInfo(
        col_index=col_idx,
        region_col_index=region_col_idx,
        total_cells=total,
        numeric_count=num_cnt,
        string_count=str_cnt,
        formula_count=fmt_cnt,
        empty_count=emp_cnt,
        numeric_density=num_density,
        string_density=str_density,
        dominant_type=dom_type,
        is_monotonic_increasing=is_inc,
        is_monotonic_decreasing=is_dec,
        distinct_value_count=distinct_count,
        value_range=val_range,
    )


def _detect_repeated_column_patterns(cols_info: List[ColumnStructureInfo]) -> List[RepeatedColumnPattern]:
    """Find contiguous runs of 2+ numeric columns with high numeric density."""
    patterns: List[RepeatedColumnPattern] = []
    run_start = None

    for i, col in enumerate(cols_info):
        is_num = col.dominant_type == "numeric" and col.numeric_density >= 0.60
        if is_num and run_start is None:
            run_start = i
        elif not is_num and run_start is not None:
            run_len = i - run_start
            if run_len >= 2:
                start_c = cols_info[run_start].col_index
                end_c = cols_info[i - 1].col_index
                avg_dens = sum(cols_info[j].numeric_density for j in range(run_start, i)) / run_len
                patterns.append(RepeatedColumnPattern(
                    pattern_type="uniform_numeric_group",
                    start_col=start_c,
                    end_col=end_c,
                    num_columns=run_len,
                    column_row_length=cols_info[run_start].total_cells,
                    numeric_density=avg_dens,
                ))
            run_start = None

    if run_start is not None:
        run_len = len(cols_info) - run_start
        if run_len >= 2:
            start_c = cols_info[run_start].col_index
            end_c = cols_info[-1].col_index
            avg_dens = sum(cols_info[j].numeric_density for j in range(run_start, len(cols_info))) / run_len
            patterns.append(RepeatedColumnPattern(
                pattern_type="uniform_numeric_group",
                start_col=start_c,
                end_col=end_c,
                num_columns=run_len,
                column_row_length=cols_info[run_start].total_cells,
                numeric_density=avg_dens,
            ))

    return patterns


def scan_sheet(sheet: RawSheet) -> List[CandidateRegion]:
    """
    Scans a RawSheet to discover CandidateRegion objects with full structural metadata.

    Parameters
    ----------
    sheet : RawSheet
        The sheet returned by Stage 1 read_workbook().

    Returns
    -------
    List[CandidateRegion]
        Discovered rectangular regions with pure structural properties.
    """
    bounds = detect_rectangular_bounds(sheet)
    if not bounds:
        return []

    regions: List[CandidateRegion] = []

    for reg_idx, (r_min, r_max, c_min, c_max) in enumerate(bounds):
        n_rows = r_max - r_min + 1
        n_cols = c_max - c_min + 1
        total_cells = n_rows * n_cols

        # Per-cell type counts across region
        non_empty = 0
        num_cnt = 0
        str_cnt = 0
        fmt_cnt = 0
        emp_cnt = 0
        bold_cnt = 0
        merged_cnt = 0

        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                cell = sheet.get(r, c)
                if cell is None or cell.data_type == RawCellType.EMPTY:
                    emp_cnt += 1
                else:
                    non_empty += 1
                    if cell.data_type == RawCellType.NUMERIC:
                        num_cnt += 1
                    elif cell.data_type == RawCellType.STRING:
                        str_cnt += 1
                    elif cell.data_type == RawCellType.FORMULA:
                        fmt_cnt += 1
                        if isinstance(cell.raw_value, (int, float)):
                            num_cnt += 1

                    if cell.is_bold:
                        bold_cnt += 1
                    if cell.is_merged:
                        merged_cnt += 1

        non_empty_dens = non_empty / total_cells if total_cells > 0 else 0.0
        num_dens = num_cnt / total_cells if total_cells > 0 else 0.0
        str_dens = str_cnt / total_cells if total_cells > 0 else 0.0
        fmt_dens = fmt_cnt / total_cells if total_cells > 0 else 0.0
        emp_dens = emp_cnt / total_cells if total_cells > 0 else 0.0

        # Empty boundaries context
        rows_before = r_min
        rows_after = max(0, sheet.n_rows - 1 - r_max)
        cols_before = c_min
        cols_after = max(0, sheet.n_cols - 1 - c_max)

        # Analyze individual columns
        cols_info: List[ColumnStructureInfo] = []
        shared_x_candidates: List[int] = []

        for offset, c in enumerate(range(c_min, c_max + 1)):
            col_struct = _analyze_column_structure(sheet, r_min, r_max, c, offset)
            cols_info.append(col_struct)

            # Monotonic numeric column with high density -> shared index/x-column candidate
            if col_struct.numeric_density >= 0.70 and (col_struct.is_monotonic_increasing or col_struct.is_monotonic_decreasing):
                shared_x_candidates.append(c)

        # Detect candidate header rows (rows in upper portion where strings dominate)
        cand_header_rows: List[int] = []
        raw_headers: List[str] = []

        for r in range(r_min, min(r_max + 1, r_min + 5)):
            r_str_cnt = sum(1 for c in range(c_min, c_max + 1)
                            if sheet.get(r, c) and sheet.get(r, c).data_type in (RawCellType.STRING, RawCellType.FORMULA))
            if r_str_cnt > 0:
                cand_header_rows.append(r)
                row_texts = [sheet.get(r, c).display_str for c in range(c_min, c_max + 1)
                             if sheet.get(r, c) and sheet.get(r, c).display_str]
                raw_headers.extend(row_texts)

        # Scan banner rows directly above r_min (up to 5 rows above) for uninterpreted text
        raw_banners: List[str] = []
        banner_scan_start = max(0, r_min - 5)
        for r in range(banner_scan_start, r_min):
            for c in range(c_min, c_max + 1):
                cell = sheet.get(r, c)
                if cell and cell.display_str and cell.display_str.strip():
                    raw_banners.append(cell.display_str.strip())

        repeated_patterns = _detect_repeated_column_patterns(cols_info)

        reg = CandidateRegion(
            region_id=f"region_{reg_idx}",
            sheet_name=sheet.sheet_name,
            row_start=r_min,
            row_end=r_max,
            col_start=c_min,
            col_end=c_max,
            n_rows=n_rows,
            n_cols=n_cols,
            total_cells=total_cells,
            non_empty_cells=non_empty,
            non_empty_density=non_empty_dens,
            numeric_density=num_dens,
            string_density=str_dens,
            formula_density=fmt_dens,
            empty_density=emp_dens,
            empty_rows_before=rows_before,
            empty_rows_after=rows_after,
            empty_cols_before=cols_before,
            empty_cols_after=cols_after,
            candidate_header_rows=cand_header_rows,
            candidate_header_cols=[c_min],
            columns_info=cols_info,
            shared_x_col_candidates=shared_x_candidates,
            repeated_col_patterns=repeated_patterns,
            neighbor_region_ids=[],
            merged_cell_count=merged_cnt,
            bold_cell_count=bold_cnt,
            raw_header_texts=raw_headers,
            raw_banner_texts=raw_banners,
        )
        regions.append(reg)

    # Establish neighbor relationships between regions
    for i, r1 in enumerate(regions):
        for j, r2 in enumerate(regions):
            if i == j:
                continue
            # Horizontally or vertically adjacent within 3 rows/cols
            h_adj = (abs(r1.col_end - r2.col_start) <= 3 or abs(r2.col_end - r1.col_start) <= 3) and max(r1.row_start, r2.row_start) <= min(r1.row_end, r2.row_end)
            v_adj = (abs(r1.row_end - r2.row_start) <= 3 or abs(r2.row_end - r1.row_start) <= 3) and max(r1.col_start, r2.col_start) <= min(r1.col_end, r2.col_end)
            if h_adj or v_adj:
                r1.neighbor_region_ids.append(r2.region_id)

    return regions
