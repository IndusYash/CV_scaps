"""
parser_v2.probes.repetition_probe
---------------------------------
Stateless Probe 5: Repetition Probe.

Identifies repeated trace structures, shared x-grids, and trace counts in a CandidateRegion:
  - Contiguous numeric column groups (trace count)
  - Consistency of row lengths across traces
  - Shared x-column relationship with adjacent y-columns

Does NOT classify experiments. Returns EvidenceItem objects with CellRef provenance.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.reader.raw_model import RawSheet, RawCellType
from app.parser_v2.scanner.region import CandidateRegion
from app.parser_v2.probes.evidence import CellRef, EvidenceItem, EvidencePolarity


PROBE_ID = "RepetitionProbe_v2.0"


def _make_cell_ref(sheet: RawSheet, r: int, c: int) -> CellRef:
    cell = sheet.get(r, c)
    if cell is None:
        return CellRef(sheet.sheet_name, r, c, None, "")
    return CellRef(sheet.sheet_name, r, c, cell.raw_value, cell.display_str)


def probe_repetition(region: CandidateRegion, sheet: RawSheet) -> List[EvidenceItem]:
    """
    Stateless evaluation of trace repetition, shared grid geometry, and column count.
    """
    items: List[EvidenceItem] = []

    if not region.columns_info:
        return items

    # Identify primary candidate x-column (first shared x candidate, or col_start)
    if region.shared_x_col_candidates:
        primary_x = region.shared_x_col_candidates[0]
    else:
        primary_x = region.col_start

    # Y-columns are all numeric columns excluding primary x-column
    y_cols = [col for col in region.columns_info if col.col_index != primary_x and col.dominant_type == "numeric"]

    trace_count = len(y_cols)
    source_cells: List[CellRef] = []

    for col in y_cols:
        # Include first data row cell ref for each trace column
        ref = _make_cell_ref(sheet, region.row_start, col.col_index)
        source_cells.append(ref)

    if trace_count >= 1:
        # Check row length consistency across traces
        row_lengths = [col.numeric_count for col in y_cols]
        is_uniform_length = len(set(row_lengths)) == 1

        items.append(EvidenceItem(
            name="repeated_trace_group",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.POSITIVE,
            confidence=0.90 if is_uniform_length else 0.70,
            value={
                "trace_count": trace_count,
                "primary_x_col": primary_x,
                "is_uniform_length": is_uniform_length,
                "row_lengths": row_lengths,
                "shared_x_cols": list(region.shared_x_col_candidates),
                "y_col_indices": [col.col_index for col in y_cols],
            },
            rationale=f"Discovered group of {trace_count} numeric trace columns associated with primary x col {primary_x} (uniform row length={is_uniform_length}).",
            source_cells=source_cells,
        ))
    else:
        items.append(EvidenceItem(
            name="repeated_trace_group",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.NEGATIVE,
            confidence=0.80,
            value={"trace_count": 0},
            rationale="No repeated numeric trace columns discovered in region.",
            source_cells=[],
        ))

    # Pattern evidence from scanner
    if region.repeated_col_patterns:
        for pat in region.repeated_col_patterns:
            items.append(EvidenceItem(
                name="uniform_column_pattern",
                probe_id=PROBE_ID,
                polarity=EvidencePolarity.POSITIVE,
                confidence=0.85,
                value={
                    "pattern_type": pat.pattern_type,
                    "start_col": pat.start_col,
                    "end_col": pat.end_col,
                },
                rationale=f"Scanner discovered repeated column pattern {pat.pattern_type} in cols {pat.start_col}..{pat.end_col}.",
                source_cells=source_cells,
            ))

    # Generic interleaved independent (axis, target) column group detection
    interleaved_item = _detect_interleaved_groups(region, sheet)
    if interleaved_item is not None:
        items.append(interleaved_item)

    return items


def _detect_interleaved_groups(
    region: CandidateRegion,
    sheet: RawSheet,
) -> Optional[EvidenceItem]:
    """
    Detect repeated independent (axis, target) column groups across region columns.

    Supports:
      - 2-col groups: [Axis | Target] [Axis | Target] ...
      - 3-col groups: [Param | Axis | Target] [Param | Axis | Target] ...
      - Multi-target groups: [Axis | Target1 | Target2] ...
    Requires at least 2 independent verified monotonic axis columns.
    """
    if not region.columns_info or len(region.columns_info) < 4:
        return None

    pairs: List[Tuple[int, int]] = []
    groups: List[Dict[str, Any]] = []
    used_cols = set()

    for col_info in region.columns_info:
        c_idx = col_info.col_index
        if c_idx in used_cols:
            continue

        if col_info.numeric_density >= 0.40 and (col_info.is_monotonic_increasing or col_info.is_monotonic_decreasing):
            # Header check: target concept headers (j1, j2, jtot, current, qe, responsivity) are not x-axes
            cell_hdr = sheet.get(region.row_start, c_idx) or sheet.get(max(0, region.row_start - 1), c_idx)
            hdr_str = (cell_hdr.display_str or "").lower() if cell_hdr else ""
            is_target_hdr = bool(re.search(r"\bj\d*[\(\_\s]|\bcurrent\b|\bq\.?e\.?\b|\bresponsivity\b|\bdetectivity\b", hdr_str))
            if is_target_hdr:
                continue

            # If pairs already has a primary x_col, subsequent x_cols must be axis concept OR have similar numeric range
            if pairs:
                is_axis_hdr = any(k in hdr_str for k in ["v(v)", "v (v)", "v[v]", "voltage", "lambda", "wavelength", "x(um)", "position"])
                if not is_axis_hdr:
                    primary_x_info = next(c for c in region.columns_info if c.col_index == pairs[0][0])
                    if primary_x_info.value_range and col_info.value_range:
                        min_0, max_0 = primary_x_info.value_range
                        min_c, max_c = col_info.value_range
                        if abs(min_c - min_0) > 5.0 or abs(max_c - max_0) > 5.0:
                            continue

            # Look for adjacent numeric target column (c_idx + 1)
            next_cand = [
                c.col_index for c in region.columns_info
                if c.col_index > c_idx and c.col_index not in used_cols and c.numeric_density >= 0.30
            ]
            if next_cand:
                y_col = next_cand[0]
                pairs.append((c_idx, y_col))
                used_cols.add(c_idx)
                used_cols.add(y_col)

                param_col = (c_idx - 1) if (c_idx - 1) >= region.col_start and (c_idx - 1) not in used_cols else None
                groups.append({"param_col": param_col, "x_col": c_idx, "y_cols": [y_col]})

    if len(pairs) < 2:
        return None

    x_cols = [p[0] for p in pairs]
    y_cols_all = [p[1] for p in pairs]

    strides = [x_cols[i] - x_cols[i - 1] for i in range(1, len(x_cols))]
    is_uniform_stride = len(set(strides)) == 1
    stride_val = strides[0] if is_uniform_stride else None

    source_cells = []
    for p in pairs:
        source_cells.append(_make_cell_ref(sheet, region.row_start, p[0]))
        source_cells.append(_make_cell_ref(sheet, region.row_start, p[1]))

    return EvidenceItem(
        name="interleaved_column_group",
        probe_id=PROBE_ID,
        polarity=EvidencePolarity.POSITIVE,
        confidence=0.92 if is_uniform_stride else 0.75,
        value={
            "n_curves": len(pairs),
            "pairs": pairs,
            "groups": groups,
            "x_cols": x_cols,
            "y_cols": y_cols_all,
            "stride": stride_val,
            "is_uniform_stride": is_uniform_stride,
        },
        rationale=f"Region contains {len(pairs)} repeated independent (axis, target) column pairs across columns (x_cols={x_cols}).",
        source_cells=source_cells,
    )
