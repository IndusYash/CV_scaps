"""
parser_v2.probes.axis_probe
---------------------------
Stateless Probe 1: Axis Probe.

Evaluates candidate independent-axis columns in a CandidateRegion.
Extracts evidence on:
  - Candidate x-column indices
  - Monotonicity (increasing/decreasing/non-monotonic)
  - Value ranges and point counts
  - Header text and units associated with candidate axis columns

Does NOT classify experiments. Returns EvidenceItem objects with full CellRef provenance.
"""
from __future__ import annotations

from typing import List, Optional

from app.parser_v2.reader.raw_model import RawSheet, RawCellType
from app.parser_v2.scanner.region import CandidateRegion
from app.parser_v2.probes.evidence import CellRef, EvidenceItem, EvidencePolarity


PROBE_ID = "AxisProbe_v2.0"


def _make_cell_ref(sheet: RawSheet, r: int, c: int) -> CellRef:
    cell = sheet.get(r, c)
    if cell is None:
        return CellRef(sheet.sheet_name, r, c, None, "")
    return CellRef(sheet.sheet_name, r, c, cell.raw_value, cell.display_str)


def probe_axis(region: CandidateRegion, sheet: RawSheet) -> List[EvidenceItem]:
    """
    Stateless evaluation of candidate independent-axis columns in a region.
    """
    items: List[EvidenceItem] = []

    if not region.columns_info:
        items.append(EvidenceItem(
            name="axis_candidate_presence",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.NEGATIVE,
            confidence=1.0,
            value=None,
            rationale="CandidateRegion contains no column structure information.",
            source_cells=[],
        ))
        return items

    # Check candidate x-columns (shared_x_col_candidates or region col 0)
    cand_cols = list(region.shared_x_col_candidates)
    if not cand_cols and region.columns_info:
        cand_cols = [region.columns_info[0].col_index]

    for col_idx in cand_cols:
        col_info = next((c for c in region.columns_info if c.col_index == col_idx), None)
        if not col_info:
            continue

        source_cells: List[CellRef] = []

        # Collect cell references for headers in candidate row
        for r_hdr in region.candidate_header_rows:
            source_cells.append(_make_cell_ref(sheet, r_hdr, col_idx))

        # Collect cell references for numeric values in candidate axis column
        numeric_refs: List[CellRef] = []
        numeric_vals: List[float] = []

        for r in range(region.row_start, region.row_end + 1):
            cell = sheet.get(r, col_idx)
            if cell and cell.data_type in (RawCellType.NUMERIC, RawCellType.FORMULA) and isinstance(cell.raw_value, (int, float)):
                ref = _make_cell_ref(sheet, r, col_idx)
                numeric_refs.append(ref)
                numeric_vals.append(float(cell.raw_value))

        source_cells.extend(numeric_refs[:10])  # Cap provenance list for summary items

        # Header association evidence
        hdr_text = ""
        for r_hdr in region.candidate_header_rows:
            c_hdr = sheet.get(r_hdr, col_idx)
            if c_hdr and c_hdr.display_str and c_hdr.display_str.strip():
                hdr_text = c_hdr.display_str.strip()
                break

        if col_info.dominant_type == "numeric" and len(numeric_vals) >= 3:
            # Monotonicity evidence
            if col_info.is_monotonic_increasing or col_info.is_monotonic_decreasing:
                direction = "increasing" if col_info.is_monotonic_increasing else "decreasing"
                items.append(EvidenceItem(
                    name="axis_monotonicity",
                    probe_id=PROBE_ID,
                    polarity=EvidencePolarity.POSITIVE,
                    confidence=0.90,
                    value={
                        "col_index": col_idx,
                        "direction": direction,
                        "n_points": len(numeric_vals),
                        "range": col_info.value_range,
                        "header_text": hdr_text,
                    },
                    rationale=f"Column {col_idx} forms a strictly monotonic {direction} numeric grid of {len(numeric_vals)} points in range {col_info.value_range}.",
                    source_cells=source_cells,
                ))
            else:
                items.append(EvidenceItem(
                    name="axis_monotonicity",
                    probe_id=PROBE_ID,
                    polarity=EvidencePolarity.NEGATIVE,
                    confidence=0.70,
                    value={
                        "col_index": col_idx,
                        "n_points": len(numeric_vals),
                        "range": col_info.value_range,
                        "header_text": hdr_text,
                    },
                    rationale=f"Column {col_idx} is numeric but not strictly monotonic.",
                    source_cells=source_cells,
                ))
        elif col_info.dominant_type == "string":
            items.append(EvidenceItem(
                name="axis_non_numeric_category",
                probe_id=PROBE_ID,
                polarity=EvidencePolarity.NEGATIVE,
                confidence=0.85,
                value={
                    "col_index": col_idx,
                    "dominant_type": "string",
                    "distinct_count": col_info.distinct_value_count,
                    "header_text": hdr_text,
                },
                rationale=f"Column {col_idx} consists of string category labels, not a numeric grid.",
                source_cells=source_cells,
            ))
        else:
            items.append(EvidenceItem(
                name="axis_candidate_presence",
                probe_id=PROBE_ID,
                polarity=EvidencePolarity.UNKNOWN,
                confidence=0.40,
                value={"col_index": col_idx, "numeric_density": col_info.numeric_density},
                rationale=f"Column {col_idx} has insufficient numeric points for axis evaluation.",
                source_cells=source_cells,
            ))

        # Check for generic stacked axis reset sequences (multi-block stacked curve grids)
        stacked_item = _detect_stacked_axis_resets(sheet, region, col_idx)
        if stacked_item is not None:
            items.append(stacked_item)

    return items


def _detect_stacked_axis_resets(
    sheet: RawSheet,
    region: CandidateRegion,
    col_idx: int,
) -> Optional[EvidenceItem]:
    """
    Detect generic stacked curve blocks from monotonic x-axis resets.

    Requires multi-signal structural evidence:
      - At least 2 distinct monotonic x-axis segments
      - Each segment must have a valid monotonic sequence before and after the reset (len >= 3)
      - Segment lengths must exhibit structural consistency across blocks
    """
    numeric_rows: List[Tuple[int, float]] = []
    for r in range(region.row_start, region.row_end + 1):
        cell = sheet.get(r, col_idx)
        if cell and cell.data_type in (RawCellType.NUMERIC, RawCellType.FORMULA) and isinstance(cell.raw_value, (int, float)):
            numeric_rows.append((r, float(cell.raw_value)))

    if len(numeric_rows) < 6:
        return None

    # Partition numeric rows into monotonic segments based on resets
    segments: List[List[Tuple[int, float]]] = []
    curr_seg: List[Tuple[int, float]] = [numeric_rows[0]]

    for i in range(1, len(numeric_rows)):
        r_prev, v_prev = numeric_rows[i - 1]
        r_curr, v_curr = numeric_rows[i]

        # Reset condition: value resets from max back to min (increasing grid) or min back to max (decreasing grid)
        # Also check row gap (> 35 rows difference indicates text banner between blocks)
        is_val_reset = (v_curr < v_prev if numeric_rows[0][1] <= numeric_rows[-1][1] else v_curr > v_prev)
        is_row_gap = (r_curr - r_prev > 25)

        if is_val_reset or is_row_gap:
            if len(curr_seg) >= 3:
                segments.append(curr_seg)
            curr_seg = [(r_curr, v_curr)]
        else:
            curr_seg.append((r_curr, v_curr))

    if len(curr_seg) >= 3:
        segments.append(curr_seg)

    if len(segments) < 2:
        return None

    # Verify structural consistency across segments:
    # 1. Each segment must be strictly monotonic (len >= 3)
    valid_blocks: List[Tuple[int, int]] = []
    seg_lengths: List[int] = []

    for seg in segments:
        if len(seg) < 3:
            continue
        vals = [v for _, v in seg]
        is_inc = all(vals[k] <= vals[k + 1] for k in range(len(vals) - 1))
        is_dec = all(vals[k] >= vals[k + 1] for k in range(len(vals) - 1))
        if is_inc or is_dec:
            valid_blocks.append((seg[0][0], seg[-1][0]))
            seg_lengths.append(len(seg))

    if len(valid_blocks) < 2:
        return None

    # Structural consistency check: ratio of median length to min/max
    median_len = sorted(seg_lengths)[len(seg_lengths) // 2]
    # Filter out severely malformed blocks (< 20% of median length)
    filtered_blocks = [b for b, l in zip(valid_blocks, seg_lengths) if l >= max(3, int(0.20 * median_len))]

    if len(filtered_blocks) < 2:
        return None

    # Build provenance cell references for boundary resets
    source_cells = [_make_cell_ref(sheet, b[0], col_idx) for b in filtered_blocks[:10]]

    return EvidenceItem(
        name="stacked_axis_reset_sequence",
        probe_id=PROBE_ID,
        polarity=EvidencePolarity.POSITIVE,
        confidence=0.92,
        value={
            "col_index": col_idx,
            "n_blocks": len(filtered_blocks),
            "block_row_ranges": filtered_blocks,
            "segment_lengths": [b[1] - b[0] + 1 for b in filtered_blocks],
            "median_length": median_len,
        },
        rationale=f"Column {col_idx} contains {len(filtered_blocks)} repeated monotonic x-axis segments (median length {median_len} rows).",
        source_cells=source_cells,
    )
