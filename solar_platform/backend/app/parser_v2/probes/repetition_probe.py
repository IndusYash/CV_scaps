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
                    "num_columns": pat.num_columns,
                    "numeric_density": pat.numeric_density,
                },
                rationale=f"Discovered contiguous {pat.pattern_type} of {pat.num_columns} columns from col {pat.start_col} to {pat.end_col}.",
                source_cells=source_cells[:pat.num_columns],
            ))

    return items
