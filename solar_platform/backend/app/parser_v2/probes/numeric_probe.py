"""
parser_v2.probes.numeric_probe
------------------------------
Stateless Probe 3: Numeric Probe.

Characterizes numerical properties of a CandidateRegion:
  - Overall numeric density and completeness
  - Value ranges (min, max, mean, std) across all numeric data cells
  - Formula-derived cell counts and distributions
  - Missingness and empty cell ratios

Does NOT impose domain/physical limits. Returns EvidenceItem objects with CellRef provenance.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.reader.raw_model import RawSheet, RawCellType
from app.parser_v2.scanner.region import CandidateRegion
from app.parser_v2.probes.evidence import CellRef, EvidenceItem, EvidencePolarity


PROBE_ID = "NumericProbe_v2.0"


def _make_cell_ref(sheet: RawSheet, r: int, c: int) -> CellRef:
    cell = sheet.get(r, c)
    if cell is None:
        return CellRef(sheet.sheet_name, r, c, None, "")
    return CellRef(sheet.sheet_name, r, c, cell.raw_value, cell.display_str)


def probe_numerics(region: CandidateRegion, sheet: RawSheet) -> List[EvidenceItem]:
    """
    Stateless evaluation of numeric cell distributions, ranges, and formulas.
    """
    items: List[EvidenceItem] = []

    all_numeric_vals: List[float] = []
    numeric_refs: List[CellRef] = []
    formula_refs: List[CellRef] = []
    empty_refs: List[CellRef] = []

    for r in range(region.row_start, region.row_end + 1):
        for c in range(region.col_start, region.col_end + 1):
            cell = sheet.get(r, c)
            if cell is None or cell.data_type == RawCellType.EMPTY:
                if len(empty_refs) < 20:
                    empty_refs.append(_make_cell_ref(sheet, r, c))
            elif cell.data_type == RawCellType.FORMULA:
                ref = _make_cell_ref(sheet, r, c)
                formula_refs.append(ref)
                if isinstance(cell.raw_value, (int, float)) and not math.isnan(float(cell.raw_value)):
                    all_numeric_vals.append(float(cell.raw_value))
                    numeric_refs.append(ref)
            elif cell.data_type == RawCellType.NUMERIC:
                if isinstance(cell.raw_value, (int, float)) and not math.isnan(float(cell.raw_value)):
                    ref = _make_cell_ref(sheet, r, c)
                    all_numeric_vals.append(float(cell.raw_value))
                    numeric_refs.append(ref)

    # 1. Density evidence
    if region.numeric_density >= 0.70:
        items.append(EvidenceItem(
            name="numeric_density_high",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.POSITIVE,
            confidence=0.90,
            value={
                "numeric_density": region.numeric_density,
                "total_numeric_cells": len(all_numeric_vals),
            },
            rationale=f"Region exhibits high numeric density ({region.numeric_density:.2%}).",
            source_cells=numeric_refs[:20],
        ))
    elif region.numeric_density <= 0.10:
        items.append(EvidenceItem(
            name="numeric_density_low",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.NEGATIVE,
            confidence=0.85,
            value={
                "numeric_density": region.numeric_density,
                "total_numeric_cells": len(all_numeric_vals),
            },
            rationale=f"Region exhibits low numeric density ({region.numeric_density:.2%}).",
            source_cells=numeric_refs[:20],
        ))
    else:
        items.append(EvidenceItem(
            name="numeric_density_mixed",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.UNKNOWN,
            confidence=0.50,
            value={
                "numeric_density": region.numeric_density,
                "total_numeric_cells": len(all_numeric_vals),
            },
            rationale=f"Region exhibits mixed numeric density ({region.numeric_density:.2%}).",
            source_cells=numeric_refs[:20],
        ))

    # 2. Value range summary
    if all_numeric_vals:
        min_v = min(all_numeric_vals)
        max_v = max(all_numeric_vals)
        mean_v = sum(all_numeric_vals) / len(all_numeric_vals)

        items.append(EvidenceItem(
            name="numeric_value_range",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.POSITIVE,
            confidence=0.95,
            value={
                "min": min_v,
                "max": max_v,
                "mean": mean_v,
                "count": len(all_numeric_vals),
            },
            rationale=f"Numeric data values range from {min_v} to {max_v} across {len(all_numeric_vals)} cells.",
            source_cells=numeric_refs[:10],
        ))

    # 3. Formula presence
    if formula_refs:
        items.append(EvidenceItem(
            name="formula_derived_cells",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.POSITIVE,
            confidence=0.80,
            value={
                "formula_count": len(formula_refs),
                "formula_density": len(formula_refs) / region.total_cells if region.total_cells > 0 else 0.0,
            },
            rationale=f"Region contains {len(formula_refs)} formula-derived cells.",
            source_cells=formula_refs[:20],
        ))

    return items
