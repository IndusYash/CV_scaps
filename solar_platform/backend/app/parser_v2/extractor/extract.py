"""
parser_v2.extractor.extract
----------------------------
Stage 5 top-level extraction entry points.

``extract_region()`` is the single public function.  It:
1. Reads the winning structural hypothesis from RegionInterpretation.
2. Calls the matching structure-specific extractor.
3. Populates RawRegionData unconditionally.
4. Assembles and returns an ExtractedDataset.

Ambiguity contract:
  - ACCEPTED:   is_ambiguous=False, all extracted fields populated normally.
  - UNCERTAIN:  is_ambiguous=True,  extraction runs but ambiguity_note is set.
  - UNRESOLVED: is_ambiguous=True,  extraction is attempted but may produce
                empty traces/rows.  raw is ALWAYS populated.

The extractor never re-classifies or re-applies scientific keyword logic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.parser_v2.extractor.ir import (
    AxisData,
    CategoryRow,
    ExtractedDataset,
    RawRegionData,
    ScalarRow,
    TraceData,
)
from app.parser_v2.extractor.category_table_extractor import extract_category_table
from app.parser_v2.extractor.numeric_curve_grid_extractor import extract_numeric_curve_grid
from app.parser_v2.extractor.scalar_sweep_table_extractor import extract_scalar_sweep_table
from app.parser_v2.probes.evidence import CellRef
from app.parser_v2.reader.raw_model import RawSheet
from app.parser_v2.reasoner.hypothesis import (
    AcceptanceState,
    RegionInterpretation,
    StructuralHypothesis,
)
from app.parser_v2.scanner.region import CandidateRegion


def _build_raw(
    sheet: RawSheet,
    region: CandidateRegion,
) -> RawRegionData:
    """Slice the verbatim cell grid for the region's bounding box."""
    cells = []
    for r in range(region.row_start, region.row_end + 1):
        row = []
        for c in range(region.col_start, region.col_end + 1):
            cell = sheet.get(r, c)
            if cell is not None:
                row.append(cell)
            else:
                # Synthesise a placeholder (should never happen in well-formed RawSheet)
                from app.parser_v2.reader.raw_model import RawCell, RawCellType
                row.append(RawCell(
                    row=r, col=c,
                    data_type=RawCellType.EMPTY,
                    raw_value=None,
                    display_str="",
                    formula_str=None,
                    is_bold=False,
                    background_color=None,
                    is_merged=False,
                    merge_parent=None,
                ))
        cells.append(row)

    return RawRegionData(
        region_id=region.region_id,
        sheet_name=region.sheet_name,
        row_start=region.row_start,
        col_start=region.col_start,
        row_end=region.row_end,
        col_end=region.col_end,
        cells=cells,
    )


def _build_hypothesis_scores(interpretation: RegionInterpretation) -> List[Dict[str, Any]]:
    """
    Preserve all hypothesis evidence-support scores from the ranked candidate set.
    Scores are evidence-support scores (NOT probabilities).
    """
    return [
        {
            "structure": c.structure.value,
            "scientific_label": c.scientific_label,
            "evidence_support_score": c.score,  # explicitly named, not "probability"
            "rationale": c.rationale,
        }
        for c in interpretation.all_candidates
    ]


def _collect_provenance(interpretation: RegionInterpretation) -> List[CellRef]:
    return list(interpretation.provenance_cells)


def extract_region(
    sheet: RawSheet,
    region: CandidateRegion,
    interpretation: RegionInterpretation,
) -> ExtractedDataset:
    """
    Extract data from one CandidateRegion given its RegionInterpretation.

    Returns an ExtractedDataset containing:
    - Extracted typed data (axis/traces, scalar_rows, or category_rows)
    - Raw verbatim cell grid (always)
    - Full hypothesis score ranking (always)
    - Ambiguity flags (is_ambiguous, ambiguity_note)

    This function is the only entry point to Stage 5.
    It does NOT re-classify or override the Reasoner's decisions.
    """
    winner = interpretation.winner
    acceptance_state = interpretation.acceptance_state

    # Determine ambiguity
    is_ambiguous = acceptance_state != AcceptanceState.ACCEPTED
    if acceptance_state == AcceptanceState.UNCERTAIN:
        ambiguity_note = (
            f"Interpretation is UNCERTAIN (evidence-support score gap = "
            f"{interpretation.winner_runner_up_gap:.2f}).  "
            f"Winner: {winner.structure.value if winner else 'None'}.  "
            f"Runner-up: {interpretation.runner_up.structure.value if interpretation.runner_up else 'None'}."
        )
    elif acceptance_state == AcceptanceState.UNRESOLVED:
        ambiguity_note = (
            f"Interpretation is UNRESOLVED (evidence-support score gap = "
            f"{interpretation.winner_runner_up_gap:.2f}).  "
            f"Best available structural hypothesis: "
            f"{winner.structure.value if winner else 'None'}.  "
            "Extraction attempted; data may be incomplete."
        )
    else:
        ambiguity_note = ""

    # Always populate raw verbatim data
    raw = _build_raw(sheet, region)

    # Initialise output containers
    axis: Optional[AxisData] = None
    traces: List[TraceData] = []
    scalar_rows: List[ScalarRow] = []
    category_rows: List[CategoryRow] = []

    # Dispatch to structure-specific extractor based on Reasoner's winner
    if winner is not None:
        structure = winner.structure

        if structure == StructuralHypothesis.NUMERIC_CURVE_GRID:
            axis, traces = extract_numeric_curve_grid(sheet, region, interpretation)

        elif structure == StructuralHypothesis.SCALAR_SWEEP_TABLE:
            scalar_rows = extract_scalar_sweep_table(sheet, region, interpretation)

        elif structure == StructuralHypothesis.CATEGORY_TABLE:
            category_rows = extract_category_table(sheet, region, interpretation)

        elif structure == StructuralHypothesis.UNKNOWN:
            # UNRESOLVED: raw is populated; extraction results are empty
            ambiguity_note = (
                "Structural hypothesis is Unknown.  "
                "Raw data preserved; no typed extraction performed."
            )

    return ExtractedDataset(
        region_id=region.region_id,
        sheet_name=region.sheet_name,

        # Interpretation metadata — taken directly from Reasoner; not re-derived
        structure_type=winner.structure.value if winner else StructuralHypothesis.UNKNOWN.value,
        scientific_label=winner.scientific_label if winner else None,
        axis_concept=winner.axis_concept if winner else None,
        target_concept=winner.target_concept if winner else None,
        acceptance_state=acceptance_state.value,
        winner_evidence_score=winner.score if winner else 0.0,
        winner_runner_up_gap=interpretation.winner_runner_up_gap,

        # Extracted typed data
        axis=axis,
        traces=traces,
        scalar_rows=scalar_rows,
        category_rows=category_rows,

        # Always-populated raw data
        raw=raw,

        # Provenance and full hypothesis ranking
        provenance_cells=_collect_provenance(interpretation),
        all_hypothesis_scores=_build_hypothesis_scores(interpretation),

        # Ambiguity visibility
        is_ambiguous=is_ambiguous,
        ambiguity_note=ambiguity_note,
    )


def extract_sheet_regions(
    sheet: RawSheet,
    regions: List[CandidateRegion],
    interpretations: List[RegionInterpretation],
) -> List[ExtractedDataset]:
    """
    Extract all regions from a sheet given their interpretations.

    Regions and interpretations must be in the same order.
    Returns one ExtractedDataset per region.
    """
    assert len(regions) == len(interpretations), (
        f"Mismatch: {len(regions)} regions vs {len(interpretations)} interpretations"
    )
    return [
        extract_region(sheet, region, interp)
        for region, interp in zip(regions, interpretations)
    ]
