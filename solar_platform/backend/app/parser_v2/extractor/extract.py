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
from app.parser_v2.extractor.interleaved_curve_grid_extractor import extract_interleaved_curve_grid
from app.parser_v2.extractor.numeric_curve_grid_extractor import extract_numeric_curve_grid
from app.parser_v2.extractor.scalar_sweep_table_extractor import extract_scalar_sweep_table
from app.parser_v2.extractor.stacked_curve_blocks_extractor import extract_stacked_curve_blocks
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
) -> List[ExtractedDataset]:
    """
    Extract structured data from one region based on its winning interpretation.

    It does NOT re-classify or override the Reasoner's decisions.
    """
    winner = interpretation.winner
    acceptance_state = interpretation.acceptance_state

    # Determine ambiguity
    is_ambiguous = (acceptance_state != AcceptanceState.ACCEPTED)

    ambiguity_note: Optional[str] = None
    if is_ambiguous:
        ambiguity_note = (
            f"Region interpretation state is {acceptance_state.value}. "
            f"Winner evidence-support score={winner.score if winner else 0.0:.2f}, "
            f"gap={interpretation.winner_runner_up_gap:.2f}. "
            f"Raw cell grid is fully preserved."
        )

    raw = _build_raw(sheet, region)

    if winner is not None and winner.structure == StructuralHypothesis.STACKED_CURVE_BLOCKS:
        blocks = extract_stacked_curve_blocks(sheet, region, interpretation)
        if blocks:
            for b in blocks:
                b.raw = raw
                b.winner_evidence_score = winner.score
                b.winner_runner_up_gap = interpretation.winner_runner_up_gap
                b.all_hypothesis_scores = _build_hypothesis_scores(interpretation)
                b.is_ambiguous = is_ambiguous
                b.ambiguity_note = ambiguity_note
            return blocks

    if winner is not None and winner.structure == StructuralHypothesis.INTERLEAVED_CURVE_GRID:
        curves = extract_interleaved_curve_grid(sheet, region, interpretation)
        if curves:
            for c in curves:
                c.raw = raw
                c.winner_evidence_score = winner.score
                c.winner_runner_up_gap = interpretation.winner_runner_up_gap
                c.all_hypothesis_scores = _build_hypothesis_scores(interpretation)
                c.is_ambiguous = is_ambiguous
                c.ambiguity_note = ambiguity_note
            return curves

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
            ambiguity_note = (
                "Structural hypothesis is Unknown. "
                "Raw data preserved; no typed extraction performed."
            )

    return [ExtractedDataset(
        region_id=region.region_id,
        sheet_name=region.sheet_name,
        structure_type=winner.structure.value if winner else StructuralHypothesis.UNKNOWN.value,
        scientific_label=winner.scientific_label if winner else None,
        axis_concept=winner.axis_concept if winner else None,
        target_concept=winner.target_concept if winner else None,
        acceptance_state=acceptance_state.value,
        winner_evidence_score=winner.score if winner else 0.0,
        winner_runner_up_gap=interpretation.winner_runner_up_gap,
        axis=axis,
        traces=traces,
        scalar_rows=scalar_rows,
        category_rows=category_rows,
        raw=raw,
        provenance_cells=_collect_provenance(interpretation),
        all_hypothesis_scores=_build_hypothesis_scores(interpretation),
        is_ambiguous=is_ambiguous,
        ambiguity_note=ambiguity_note,
    )]


def extract_sheet_regions(
    sheet: RawSheet,
    regions: List[CandidateRegion],
    interpretations: List[RegionInterpretation],
) -> List[ExtractedDataset]:
    """
    Extract all regions from a sheet given their interpretations.
    """
    assert len(regions) == len(interpretations), (
        f"Mismatch: {len(regions)} regions vs {len(interpretations)} interpretations"
    )
    all_datasets: List[ExtractedDataset] = []
    for region, interp in zip(regions, interpretations):
        extracted = extract_region(sheet, region, interp)
        all_datasets.extend(extracted)
    return all_datasets
