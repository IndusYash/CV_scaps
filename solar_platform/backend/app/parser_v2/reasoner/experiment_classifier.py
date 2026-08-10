"""
parser_v2.reasoner.experiment_classifier
----------------------------------------
Main Stage 4 Reasoning Engine.

Flow:
  1. Gathers evidence from all 6 probes to populate an EvidenceBag.
  2. Scores all generic structural hypotheses (NumericCurveGrid, ScalarSweepTable, CategoryTable, Unknown).
  3. Layers scientific interpretations on top of structural compatibility.
  4. Ranks all candidate hypotheses by score descending (NEVER discards runner-ups!).
  5. Determines AcceptanceState (ACCEPTED / UNCERTAIN / UNRESOLVED) using winner score & winner-runner_up gap.
  6. Preserves full provenance pointers to source CellRefs.

Does NOT perform numerical dataset extraction into long DataFrames or ScientificExperiment objects.
Produces RegionInterpretation objects only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.reader.raw_model import RawSheet
from app.parser_v2.scanner.region import CandidateRegion
from app.parser_v2.probes.evidence import CellRef, EvidenceBag
from app.parser_v2.probes import (
    axis_probe,
    header_probe,
    numeric_probe,
    parameter_probe,
    repetition_probe,
    label_probe,
)
from app.parser_v2.reasoner.hypothesis import (
    AcceptanceState,
    HypothesisCandidate,
    RegionInterpretation,
    StructuralHypothesis,
)
from app.parser_v2.reasoner.hypothesis_scorers import (
    score_category_table,
    score_numeric_curve_grid,
    score_scalar_sweep_table,
    score_unknown,
)


def collect_evidence_bag(region: CandidateRegion, sheet: RawSheet) -> EvidenceBag:
    """Run all 6 probes to gather complete evidence before hypothesis scoring."""
    bag = EvidenceBag(region_id=region.region_id, sheet_name=region.sheet_name)

    items = []
    items.extend(axis_probe.probe_axis(region, sheet))
    items.extend(header_probe.probe_headers(region, sheet))
    items.extend(numeric_probe.probe_numerics(region, sheet))
    items.extend(parameter_probe.probe_parameters(region, sheet))
    items.extend(repetition_probe.probe_repetition(region, sheet))
    items.extend(label_probe.probe_labels(region, sheet))

    for item in items:
        bag.add(item)

    return bag


def classify_region(
    region: CandidateRegion,
    sheet: RawSheet,
    neighbor_bags: Optional[Dict[str, EvidenceBag]] = None,
) -> RegionInterpretation:
    """
    Reason over an EvidenceBag for a CandidateRegion and produce a RegionInterpretation.
    """
    bag = collect_evidence_bag(region, sheet)

    # Score all candidate structural hypotheses
    candidates: List[HypothesisCandidate] = [
        score_numeric_curve_grid(bag, region),
        score_scalar_sweep_table(bag, region),
        score_category_table(bag, region),
        score_unknown(bag, region),
    ]

    # Sort candidates by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    winner = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    gap = winner.score - (runner_up.score if runner_up else 0.0)

    # Acceptance state determination based on winner score and gap
    if winner.score >= 0.75 and gap >= 0.20 and winner.structure != StructuralHypothesis.UNKNOWN:
        acceptance_state = AcceptanceState.ACCEPTED
    elif winner.score >= 0.50 and gap >= 0.15 and winner.structure != StructuralHypothesis.UNKNOWN:
        acceptance_state = AcceptanceState.UNCERTAIN
    else:
        acceptance_state = AcceptanceState.UNRESOLVED

    # Collect all cell refs for provenance
    prov_cells: List[CellRef] = []
    for cand in candidates:
        for ref in cand.provenance_cells:
            if ref not in prov_cells:
                prov_cells.append(ref)
                if len(prov_cells) >= 20:
                    break

    return RegionInterpretation(
        region_id=region.region_id,
        sheet_name=sheet.sheet_name,
        acceptance_state=acceptance_state,
        winner=winner,
        runner_up=runner_up,
        winner_runner_up_gap=round(gap, 4),
        all_candidates=candidates,  # Full ranked set preserved!
        evidence_bag=bag,
        provenance_cells=prov_cells,
    )


def classify_sheet_regions(sheet: RawSheet, regions: List[CandidateRegion]) -> List[RegionInterpretation]:
    """
    Scans and reasons over all CandidateRegions in a RawSheet.
    Keeps each RegionInterpretation separate while allowing neighbor evidence lookup.
    """
    if not regions:
        return []

    # First pass: collect evidence bags for all regions
    bags: Dict[str, EvidenceBag] = {
        reg.region_id: collect_evidence_bag(reg, sheet) for reg in regions
    }

    interpretations: List[RegionInterpretation] = []
    for reg in regions:
        neighbor_bags = {nid: bags[nid] for nid in reg.neighbor_region_ids if nid in bags}
        interp = classify_region(reg, sheet, neighbor_bags)
        interpretations.append(interp)

    return interpretations
