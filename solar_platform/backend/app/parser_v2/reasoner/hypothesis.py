"""
parser_v2.reasoner.hypothesis
-----------------------------
Data models for structural and scientific hypotheses, ranked candidate sets,
and region interpretations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.probes.evidence import CellRef, EvidenceBag


class AcceptanceState(str, Enum):
    """Explicit acceptance state based on winner score and winner/runner-up gap."""
    ACCEPTED   = "ACCEPTED"      # Winner score >= 0.75 and gap >= 0.20
    UNCERTAIN  = "UNCERTAIN"     # Winner score >= 0.50 and gap >= 0.15, but competing candidate close
    UNRESOLVED = "UNRESOLVED"    # Winner score < 0.50 or gap < 0.15; unclassified or ambiguous


class StructuralHypothesis(str, Enum):
    """The three approved generic structural forms + Unknown."""
    NUMERIC_CURVE_GRID = "NumericCurveGrid"
    SCALAR_SWEEP_TABLE = "ScalarSweepTable"
    CATEGORY_TABLE     = "CategoryTable"
    UNKNOWN            = "Unknown"


@dataclass
class HypothesisCandidate:
    """
    A single evaluated hypothesis combining generic structure and layered scientific interpretation.
    """
    structure: StructuralHypothesis
    scientific_label: Optional[str]      # "JV", "QE", "RESPONSIVITY", "DETECTIVITY", "ABSORPTION", "SCALAR_SWEEP", "CATEGORY", "UNKNOWN"
    axis_concept: Optional[str]          # "Voltage", "Wavelength", "Temperature", "Acceptor_Density", "Position", etc.
    target_concept: Optional[str]        # "Current", "QE", "Responsivity", "Detectivity", "PCE", etc.
    score: float                         # 0.0 to 1.0

    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    provenance_cells: List[CellRef] = field(default_factory=list)
    rationale: str = ""


@dataclass
class RegionInterpretation:
    """
    Complete reasoning outcome for a CandidateRegion.
    Always retains the full ranked list of all evaluated hypotheses.
    """
    region_id: str
    sheet_name: str
    acceptance_state: AcceptanceState
    winner: Optional[HypothesisCandidate]
    runner_up: Optional[HypothesisCandidate]
    winner_runner_up_gap: float

    all_candidates: List[HypothesisCandidate]   # Full ranked hypothesis set, NEVER discarded!
    evidence_bag: EvidenceBag
    provenance_cells: List[CellRef] = field(default_factory=list)
