"""
parser_v2.reasoner.hypothesis
-----------------------------
Data models for structural and scientific hypotheses, ranked candidate sets,
and region interpretations.

Concept separation
------------------
Three distinct concepts are maintained independently throughout the Reasoner:

1. **Structural hypothesis** (``StructuralHypothesis``)
   Generic structural form of the region: NumericCurveGrid, ScalarSweepTable,
   CategoryTable, or Unknown.  This is always set to the best structural match
   found by the scorer, even when the overall interpretation is UNRESOLVED.

2. **Scientific interpretation** (``HypothesisCandidate.scientific_label``)
   Layered semantic classification of what experiment this structure represents:
   JV, QE, RESPONSIVITY, DETECTIVITY, SCALAR_SWEEP, CATEGORY, UNKNOWN, etc.
   This is derived *after* structural scoring and may be UNKNOWN even when the
   structural hypothesis is clear.

3. **Overall acceptance state** (``AcceptanceState``)
   Reflects the *confidence of the complete interpretation* given the available
   evidence.  UNRESOLVED means the overall region interpretation cannot be
   confirmed — it does NOT mean the winning structural hypothesis is wrong.

Example (Sheet1 device-profile sheet)::

    winner.structure       = StructuralHypothesis.CATEGORY_TABLE   # structure is known
    winner.scientific_label = "UNKNOWN"                             # semantics unclear
    acceptance_state        = AcceptanceState.UNRESOLVED            # interpretation unclear

Score semantics
---------------
All ``score`` fields are **evidence-support scores**, not probabilities or
percentages.  A score of 0.00 means that no positive scoring rule contributed
positive evidence for that hypothesis — it does NOT mean the hypothesis is
deemed impossible or ruled out.  A score of 1.00 means the maximum evidence
capacity was reached, not that the probability of the hypothesis is 100%.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.probes.evidence import CellRef, EvidenceBag


class AcceptanceState(str, Enum):
    """
    Overall confidence in the complete region interpretation.

    This reflects confidence in the *interpretation as a whole* (structure +
    scientific label together) given the evidence gathered.  It is NOT a
    statement about whether the winning structural hypothesis is correct.

    A region with UNRESOLVED state still has a ``winner.structure`` value — the
    reasoner always selects the best available structural hypothesis.  UNRESOLVED
    simply means that interpretation could not be confirmed with sufficient
    evidence-support score and separation from competing hypotheses.
    """
    ACCEPTED   = "ACCEPTED"      # Winner evidence-support score >= 0.75 and winner/runner-up gap >= 0.20
    UNCERTAIN  = "UNCERTAIN"     # Winner evidence-support score >= 0.50 and gap >= 0.15
    UNRESOLVED = "UNRESOLVED"    # Winner evidence-support score < 0.50 or gap < 0.15; overall interpretation is ambiguous


class StructuralHypothesis(str, Enum):
    """The approved generic structural forms + Unknown."""
    NUMERIC_CURVE_GRID = "NumericCurveGrid"
    SCALAR_SWEEP_TABLE = "ScalarSweepTable"
    CATEGORY_TABLE     = "CategoryTable"
    STACKED_CURVE_BLOCKS = "StackedCurveBlocks"
    UNKNOWN            = "Unknown"


@dataclass
class HypothesisCandidate:
    """
    A single evaluated hypothesis: generic structural form + layered scientific interpretation.

    The three concepts on this object are always kept separate:

    * ``structure``        — the generic structural form (``StructuralHypothesis``)
    * ``scientific_label`` — the layered scientific interpretation string
    * ``score``            — the evidence-support score (see below)

    Score semantics
    ~~~~~~~~~~~~~~~
    ``score`` is an **evidence-support score** in the range [0.0, 1.0].  It
    accumulates positive contributions from supporting evidence items and
    subtracts contributions from contradicting evidence items.  The range is
    clamped at 0.0 and 1.0 as lower and upper evidence-capacity bounds.

    * score == 0.00  means no positive scoring rule contributed evidence for
                     this hypothesis.  It does NOT mean the hypothesis is
                     impossible or ruled out.
    * score == 1.00  means the maximum evidence capacity was reached.

    Scores must NEVER be interpreted as probabilities or percentages.
    """
    # --- Structural hypothesis (concept 1) ---
    structure: StructuralHypothesis

    # --- Layered scientific interpretation (concept 2) ---
    # One of: "JV", "QE", "RESPONSIVITY", "DETECTIVITY", "SCALAR_SWEEP",
    #         "CATEGORY", "UNKNOWN", or None (if not yet interpretable).
    scientific_label: Optional[str]

    # Axis and target concepts derived from label-probe evidence.
    # Examples: axis="Voltage", target="Current" → JV curve.
    axis_concept: Optional[str]
    target_concept: Optional[str]

    # --- Evidence-support score (NOT a probability) ---
    score: float  # Clamped to [0.0, 1.0]; see class docstring for semantics.

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
