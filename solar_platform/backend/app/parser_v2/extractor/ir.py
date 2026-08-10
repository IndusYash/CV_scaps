"""
parser_v2.extractor.ir
----------------------
Intermediate Representation (IR) produced by Stage 5 Extractors.

Design principles
-----------------
* The extractor is a **deterministic consumer** of the Reasoner's
  RegionInterpretation.  It does NOT re-classify, re-score, or re-apply
  scientific keyword logic.
* All three output structures (NumericCurveGrid, ScalarSweepTable,
  CategoryTable) share a single ``ExtractedDataset`` container so that
  downstream consumers always receive the same type regardless of structure.
* ``raw`` is ALWAYS populated, even for UNRESOLVED interpretations.
  Data is never silently dropped.
* Cell-level provenance (``CellRef``) is carried through every value
  extracted — ``len(trace.values) == len(trace.source_refs)`` is an invariant.
* Ambiguous/UNRESOLVED extractions are visible:
  - ``is_ambiguous`` is True when ``acceptance_state != ACCEPTED``
  - ``ambiguity_note`` gives a human-readable explanation
  This prevents silent coercion of ambiguous data into scientific datasets.

Scores (from hypothesis candidates) are NOT reproduced here as probabilities.
They are preserved as evidence-support scores for diagnostic purposes only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.parser_v2.probes.evidence import CellRef
from app.parser_v2.reader.raw_model import RawCell


# ---------------------------------------------------------------------------
# Axis and trace data (used by NumericCurveGrid)
# ---------------------------------------------------------------------------

@dataclass
class AxisData:
    """
    Extracted independent-axis column for a NumericCurveGrid.

    concept and unit come directly from the Reasoner's winner.axis_concept
    and the LabelProbe's suggested_unit — the extractor does NOT re-derive them.
    """
    concept: str                   # e.g. "Voltage", "Wavelength", "Position"
    unit: str                      # e.g. "V", "nm", "um"
    values: List[float]            # extracted numeric values in row order
    source_refs: List[CellRef]     # one CellRef per value; len == len(values)


@dataclass
class TraceData:
    """
    One extracted data trace (y-column) from a NumericCurveGrid.

    Provenance invariant: ``len(values) == len(source_refs)`` always holds.

    For parameterized traces:
    - ``parameter_value`` is the parsed numeric parameter value
    - ``parameter_raw_text`` is the verbatim header text (e.g. "t=100[nm]",
      "10^13", "1 ohm/cm^2") — always preserved even if parsing succeeded
    - ``parameter_source`` is the CellRef of the header cell

    For un-parameterized (single-device) traces:
    - ``parameter_value``, ``parameter_raw_text``, ``parameter_source`` are None
    """
    trace_index: int               # 0-indexed position within this dataset's traces

    # Parameter identity (from parameter_probe; None for un-parameterized traces)
    parameter_value: Optional[float]
    parameter_unit: Optional[str]
    parameter_raw_text: Optional[str]    # verbatim original header text
    parameter_source: Optional[CellRef]  # header cell provenance

    # Y values
    values: List[float]
    source_refs: List[CellRef]     # one per value; len == len(values)


# ---------------------------------------------------------------------------
# Scalar sweep data (used by ScalarSweepTable)
# ---------------------------------------------------------------------------

@dataclass
class ScalarRow:
    """
    One row extracted from a ScalarSweepTable.

    Each row corresponds to one parameter value and zero or more scalar metrics.
    Metric names come directly from the Reasoner's label_probe output —
    they are not re-derived by the extractor.

    Provenance invariant: every key in ``metrics`` has a corresponding key
    in ``metric_sources``.
    """
    parameter_value: Optional[float]     # parsed numeric parameter value
    parameter_unit: Optional[str]
    parameter_raw_text: Optional[str]    # verbatim cell text (e.g. "10^13")
    parameter_source: Optional[CellRef]

    metrics: Dict[str, float]            # concept_name → value
    metric_sources: Dict[str, CellRef]   # concept_name → source CellRef


# ---------------------------------------------------------------------------
# Category table data (used by CategoryTable)
# ---------------------------------------------------------------------------

@dataclass
class CategoryRow:
    """
    One row extracted from a CategoryTable.

    The extractor does NOT assign scientific meaning to categorical labels.
    It preserves them verbatim.
    """
    label: str                           # verbatim string from col 0
    label_source: CellRef

    values: Dict[str, Any]               # column_header → raw value
    value_sources: Dict[str, CellRef]    # column_header → source CellRef


# ---------------------------------------------------------------------------
# Raw verbatim cell grid
# ---------------------------------------------------------------------------

@dataclass
class RawRegionData:
    """
    Always-populated verbatim cell grid for the extracted region.

    This is a rectangular slice of the RawSheet containing every cell
    within the region's bounding box, including empty cells.  It is
    NEVER None, even for UNRESOLVED extractions.
    """
    region_id: str
    sheet_name: str
    row_start: int          # 0-indexed, inclusive
    col_start: int          # 0-indexed, inclusive
    row_end: int            # 0-indexed, inclusive
    col_end: int            # 0-indexed, inclusive
    cells: List[List[RawCell]]  # cells[row_offset][col_offset]


# ---------------------------------------------------------------------------
# Top-level extraction result
# ---------------------------------------------------------------------------

@dataclass
class ExtractedDataset:
    """
    Complete extraction result for one CandidateRegion.

    The extractor populates only the fields that match the winning
    structural hypothesis:
    - NumericCurveGrid  → ``axis`` and ``traces``
    - ScalarSweepTable  → ``scalar_rows``
    - CategoryTable     → ``category_rows``
    - Unknown/UNRESOLVED → all of the above are empty; ``raw`` is still set

    All three field groups are always present (possibly empty) so that
    downstream consumers can safely access them without type switching.

    Ambiguity is made explicit:
    - ``is_ambiguous`` is True when acceptance_state != "ACCEPTED"
    - ``ambiguity_note`` is a human-readable description of the ambiguity

    Hypothesis scores are preserved in ``all_hypothesis_scores`` as
    evidence-support scores (NOT probabilities) for diagnostic use.
    """
    region_id: str
    sheet_name: str

    # --- Interpretation metadata (from RegionInterpretation, not re-derived) ---
    structure_type: str            # winner.structure.value
    scientific_label: Optional[str]  # winner.scientific_label
    axis_concept: Optional[str]    # winner.axis_concept
    target_concept: Optional[str]  # winner.target_concept
    acceptance_state: str          # interpretation.acceptance_state.value
    winner_evidence_score: float   # evidence-support score (NOT a probability)
    winner_runner_up_gap: float
    parent_region_id: Optional[str] = field(default=None)

    # --- Extracted data (populated by structure-specific extractors) ---
    # NumericCurveGrid
    axis: Optional[AxisData] = field(default=None)
    traces: List[TraceData] = field(default_factory=list)

    # ScalarSweepTable
    scalar_rows: List[ScalarRow] = field(default_factory=list)

    # CategoryTable
    category_rows: List[CategoryRow] = field(default_factory=list)

    # --- Raw cell grid (ALWAYS populated) ---
    raw: Optional[RawRegionData] = field(default=None)

    # --- Provenance ---
    provenance_cells: List[CellRef] = field(default_factory=list)

    # --- Full hypothesis ranking (evidence-support scores, NOT probabilities) ---
    all_hypothesis_scores: List[Dict[str, Any]] = field(default_factory=list)

    # --- Ambiguity visibility ---
    is_ambiguous: bool = False
    ambiguity_note: str = ""
