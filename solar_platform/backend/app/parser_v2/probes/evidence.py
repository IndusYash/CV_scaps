"""
parser_v2.probes.evidence
-------------------------
Typed evidence data structures and CellRef provenance tracking.

Each EvidenceItem captures a single structural or semantic observation
with explicit polarity (POSITIVE, NEGATIVE, UNKNOWN), confidence score,
rationale, and source cell provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CellRef:
    """Cell-level provenance tracking."""
    sheet_name: str
    row: int                             # 0-indexed row
    col: int                             # 0-indexed column
    raw_value: Any
    display_str: str


class EvidencePolarity(str, Enum):
    """Explicit polarity of an evidence observation."""
    POSITIVE = "POSITIVE"                # Supports a structural or semantic hypothesis
    NEGATIVE = "NEGATIVE"                # Contradicts a structural or semantic hypothesis
    UNKNOWN  = "UNKNOWN"                 # Ambiguous or insufficient signal


@dataclass
class EvidenceItem:
    """
    A single typed evidence observation produced by a probe.

    Fields:
      name: str identifier (e.g. "axis_monotonicity", "parameter_expression", "label_concept_candidate")
      probe_id: str identifier of the probe that generated this item
      polarity: EvidencePolarity (POSITIVE / NEGATIVE / UNKNOWN)
      confidence: float [0.0, 1.0]
      value: Any structured payload or signal value
      rationale: Human-readable explanation of why this evidence was produced
      source_cells: List[CellRef] supporting provenance for this observation
    """
    name: str
    probe_id: str
    polarity: EvidencePolarity
    confidence: float
    value: Any
    rationale: str
    source_cells: List[CellRef] = field(default_factory=list)


@dataclass
class EvidenceBag:
    """Container storing all evidence items collected for a CandidateRegion."""
    region_id: str
    sheet_name: str
    items: List[EvidenceItem] = field(default_factory=list)

    def add(self, item: EvidenceItem) -> None:
        self.items.append(item)

    def filter_by_probe(self, probe_id: str) -> List[EvidenceItem]:
        return [item for item in self.items if item.probe_id == probe_id]

    def filter_by_name(self, name: str) -> List[EvidenceItem]:
        return [item for item in self.items if item.name == name]

    def filter_by_polarity(self, polarity: EvidencePolarity) -> List[EvidenceItem]:
        return [item for item in self.items if item.polarity == polarity]
