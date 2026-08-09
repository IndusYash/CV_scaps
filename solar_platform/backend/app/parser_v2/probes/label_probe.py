"""
parser_v2.probes.label_probe
---------------------------
Stateless Probe 6: Label Probe.

Maps textual evidence from headers, banners, column strings, and sheet name to
candidate scientific concepts (e.g. Voltage, Current, QE, Responsivity, Detectivity,
Jsc, Voc, FF, PCE, Temperature, Thickness, Defect Density, Acceptor Density, etc.).

Contracts:
  - Returns CandidateConcept score items in EvidenceItem payloads with full CellRef provenance.
  - Can rank and suggest multiple candidate concept alternatives.
  - DOES NOT emit a final ScientificExperiment classification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.reader.raw_model import RawSheet, RawCellType
from app.parser_v2.scanner.region import CandidateRegion
from app.parser_v2.probes.evidence import CellRef, EvidenceItem, EvidencePolarity


PROBE_ID = "LabelProbe_v2.0"


@dataclass
class CandidateConcept:
    """A candidate scientific concept mapped from textual evidence."""
    concept_name: str                    # e.g. "Voltage", "Current", "QE", "Responsivity", "Jsc", "Voc", "PCE", "Temperature", "Thickness", "Defect_Density"
    concept_type: str                    # "axis", "target", "parameter", "scalar_metric"
    suggested_unit: str                  # e.g. "V", "mA/cm^2", "%", "nm", "K", "cm^-3", "A/W"
    score: float                         # 0.0 to 1.0 match score
    match_source: str                    # e.g. "header_token", "banner_token", "sheet_name"
    raw_text: str
    source_cell: Optional[CellRef]


def _make_cell_ref(sheet: RawSheet, r: int, c: int) -> CellRef:
    cell = sheet.get(r, c)
    if cell is None:
        return CellRef(sheet.sheet_name, r, c, None, "")
    return CellRef(sheet.sheet_name, r, c, cell.raw_value, cell.display_str)


# Controlled mapping rules for candidate concept evaluation
CONCEPT_RULES: List[Dict[str, Any]] = [
    # Axis concepts
    {"keywords": ["v(v)", "voltage", "v (v)", "v[v]"], "concept": "Voltage", "type": "axis", "unit": "V", "base_score": 0.95},
    {"keywords": ["lambda", "wavelength", "nm"], "concept": "Wavelength", "type": "axis", "unit": "nm", "base_score": 0.95},
    {"keywords": ["x(um)", "x(micrometer)", "position"], "concept": "Position", "type": "axis", "unit": "um", "base_score": 0.90},
    {"keywords": ["temperature", "t(k)", "t (k)"], "concept": "Temperature", "type": "parameter", "unit": "K", "base_score": 0.90},
    {"keywords": ["thickness", "thick"], "concept": "Thickness", "type": "parameter", "unit": "nm", "base_score": 0.90},

    # Target concepts
    {"keywords": ["jtot", "j(ma", "current", "jtot(ma/cm2)"], "concept": "Current", "type": "target", "unit": "mA/cm^2", "base_score": 0.95},
    {"keywords": ["qe", "q.e.", "quantum eff"], "concept": "QE", "type": "target", "unit": "%", "base_score": 0.95},
    {"keywords": ["r(a/w)", "r[a/w]", "responsivity"], "concept": "Responsivity", "type": "target", "unit": "A/W", "base_score": 0.95},
    {"keywords": ["detectivity", "d*"], "concept": "Detectivity", "type": "target", "unit": "Jones", "base_score": 0.95},
    {"keywords": ["alpha", "absorption"], "concept": "Absorption", "type": "target", "unit": "1/cm", "base_score": 0.90},

    # Scalar metrics
    {"keywords": ["jsc"], "concept": "Jsc", "type": "scalar_metric", "unit": "mA/cm^2", "base_score": 0.90},
    {"keywords": ["voc"], "concept": "Voc", "type": "scalar_metric", "unit": "V", "base_score": 0.90},
    {"keywords": ["ff"], "concept": "FF", "type": "scalar_metric", "unit": "%", "base_score": 0.90},
    {"keywords": ["pce", "eta"], "concept": "PCE", "type": "scalar_metric", "unit": "%", "base_score": 0.90},
    {"keywords": ["v_mpp"], "concept": "V_MPP", "type": "scalar_metric", "unit": "V", "base_score": 0.90},
    {"keywords": ["j_mpp"], "concept": "J_MPP", "type": "scalar_metric", "unit": "mA/cm^2", "base_score": 0.90},

    # Parameter concepts
    {"keywords": ["defect density", "idd", "nt="], "concept": "Defect_Density", "type": "parameter", "unit": "cm^-3", "base_score": 0.90},
    {"keywords": ["acceptor", "na="], "concept": "Acceptor_Density", "type": "parameter", "unit": "cm^-3", "base_score": 0.90},
    {"keywords": ["donor", "nd="], "concept": "Donor_Density", "type": "parameter", "unit": "cm^-3", "base_score": 0.90},
    {"keywords": ["rs=", "ohm/cm^2", "ohm.cm^2"], "concept": "Series_Resistance", "type": "parameter", "unit": "ohm.cm^2", "base_score": 0.85},
    {"keywords": ["rsh="], "concept": "Shunt_Resistance", "type": "parameter", "unit": "ohm.cm^2", "base_score": 0.85},
    {"keywords": ["interface density", "sns2/ps", "ps/zn3p2"], "concept": "Interface_Density", "type": "parameter", "unit": "cm^-2", "base_score": 0.90},
]


def match_text_to_concepts(text: str, source: str, cell_ref: Optional[CellRef]) -> List[CandidateConcept]:
    """Map a single text token to candidate concepts."""
    matches: List[CandidateConcept] = []
    if not text:
        return matches

    t_lower = text.strip().lower()

    for rule in CONCEPT_RULES:
        for kw in rule["keywords"]:
            if kw in t_lower:
                matches.append(CandidateConcept(
                    concept_name=rule["concept"],
                    concept_type=rule["type"],
                    suggested_unit=rule["unit"],
                    score=rule["base_score"],
                    match_source=source,
                    raw_text=text,
                    source_cell=cell_ref,
                ))
                break  # Don't duplicate matches for same rule

    return matches


def probe_labels(region: CandidateRegion, sheet: RawSheet) -> List[EvidenceItem]:
    """
    Stateless mapping of textual tokens to candidate scientific concepts.
    """
    items: List[EvidenceItem] = []
    candidates: List[CandidateConcept] = []

    # 1. Match candidate header row cells
    for r_hdr in region.candidate_header_rows:
        for c in range(region.col_start, region.col_end + 1):
            cell = sheet.get(r_hdr, c)
            if cell and cell.display_str and cell.display_str.strip():
                ref = _make_cell_ref(sheet, r_hdr, c)
                matches = match_text_to_concepts(cell.display_str, "header_token", ref)
                candidates.extend(matches)

    # 2. Match banner cells directly above region
    banner_start = max(0, region.row_start - 5)
    for r_ban in range(banner_start, region.row_start):
        for c in range(region.col_start, region.col_end + 1):
            cell = sheet.get(r_ban, c)
            if cell and cell.display_str and cell.display_str.strip():
                ref = _make_cell_ref(sheet, r_ban, c)
                matches = match_text_to_concepts(cell.display_str, "banner_token", ref)
                candidates.extend(matches)

    # 3. Match sheet name context
    sheet_matches = match_text_to_concepts(sheet.sheet_name, "sheet_name", None)
    candidates.extend(sheet_matches)

    if candidates:
        # Group candidate concepts by type
        concepts_by_name = list({c.concept_name for c in candidates})
        payload = [
            {
                "concept": c.concept_name,
                "type": c.concept_type,
                "unit": c.suggested_unit,
                "score": c.score,
                "source": c.match_source,
                "raw_text": c.raw_text,
                "cell_ref": c.source_cell,
            }
            for c in candidates
        ]

        items.append(EvidenceItem(
            name="candidate_label_concepts",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.POSITIVE,
            confidence=max(c.score for c in candidates),
            value={
                "distinct_concepts": concepts_by_name,
                "concept_candidates": payload,
            },
            rationale=f"Mapped textual tokens to candidate scientific concepts: {concepts_by_name}.",
            source_cells=[c.source_cell for c in candidates if c.source_cell is not None],
        ))
    else:
        items.append(EvidenceItem(
            name="candidate_label_concepts",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.UNKNOWN,
            confidence=0.30,
            value={"distinct_concepts": [], "concept_candidates": []},
            rationale="No scientific concept keywords matched in header/banner text.",
            source_cells=[],
        ))

    return items
