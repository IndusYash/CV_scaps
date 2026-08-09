"""
parser_v2.probes.parameter_probe
--------------------------------
Stateless Probe 4: Parameter Probe.

Identifies candidate parameter expressions in header rows, banner rows, and column headers:
  - 't=100[nm]' -> value: 100.0, unit: 'nm'
  - 'T=300' or 'T=300K' -> value: 300.0, unit: 'K'
  - '10^13' -> value: 1e13
  - '1 ohm/cm^2' -> value: 1.0, unit: 'ohm/cm^2'
  - Plain numeric headers ('1000', '2000') when in header positions

Returns CandidateParameter structs in EvidenceItem payloads with full CellRef provenance.
Does NOT make final experiment classifications.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.reader.raw_model import RawSheet, RawCellType
from app.parser_v2.scanner.region import CandidateRegion
from app.parser_v2.probes.evidence import CellRef, EvidenceItem, EvidencePolarity


PROBE_ID = "ParameterProbe_v2.0"


@dataclass
class CandidateParameter:
    """A candidate swept parameter extracted from text."""
    raw_text: str
    parsed_value: float
    unit_hint: Optional[str]
    expression_pattern: str             # e.g. "thickness_t=", "temperature_T=", "power_exponent_10^N", "inline_unit", "plain_numeric"
    source_cell: CellRef


def _make_cell_ref(sheet: RawSheet, r: int, c: int) -> CellRef:
    cell = sheet.get(r, c)
    if cell is None:
        return CellRef(sheet.sheet_name, r, c, None, "")
    return CellRef(sheet.sheet_name, r, c, cell.raw_value, cell.display_str)


def parse_parameter_expression(text: str, cell_ref: CellRef) -> Optional[CandidateParameter]:
    """
    Parse a text token for parameter expression patterns.
    """
    if not text:
        return None
    s = text.strip()

    # Pattern 1: t=100[nm] or t=100nm or t=100
    m_t = re.search(r"\bt\s*=\s*([\d\.\+eE]+)\s*(?:\[?\s*([a-zA-Z\^/\-]+)\s*\]?)?", s, re.IGNORECASE)
    if m_t:
        val = float(m_t.group(1))
        unit = m_t.group(2) if m_t.group(2) else "nm"
        return CandidateParameter(
            raw_text=s,
            parsed_value=val,
            unit_hint=unit,
            expression_pattern="thickness_t=",
            source_cell=cell_ref,
        )

    # Pattern 2: T=300 or T=300K
    m_temp = re.search(r"\bT\s*=\s*([\d\.\+eE]+)\s*([kK])?\b", s)
    if m_temp:
        val = float(m_temp.group(1))
        unit = "K"
        return CandidateParameter(
            raw_text=s,
            parsed_value=val,
            unit_hint=unit,
            expression_pattern="temperature_T=",
            source_cell=cell_ref,
        )

    # Pattern 3: 10^13 or 10^N
    m_pow = re.search(r"\b10\^(\d+)\b", s)
    if m_pow:
        exp = int(m_pow.group(1))
        val = float(10 ** exp)
        return CandidateParameter(
            raw_text=s,
            parsed_value=val,
            unit_hint=None,  # Unit inferred from sheet context by label probe
            expression_pattern="power_exponent_10^N",
            source_cell=cell_ref,
        )

    # Pattern 4: Inline value + unit (e.g., "1 ohm/cm^2" or "2 ohm/cm2")
    m_unit = re.search(r"^([\d\.\+eE]+)\s+([a-zA-Z]+[a-zA-Z0-9\^/\-\.]*)$", s)
    if m_unit:
        try:
            val = float(m_unit.group(1))
            unit = m_unit.group(2)
            return CandidateParameter(
                raw_text=s,
                parsed_value=val,
                unit_hint=unit,
                expression_pattern="inline_unit",
                source_cell=cell_ref,
            )
        except ValueError:
            pass

    # Pattern 5: Plain numeric string in header context (e.g. "1000", "2000")
    if re.match(r"^[\d\.\+eE]+$", s):
        try:
            val = float(s)
            return CandidateParameter(
                raw_text=s,
                parsed_value=val,
                unit_hint=None,
                expression_pattern="plain_numeric",
                source_cell=cell_ref,
            )
        except ValueError:
            pass

    return None


def probe_parameters(region: CandidateRegion, sheet: RawSheet) -> List[EvidenceItem]:
    """
    Stateless scanning for candidate parameter expressions.
    """
    items: List[EvidenceItem] = []
    cand_params: List[CandidateParameter] = []

    # 1. Scan candidate header rows
    for r_hdr in region.candidate_header_rows:
        for c in range(region.col_start, region.col_end + 1):
            cell = sheet.get(r_hdr, c)
            if cell and cell.display_str and cell.display_str.strip():
                ref = _make_cell_ref(sheet, r_hdr, c)
                cand = parse_parameter_expression(cell.display_str, ref)
                if cand:
                    cand_params.append(cand)

    # 2. Scan banner rows directly above region
    banner_start = max(0, region.row_start - 5)
    for r_ban in range(banner_start, region.row_start):
        for c in range(region.col_start, region.col_end + 1):
            cell = sheet.get(r_ban, c)
            if cell and cell.display_str and cell.display_str.strip():
                ref = _make_cell_ref(sheet, r_ban, c)
                cand = parse_parameter_expression(cell.display_str, ref)
                if cand:
                    cand_params.append(cand)

    if cand_params:
        # Group by pattern type
        patterns_found = list({p.expression_pattern for p in cand_params})
        parsed_payload = [
            {
                "raw_text": p.raw_text,
                "parsed_value": p.parsed_value,
                "unit_hint": p.unit_hint,
                "pattern": p.expression_pattern,
                "cell_ref": p.source_cell,
            }
            for p in cand_params
        ]

        items.append(EvidenceItem(
            name="candidate_parameter_expressions",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.POSITIVE,
            confidence=0.85,
            value={
                "count": len(cand_params),
                "patterns": patterns_found,
                "parameters": parsed_payload,
            },
            rationale=f"Found {len(cand_params)} candidate parameter expressions matching patterns: {patterns_found}.",
            source_cells=[p.source_cell for p in cand_params],
        ))
    else:
        items.append(EvidenceItem(
            name="candidate_parameter_expressions",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.NEGATIVE,
            confidence=0.60,
            value={"count": 0, "patterns": []},
            rationale="No parameter expressions found in header or banner cells.",
            source_cells=[],
        ))

    return items
