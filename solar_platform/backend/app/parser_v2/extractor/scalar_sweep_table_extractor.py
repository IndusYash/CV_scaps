"""
parser_v2.extractor.scalar_sweep_table_extractor
-------------------------------------------------
Deterministic extractor for ScalarSweepTable regions.

Consumes:
  - RawSheet            (raw cell values with provenance)
  - CandidateRegion     (structural bounding box and column info)
  - RegionInterpretation (winner, evidence_bag)

Produces:
  - List[ScalarRow]     (one per parameter value, with metric columns)

Contract:
  - Does NOT re-classify or re-apply scientific keyword logic.
  - Parameter names and metric names come from the Reasoner's label_probe evidence.
  - String-encoded parameters (e.g. "10^13") are parsed via parameter_probe.
  - Provenance invariant: every metrics[key] has a metric_sources[key].
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.probes.evidence import CellRef, EvidenceBag, EvidencePolarity
from app.parser_v2.probes.parameter_probe import parse_parameter_expression
from app.parser_v2.reader.raw_model import RawCell, RawCellType, RawSheet
from app.parser_v2.reasoner.hypothesis import RegionInterpretation
from app.parser_v2.scanner.region import CandidateRegion
from app.parser_v2.extractor.ir import ScalarRow


def _make_cell_ref(sheet: RawSheet, r: int, c: int) -> CellRef:
    cell = sheet.get(r, c)
    if cell is None:
        return CellRef(sheet.sheet_name, r, c, None, "")
    return CellRef(sheet.sheet_name, r, c, cell.raw_value, cell.display_str)


def _cell_float(cell: Optional[RawCell]) -> Optional[float]:
    if cell is None:
        return None
    if cell.data_type in (RawCellType.NUMERIC, RawCellType.FORMULA):
        try:
            return float(cell.raw_value)
        except (TypeError, ValueError):
            return None
    # String cell that might encode "10^13"
    if cell.data_type == RawCellType.STRING and cell.display_str:
        import re
        m = re.search(r"\b10\^(\d+)\b", cell.display_str)
        if m:
            return float(10 ** int(m.group(1)))
        try:
            return float(cell.display_str.strip())
        except ValueError:
            return None
    return None


def _identify_header_row(region: CandidateRegion) -> Optional[int]:
    if region.candidate_header_rows:
        return region.candidate_header_rows[0]
    return None


def _identify_metric_columns(
    sheet: RawSheet,
    region: CandidateRegion,
    header_row: Optional[int],
    param_col: int,
    bag: EvidenceBag,
) -> List[Tuple[int, str]]:
    """
    Return (col_index, metric_name) pairs for metric columns in this region.

    Metric column names come from the label_probe evidence concept candidates
    where concept_type == 'scalar_metric'.  If no such match is found for a
    column, the header text is used verbatim.
    """
    if header_row is None:
        return []

    # Build a lookup of header text → concept name from label_probe evidence
    header_to_concept: Dict[str, str] = {}
    for item in bag.filter_by_name("candidate_label_concepts"):
        for cand in item.value.get("concept_candidates", []):
            if cand.get("type") == "scalar_metric":
                raw_text = (cand.get("raw_text") or "").strip().lower()
                if raw_text:
                    header_to_concept[raw_text] = cand.get("concept", "")

    result: List[Tuple[int, str]] = []
    for c in range(region.col_start, region.col_end + 1):
        if c == param_col:
            continue
        hdr_cell = sheet.get(header_row, c)
        if not hdr_cell or not hdr_cell.display_str.strip():
            continue
        hdr_text = hdr_cell.display_str.strip()
        # Lookup concept name from label_probe; fallback to verbatim header
        concept_name = header_to_concept.get(hdr_text.lower(), hdr_text)
        result.append((c, concept_name))

    return result


def extract_scalar_sweep_table(
    sheet: RawSheet,
    region: CandidateRegion,
    interpretation: RegionInterpretation,
) -> List[ScalarRow]:
    """
    Extract scalar sweep rows from a ScalarSweepTable region.

    Returns a list of ScalarRow objects.
    """
    winner = interpretation.winner
    bag = interpretation.evidence_bag

    header_row = _identify_header_row(region)
    param_col = region.col_start   # parameter column is always leftmost

    data_row_start = (header_row + 1) if header_row is not None else region.row_start

    # Identify metric columns from label_probe evidence
    metric_cols = _identify_metric_columns(sheet, region, header_row, param_col, bag)

    rows: List[ScalarRow] = []

    for r in range(data_row_start, region.row_end + 1):
        param_cell = sheet.get(r, param_col)
        if param_cell is None or param_cell.raw_value is None:
            # Check if it's a string-encoded parameter
            if param_cell is None or (
                param_cell.data_type == RawCellType.EMPTY
            ):
                continue

        param_source = _make_cell_ref(sheet, r, param_col)
        param_raw_text = param_cell.display_str.strip() if param_cell and param_cell.display_str else None

        # Parse parameter value
        param_value: Optional[float] = None
        param_unit: Optional[str] = None

        if param_raw_text:
            cand = parse_parameter_expression(param_raw_text, param_source)
            if cand is not None:
                param_value = cand.parsed_value
                param_unit = cand.unit_hint
            else:
                # Direct numeric cell
                param_value = _cell_float(param_cell)
                # Unit from label_probe context — don't re-derive, use winner axis_concept
                if winner and winner.axis_concept:
                    from app.parser_v2.probes.label_probe import CONCEPT_RULES
                    for rule in CONCEPT_RULES:
                        if rule["concept"] == winner.axis_concept and winner.axis_concept in (
                            "Thickness", "Temperature", "Defect_Density",
                            "Series_Resistance", "Shunt_Resistance", "Interface_Density",
                        ):
                            param_unit = rule["unit"]
                            break

        if param_value is None and param_raw_text:
            # Last attempt: string "10^N" or plain float
            param_value = _cell_float(param_cell)

        if param_value is None and (param_cell is None or param_cell.data_type == RawCellType.EMPTY):
            continue

        # Collect metric values
        metrics: Dict[str, float] = {}
        metric_sources: Dict[str, CellRef] = {}

        for metric_col, metric_name in metric_cols:
            val_cell = sheet.get(r, metric_col)
            val = _cell_float(val_cell)
            if val is not None:
                metrics[metric_name] = val
                metric_sources[metric_name] = _make_cell_ref(sheet, r, metric_col)

        rows.append(ScalarRow(
            parameter_value=param_value,
            parameter_unit=param_unit,
            parameter_raw_text=param_raw_text,
            parameter_source=param_source,
            metrics=metrics,
            metric_sources=metric_sources,
        ))

    return rows
