"""
parser_v2.probes.header_probe
-----------------------------
Stateless Probe 2: Header Probe.

Extracts and normalizes text tokens from candidate header rows and parameter banner rows.
Preserves original verbatim text, cell locations, and full CellRef provenance.

Does NOT perform scientific classification or filtering.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.reader.raw_model import RawSheet, RawCellType
from app.parser_v2.scanner.region import CandidateRegion
from app.parser_v2.probes.evidence import CellRef, EvidenceItem, EvidencePolarity


PROBE_ID = "HeaderProbe_v2.0"


def _make_cell_ref(sheet: RawSheet, r: int, c: int) -> CellRef:
    cell = sheet.get(r, c)
    if cell is None:
        return CellRef(sheet.sheet_name, r, c, None, "")
    return CellRef(sheet.sheet_name, r, c, cell.raw_value, cell.display_str)


def normalize_token(text: str) -> str:
    """Clean extra spaces, strip outer brackets/quotes, and normalize whitespace."""
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def probe_headers(region: CandidateRegion, sheet: RawSheet) -> List[EvidenceItem]:
    """
    Stateless extraction and normalization of header and banner text.
    """
    items: List[EvidenceItem] = []

    # 1. Header rows in candidate region
    header_tokens: List[Dict[str, Any]] = []

    for r_hdr in region.candidate_header_rows:
        for c in range(region.col_start, region.col_end + 1):
            cell = sheet.get(r_hdr, c)
            if cell and cell.display_str and cell.display_str.strip():
                raw_txt = cell.display_str.strip()
                norm_txt = normalize_token(raw_txt)
                ref = _make_cell_ref(sheet, r_hdr, c)

                header_tokens.append({
                    "row": r_hdr,
                    "col": c,
                    "raw_text": raw_txt,
                    "normalized_text": norm_txt,
                    "cell_ref": ref,
                })

    if header_tokens:
        items.append(EvidenceItem(
            name="header_row_presence",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.POSITIVE,
            confidence=0.85,
            value={"token_count": len(header_tokens), "tokens": header_tokens},
            rationale=f"Discovered {len(header_tokens)} header text tokens in region candidate header rows.",
            source_cells=[t["cell_ref"] for t in header_tokens],
        ))
    else:
        items.append(EvidenceItem(
            name="header_row_presence",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.NEGATIVE,
            confidence=0.60,
            value={"token_count": 0},
            rationale="No text headers discovered in region candidate header rows.",
            source_cells=[],
        ))

    # 2. Banner rows directly above candidate region
    banner_tokens: List[Dict[str, Any]] = []
    banner_start = max(0, region.row_start - 5)

    for r_ban in range(banner_start, region.row_start):
        for c in range(region.col_start, region.col_end + 1):
            cell = sheet.get(r_ban, c)
            if cell and cell.display_str and cell.display_str.strip():
                raw_txt = cell.display_str.strip()
                norm_txt = normalize_token(raw_txt)
                ref = _make_cell_ref(sheet, r_ban, c)

                banner_tokens.append({
                    "row": r_ban,
                    "col": c,
                    "raw_text": raw_txt,
                    "normalized_text": norm_txt,
                    "cell_ref": ref,
                })

    if banner_tokens:
        items.append(EvidenceItem(
            name="banner_tokens_presence",
            probe_id=PROBE_ID,
            polarity=EvidencePolarity.POSITIVE,
            confidence=0.75,
            value={"banner_token_count": len(banner_tokens), "banner_tokens": banner_tokens},
            rationale=f"Discovered {len(banner_tokens)} parameter banner text tokens above region.",
            source_cells=[t["cell_ref"] for t in banner_tokens],
        ))

    return items
