from __future__ import annotations

from typing import List
from app.domain.audit import DataBlockAudit, SheetAudit, SheetParsingStatus
from app.domain.document import DocumentModel, DocumentSheet
from app.layout.segmenter import TableSegmenter


class LayoutDiscoveryEngine:
    def __init__(self, segmenter: TableSegmenter = None):
        self.segmenter = segmenter or TableSegmenter()

    def discover_layout(self, doc_model: DocumentModel) -> List[SheetAudit]:
        """Scans all worksheets in a DocumentModel and discovers independent table blocks."""
        sheet_audits: List[SheetAudit] = []

        for sheet in doc_model.sheets:
            if sheet.row_count == 0 or sheet.col_count == 0:
                sheet_audits.append(
                    SheetAudit(
                        sheet_name=sheet.sheet_name,
                        sheet_index=sheet.sheet_index,
                        row_count=sheet.row_count,
                        col_count=sheet.col_count,
                        status=SheetParsingStatus.EMPTY,
                        warnings=["Sheet contains no rows or columns."],
                        blocks=[],
                    )
                )
                continue

            blocks: List[DataBlockAudit] = self.segmenter.segment_sheet(sheet)
            status = SheetParsingStatus.FULLY_PARSED if blocks else SheetParsingStatus.EMPTY

            sheet_audits.append(
                SheetAudit(
                    sheet_name=sheet.sheet_name,
                    sheet_index=sheet.sheet_index,
                    row_count=sheet.row_count,
                    col_count=sheet.col_count,
                    status=status,
                    warnings=[] if blocks else ["No non-empty table blocks discovered in sheet."],
                    blocks=blocks,
                )
            )

        return sheet_audits
