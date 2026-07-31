from __future__ import annotations

import io
import pandas as pd
from typing import List
from app.domain.document import (
    CellDataType,
    DocumentCell,
    DocumentModel,
    DocumentSheet,
    FormatType,
    StyleMetadata,
)
from app.readers.base_reader import IInputReader


class ExcelInputReader(IInputReader):
    @property
    def supported_format(self) -> FormatType:
        return FormatType.EXCEL

    def can_read(self, filename: str, contents: bytes) -> bool:
        return filename.lower().endswith((".xlsx", ".xls"))

    def read(self, filename: str, contents: bytes) -> DocumentModel:
        excel_file = pd.ExcelFile(io.BytesIO(contents))
        doc_sheets: List[DocumentSheet] = []

        for s_idx, sheet_name in enumerate(excel_file.sheet_names):
            df_raw = excel_file.parse(sheet_name, header=None)
            row_count, col_count = df_raw.shape

            cells_grid: List[List[DocumentCell]] = []
            for r_idx in range(row_count):
                row_cells: List[DocumentCell] = []
                for c_idx in range(col_count):
                    val = df_raw.iat[r_idx, c_idx]
                    
                    if pd.isna(val) or val is None:
                        d_type = CellDataType.EMPTY
                        raw_v = None
                        fmt_v = ""
                    elif isinstance(val, (int, float)):
                        d_type = CellDataType.NUMBER
                        raw_v = float(val)
                        fmt_v = str(val)
                    elif isinstance(val, bool):
                        d_type = CellDataType.BOOLEAN
                        raw_v = val
                        fmt_v = str(val)
                    else:
                        str_val = str(val).strip()
                        if not str_val:
                            d_type = CellDataType.EMPTY
                            raw_v = None
                            fmt_v = ""
                        else:
                            try:
                                num_v = float(str_val)
                                d_type = CellDataType.NUMBER
                                raw_v = num_v
                                fmt_v = str_val
                            except ValueError:
                                d_type = CellDataType.STRING
                                raw_v = str_val
                                fmt_v = str_val

                    cell = DocumentCell(
                        row_index=r_idx,
                        col_index=c_idx,
                        data_type=d_type,
                        raw_value=raw_v,
                        formatted_value=fmt_v,
                        style_metadata=StyleMetadata(),
                    )
                    row_cells.append(cell)
                cells_grid.append(row_cells)

            doc_sheet = DocumentSheet(
                sheet_name=sheet_name,
                sheet_index=s_idx,
                row_count=row_count,
                col_count=col_count,
                cells=cells_grid,
            )
            doc_sheets.append(doc_sheet)

        return DocumentModel(
            document_name=filename,
            format_type=self.supported_format,
            sheets=doc_sheets,
        )
