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
)
from app.readers.base_reader import IInputReader


class CSVInputReader(IInputReader):
    @property
    def supported_format(self) -> FormatType:
        return FormatType.CSV

    def can_read(self, filename: str, contents: bytes) -> bool:
        return filename.endswith(".csv")

    def read(self, filename: str, contents: bytes) -> DocumentModel:
        df_raw = pd.read_csv(io.BytesIO(contents), header=None)
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
                    raw_value=raw_v,
                    formatted_value=fmt_v,
                    data_type=d_type,
                    is_merged=False,
                )
                row_cells.append(cell)
            cells_grid.append(row_cells)

        doc_sheet = DocumentSheet(
            sheet_name="CSV_Data",
            sheet_index=0,
            row_count=row_count,
            col_count=col_count,
            cells=cells_grid,
        )

        return DocumentModel(
            document_name=filename,
            format_type=FormatType.CSV,
            metadata={"sheet_count": 1},
            sheets=[doc_sheet],
        )
