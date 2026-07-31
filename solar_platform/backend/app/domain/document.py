from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CellDataType(str, Enum):
    STRING = "STRING"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    DATE = "DATE"
    EMPTY = "EMPTY"


class StyleMetadata(BaseModel):
    font_bold: bool = False
    background_color: Optional[str] = None


class DocumentCell(BaseModel):
    row_index: int
    col_index: int
    raw_value: Any = None
    formatted_value: str = ""
    data_type: CellDataType = CellDataType.EMPTY
    is_merged: bool = False
    style_metadata: Optional[StyleMetadata] = None

    def is_numeric(self) -> bool:
        if self.data_type == CellDataType.NUMBER:
            return True
        if self.data_type == CellDataType.STRING and self.raw_value is not None:
            try:
                float(str(self.raw_value).strip())
                return True
            except ValueError:
                return False
        return False

    def to_float(self) -> Optional[float]:
        if self.raw_value is None:
            return None
        try:
            return float(self.raw_value)
        except (ValueError, TypeError):
            return None


class DocumentSheet(BaseModel):
    sheet_name: str
    sheet_index: int
    row_count: int
    col_count: int
    cells: List[List[DocumentCell]] = Field(default_factory=list)

    def get_cell(self, row: int, col: int) -> Optional[DocumentCell]:
        if 0 <= row < self.row_count and 0 <= col < self.col_count:
            return self.cells[row][col]
        return None


class FormatType(str, Enum):
    EXCEL = "EXCEL"
    CSV = "CSV"
    JSON = "JSON"
    LIBREOFFICE = "LIBREOFFICE"


class DocumentModel(BaseModel):
    document_name: str = Field(..., alias="document_name")
    format_type: FormatType = FormatType.EXCEL
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sheets: List[DocumentSheet] = Field(default_factory=list)
