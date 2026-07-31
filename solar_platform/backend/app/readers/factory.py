from __future__ import annotations

from typing import Dict, Type
from app.exceptions import UnsupportedFormatException
from app.readers.base_reader import IInputReader
from app.readers.csv_reader import CSVInputReader
from app.readers.excel_reader import ExcelInputReader


class ReaderFactory:
    _EXTENSION_MAP: Dict[str, Type[IInputReader]] = {
        ".xlsx": ExcelInputReader,
        ".xls": ExcelInputReader,
        ".csv": CSVInputReader,
    }

    @classmethod
    def get_reader(cls, filename: str, contents: bytes) -> IInputReader:
        """Resolves the appropriate IInputReader based on filename extension and magic content headers."""
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        reader_cls = cls._EXTENSION_MAP.get(ext)

        if not reader_cls:
            raise UnsupportedFormatException(f"Unsupported file extension '{ext}' for file '{filename}'.")

        reader_instance = reader_cls()
        if not reader_instance.can_read(filename, contents):
            raise UnsupportedFormatException(f"Reader '{reader_cls.__name__}' cannot parse contents of '{filename}'.")

        return reader_instance
