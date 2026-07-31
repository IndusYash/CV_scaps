from __future__ import annotations

from abc import ABC, abstractmethod
from app.domain.document import DocumentModel, FormatType


class IInputReader(ABC):
    @property
    @abstractmethod
    def supported_format(self) -> FormatType:
        """Returns the format type supported by this reader."""
        pass

    @abstractmethod
    def can_read(self, filename: str, contents: bytes) -> bool:
        """Determines if this reader can parse the input file."""
        pass

    @abstractmethod
    def read(self, filename: str, contents: bytes) -> DocumentModel:
        """Parses input file contents into an abstract DocumentModel."""
        pass
