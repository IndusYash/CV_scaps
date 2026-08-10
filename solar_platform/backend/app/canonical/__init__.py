"""
app.canonical
-------------
Canonical Scientific Dataset Layer.
Provides a clean, standardized external dataset representation between parser_v2 and future ML systems.
"""

from app.canonical.converter import (
    build_canonical_workbook,
    convert_validated_dataset_to_canonical,
)
from app.canonical.model import (
    CanonicalParameter,
    CanonicalScalarRecord,
    CanonicalTrace,
    CanonicalWorkbook,
    CellRefDTO,
    ScientificDataset,
)

__all__ = [
    "CellRefDTO",
    "CanonicalParameter",
    "CanonicalTrace",
    "CanonicalScalarRecord",
    "ScientificDataset",
    "CanonicalWorkbook",
    "convert_validated_dataset_to_canonical",
    "build_canonical_workbook",
]
