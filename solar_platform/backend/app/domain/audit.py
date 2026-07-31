from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.domain.ir import IntermediateRepresentationV2
from app.domain.provenance import CellLineage


class BlockCategory(str, Enum):
    CURVE = "CURVE"
    TABLE = "TABLE"
    MATRIX = "MATRIX"
    SUMMARY = "SUMMARY"
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    UNKNOWN = "UNKNOWN"


class BlockParsingStatus(str, Enum):
    PARSED = "PARSED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"


class SheetParsingStatus(str, Enum):
    FULLY_PARSED = "FULLY_PARSED"
    PARTIALLY_PARSED = "PARTIALLY_PARSED"
    UNSUPPORTED = "UNSUPPORTED"
    EMPTY = "EMPTY"
    ERROR = "ERROR"


class BoundingBox(BaseModel):
    start_row: int
    end_row: int
    start_col: int
    end_col: int


class DecisionTraceEntry(BaseModel):
    decision_type: str
    decision: str
    evidence: str
    confidence: float


class CandidateEvaluation(BaseModel):
    plugin_id: str
    score: float
    match_reasons: List[str] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)


class PluginSelectionAudit(BaseModel):
    candidate_evaluations: List[CandidateEvaluation] = Field(default_factory=list)
    selected_plugin_id: str
    selection_rationale: str
    selection_confidence: float


class DataBlockAudit(BaseModel):
    block_id: str
    category: BlockCategory = BlockCategory.UNKNOWN
    semantic_label: str = "GENERIC_PARAMETRIC"
    bounding_box: BoundingBox
    parsing_confidence: float = 1.0
    scientific_validity: float = 1.0
    parsing_status: BlockParsingStatus = BlockParsingStatus.PARSED
    reason_if_unsupported: Optional[str] = None
    plugin_selection: Optional[PluginSelectionAudit] = None
    provenance: Optional[CellLineage] = None
    decision_traces: List[DecisionTraceEntry] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    recommended_plugin: str = "generic_plugin"
    intermediate_representation: Optional[IntermediateRepresentationV2] = None


class SheetAudit(BaseModel):
    sheet_name: str
    sheet_index: int
    row_count: int
    col_count: int
    status: SheetParsingStatus = SheetParsingStatus.FULLY_PARSED
    warnings: List[str] = Field(default_factory=list)
    blocks: List[DataBlockAudit] = Field(default_factory=list)


class VersionManifest(BaseModel):
    parser_core_version: str = "v2.0.0"
    semantic_registry_version: str = "v1.4.0"
    validation_rule_version: str = "v1.1.0"
    plugin_versions: Dict[str, str] = Field(default_factory=dict)


class WorkbookAuditResult(BaseModel):
    file_id: str
    filename: str
    timestamp: str
    version_manifest: VersionManifest = Field(default_factory=VersionManifest)
    overall_parsing_confidence: float = 1.0
    overall_scientific_validity: float = 1.0
    total_blocks_found: int = 0
    total_blocks_parsed: int = 0
    sheets: List[SheetAudit] = Field(default_factory=list)
