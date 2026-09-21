from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple, Type
import pandas as pd

from app.adapters.rf_adapter import RandomForestMLAdapter
from app.domain.audit import SheetParsingStatus, WorkbookAuditResult
from app.domain.document import DocumentModel
from app.domain.ir_generator import IRGeneratorEngine
from app.layout.discovery import LayoutDiscoveryEngine
from app.readers.factory import ReaderFactory
from app.semantics.registry import SemanticRegistry
from app.validation.validator import ScientificValidator


class UnifiedParserPipeline:
    def __init__(
        self,
        reader_factory: Type[ReaderFactory] = ReaderFactory,
        layout_engine: Optional[LayoutDiscoveryEngine] = None,
        ir_engine: Optional[IRGeneratorEngine] = None,
        semantic_registry: Optional[SemanticRegistry] = None,
        validator: Optional[ScientificValidator] = None,
        ml_adapter: Optional[RandomForestMLAdapter] = None,
    ):
        self.reader_factory = reader_factory
        self.layout_engine = layout_engine or LayoutDiscoveryEngine()
        self.ir_engine = ir_engine or IRGeneratorEngine()
        self.semantic_registry = semantic_registry or SemanticRegistry()
        self.validator = validator or ScientificValidator()
        self.ml_adapter = ml_adapter or RandomForestMLAdapter()

    def parse_workbook(
        self,
        filename: str,
        file_bytes: bytes,
        file_id: str,
    ) -> Tuple[WorkbookAuditResult, Dict[str, Dict[str, pd.DataFrame]]]:
        """Parses a workbook or CSV file through the production architecture.

        Returns:
            - WorkbookAuditResult: Full audit object with decision traces, provenance, & confidence.
            - Dict[str, Dict[str, pd.DataFrame]]: Map of {sheet_name: {block_id: long_df}} preserving all blocks.
        """
        # Stage 1: Dynamic Ingestion to DocumentModel via ReaderFactory
        reader = self.reader_factory.get_reader(filename, file_bytes)
        doc_model: DocumentModel = reader.read(filename, file_bytes)

        # Stage 2: Layout Discovery & Table Segmentation
        sheet_audits = self.layout_engine.discover_layout(doc_model)

        total_blocks = 0
        parsed_blocks = 0
        downstream_data: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Stage 3 to 6: IR Generation, Semantic Classification, Validation, and ML Adapter
        for sheet_audit in sheet_audits:
            sheet_name = sheet_audit.sheet_name
            doc_sheet = next((s for s in doc_model.sheets if s.sheet_name == sheet_name), None)
            if not doc_sheet:
                continue

            sheet_block_data: Dict[str, pd.DataFrame] = {}

            for block_audit in sheet_audit.blocks:
                total_blocks += 1

                # Generate IR
                ir = self.ir_engine.generate_ir_for_block(
                doc_sheet,
                block_audit,
                source_filename=filename,
                )
                block_audit.intermediate_representation = ir

                # Semantic Tagging
                curve_type = self.semantic_registry.evaluate_and_tag(ir, block_audit)
                curve_key = curve_type.lower()

                # Scientific Validation
                self.validator.validate_ir(ir, block_audit)

                # ML Adapter Long DataFrame Transformation
                df_long = self.ml_adapter.transform_to_long_dataframe(ir)
                if not df_long.empty:
                    df_long.attrs["block_id"] = block_audit.block_id
                    df_long.attrs["semantic_label"] = curve_type
                    
                    # Store by block_id to prevent block collisions, and also store by curve_key fallback
                    sheet_block_data[block_audit.block_id] = df_long
                    if curve_key not in sheet_block_data:
                        sheet_block_data[curve_key] = df_long
                    parsed_blocks += 1

            if sheet_block_data:
                downstream_data[sheet_name] = sheet_block_data

        audit_result = WorkbookAuditResult(
            file_id=file_id,
            filename=filename,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            overall_parsing_confidence=1.0 if parsed_blocks > 0 else 0.5,
            overall_scientific_validity=1.0,
            total_blocks_found=total_blocks,
            total_blocks_parsed=parsed_blocks,
            sheets=sheet_audits,
        )

        return audit_result, downstream_data
