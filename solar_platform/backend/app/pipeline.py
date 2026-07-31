from __future__ import annotations

import datetime
from typing import Dict, List, Tuple
import pandas as pd

from app.adapters.rf_adapter import RandomForestMLAdapter
from app.domain.audit import SheetParsingStatus, WorkbookAuditResult
from app.domain.document import DocumentModel
from app.domain.ir_generator import IRGeneratorEngine
from app.layout.discovery import LayoutDiscoveryEngine
from app.readers.excel_reader import ExcelInputReader
from app.semantics.registry import SemanticRegistry
from app.validation.validator import ScientificValidator


class UnifiedParserPipeline:
    def __init__(self):
        self.excel_reader = ExcelInputReader()
        self.layout_engine = LayoutDiscoveryEngine()
        self.ir_engine = IRGeneratorEngine()
        self.semantic_registry = SemanticRegistry()
        self.validator = ScientificValidator()
        self.ml_adapter = RandomForestMLAdapter()

    def parse_workbook(
        self,
        filename: str,
        file_bytes: bytes,
        file_id: str,
    ) -> Tuple[WorkbookAuditResult, Dict[str, Dict[str, pd.DataFrame]]]:
        """Parses a workbook through the production architecture.

        Returns:
            - WorkbookAuditResult: Full audit object with decision traces, provenance, & confidence.
            - Dict[str, Dict[str, pd.DataFrame]]: Map of {sheet_name: {curve_type: long_df}} for ml_engine.py.
        """
        # Stage 1: Ingestion to DocumentModel
        doc_model: DocumentModel = self.excel_reader.read(filename, file_bytes)

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

            sheet_curve_data: Dict[str, pd.DataFrame] = {}

            for block_audit in sheet_audit.blocks:
                total_blocks += 1

                # Generate IR
                ir = self.ir_engine.generate_ir_for_block(doc_sheet, block_audit)
                block_audit.intermediate_representation = ir

                # Semantic Tagging
                curve_type = self.semantic_registry.evaluate_and_tag(ir, block_audit)
                curve_key = curve_type.lower()

                # Scientific Validation
                self.validator.validate_ir(ir, block_audit)

                # ML Adapter Long DataFrame Transformation
                df_long = self.ml_adapter.transform_to_long_dataframe(ir)
                if not df_long.empty:
                    sheet_curve_data[curve_key] = df_long
                    parsed_blocks += 1

            if sheet_curve_data:
                downstream_data[sheet_name] = sheet_curve_data

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
