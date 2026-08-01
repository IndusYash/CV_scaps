import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.semantics.segmenter import TableSegmenter
from app.semantics.registry import SemanticRegistry
from app.domain.ir_generator import IRGeneratorEngine
from app.adapters.rf_adapter import RandomForestMLAdapter
from app.pipeline import UnifiedParserPipeline
from app.services.ml_engine import train_sheet_models, predict_curve_comparison, get_trained_session

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

print(f"=== FORENSIC TRACER: Loading {wb_path.name} ===")
with open(wb_path, "rb") as f:
    contents = f.read()

# Stage 1: Excel Reader
reader = ExcelInputReader()
doc_model = reader.read(wb_path.name, contents)
print(f"Document Loaded: {doc_model.document_name}, Sheets: {len(doc_model.sheets)}")

pipeline = UnifiedParserPipeline()
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "file_forensic_001")

print("\n=== SHEET OVERVIEW ===")
for sheet_audit in audit_result.sheets:
    sname = sheet_audit.sheet_name
    blocks = sheet_audit.blocks
    print(f"Sheet '{sname}': {len(blocks)} blocks detected")
    categories = {}
    labels = {}
    for b in blocks:
        categories[str(b.category)] = categories.get(str(b.category), 0) + 1
        labels[b.semantic_label] = labels.get(b.semantic_label, 0) + 1
    print(f"   Categories: {categories}")
    print(f"   Semantic Labels: {labels}")
