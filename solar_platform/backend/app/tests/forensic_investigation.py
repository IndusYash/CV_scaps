import sys
import io
import json
import pandas as pd
import numpy as np
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.layout.segmenter import TableSegmenter
from app.semantics.registry import SemanticRegistry
from app.domain.ir_generator import IRGeneratorEngine
from app.adapters.rf_adapter import RandomForestMLAdapter
from app.pipeline import UnifiedParserPipeline
from app.services.ml_engine import train_sheet_models, predict_curve_comparison, get_trained_session

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

print(f"====================================================================================================")
print(f"                                SCAPS FORENSIC DEBUGGING INVESTIGATION                              ")
print(f"====================================================================================================")
print(f"Target Workbook: {wb_path.name}\n")

with open(wb_path, "rb") as f:
    contents = f.read()

reader = ExcelInputReader()
doc_model = reader.read(wb_path.name, contents)

pipeline = UnifiedParserPipeline()
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "file_forensic_001")

doc_sheets = {s.sheet_name: s for s in doc_model.sheets}

def inspect_block_pipeline(sheet_name, block_index_list):
    sheet_audit = next((s for s in audit_result.sheets if s.sheet_name == sheet_name), None)
    if not sheet_audit:
        print(f"Sheet '{sheet_name}' not found!")
        return

    doc_sheet = doc_sheets.get(sheet_name)
    sheet_parsed = parsed_data.get(sheet_name, {})

    print(f"\n" + "="*100)
    print(f"INVESTIGATION: SHEET '{sheet_name}' (Total Blocks Discovered: {len(sheet_audit.blocks)})")
    print("="*100)

    for idx in block_index_list:
        if idx >= len(sheet_audit.blocks):
            print(f"Block index {idx} out of range!")
            continue
        
        b = sheet_audit.blocks[idx]
        b_id = b.block_id
        ir = b.intermediate_representation
        df_parsed = sheet_parsed.get(b_id, pd.DataFrame())

        print(f"\n--------------------------------------------------------------------------------")
        print(f"--- BLOCK #{idx + 1}: {b_id} ---")
        print(f"--------------------------------------------------------------------------------")

        # 1. RAW EXCEL CELLS
        bbox = b.bounding_box
        min_r = bbox.start_row
        min_c = bbox.start_col
        max_r = bbox.end_row
        max_c = bbox.end_col
        
        print(f"1. SOURCE EXCEL REGION:")
        print(f"   Bounding Box: Rows [{min_r}:{max_r}], Cols [{min_c}:{max_c}]")
        
        raw_cells_snippet = []
        for r in range(min_r, min(min_r + 5, max_r)):
            r_vals = []
            for c in range(min_c, min(min_c + 5, max_c)):
                cell = doc_sheet.get_cell(r, c)
                r_vals.append(cell.formatted_value if cell else "")
            raw_cells_snippet.append(" | ".join(r_vals))
        print("   Raw Top-Left Cells Snippet:")
        for r_str in raw_cells_snippet:
            print(f"     [ {r_str} ]")

        # 2. SEGMENTATION & CLASSIFICATION
        print(f"\n2. SEGMENTATION & SEMANTIC CLASSIFICATION:")
        print(f"   Category: {b.category}")
        print(f"   Semantic Label: {b.semantic_label}")
        print(f"   Confidence: {b.parsing_confidence:.2f}")

        # 3. INTERMEDIATE REPRESENTATION (IR)
        print(f"\n3. INTERMEDIATE REPRESENTATION (IR):")
        if ir:
            if ir.axes:
                ax = ir.axes[0]
                print(f"   Axis Name: '{ax.name}', Unit: '{ax.unit}', Values Count: {len(ax.values)}")
                print(f"     First 10 Axis Values: {ax.values[:10]}")
                print(f"     Last 10 Axis Values:  {ax.values[-10:] if len(ax.values)>=10 else ax.values}")
            if ir.parameters:
                p_info = [f"'{p.name}'={p.values[:3]} ({p.unit})" for p in ir.parameters]
                print(f"   Parameters ({len(ir.parameters)}): {p_info}")
            if ir.targets:
                tgt = ir.targets[0]
                print(f"   Target Name: '{tgt.name}', Unit: '{tgt.unit}', Shape: {tgt.data_shape}")
                if tgt.dense_matrix:
                    flat_vals = [row[0] for row in tgt.dense_matrix if row]
                    print(f"     First 10 Target Values (Col 0): {flat_vals[:10]}")
                    print(f"     Last 10 Target Values (Col 0):  {flat_vals[-10:] if len(flat_vals)>=10 else flat_vals}")
            if ir.summary_metrics:
                print(f"   Summary Metrics: {ir.summary_metrics}")

        # 4. DOWNSTREAM PARSED DATAFRAME
        print(f"\n4. DOWNSTREAM PARSED DATAFRAME:")
        print(f"   Shape: {df_parsed.shape}")
        print(f"   Columns: {list(df_parsed.columns)}")
        if not df_parsed.empty:
            print(f"   Head (5 rows):")
            print(df_parsed.head(5).to_string())
            print(f"   Tail (5 rows):")
            print(df_parsed.tail(5).to_string())

# Run Investigation 1: QE (T+DD)
inspect_block_pipeline("QE (T+DD)", [0, 43, 87])

# Run Investigation 2: QE (IDD)
inspect_block_pipeline("QE (IDD)", [0, 7, 15])

# Run Investigation 3: Illumination
inspect_block_pipeline("Illumination", [0])

# Run Investigation 4: Optimized
inspect_block_pipeline("Optimized", [0, 1, 2, 3])
