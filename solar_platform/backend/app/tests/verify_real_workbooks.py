from __future__ import annotations

import os
import sys
import pandas as pd

sys.path.insert(0, ".")

from app.pipeline import UnifiedParserPipeline
from app.services.parser import parse_excel_workbook

WORKBOOKS = [
    "sample_data.xlsx",
    "Ca3PI3.xlsx",
    "Data file.xlsx",
]

def find_workbook_path(filename: str) -> str:
    candidates = [
        filename,
        os.path.join("..", filename),
        os.path.join("solar_platform", filename),
        os.path.join("..", "solar_platform", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""

def run_real_workbook_verification():
    pipeline = UnifiedParserPipeline()
    matrix_rows = []

    print("=" * 100)
    print("      REAL SCAPS WORKBOOK PARSER VALIDATION & COMPARISON REPORT")
    print("=" * 100)

    for wb_name in WORKBOOKS:
        wb_path = find_workbook_path(wb_name)
        if not wb_path:
            print(f"\n[WARNING] Workbook '{wb_name}' not found locally. Skipping.")
            continue

        print(f"\n\n>>> TESTING WORKBOOK: {wb_name} (Path: {wb_path})")
        with open(wb_path, "rb") as f:
            contents = f.read()

        # Run Legacy Parser
        legacy_results = {}
        try:
            legacy_results = parse_excel_workbook(contents)
        except Exception as e:
            legacy_err = str(e)

        # Run New Unified Parser Pipeline
        audit_result, new_results = pipeline.parse_workbook(wb_name, contents, f"file_id_{wb_name}")

        print(f"\n--- STRUCTURAL BLOCK DISCOVERY DEBUGGING REPORT ({wb_name}) ---")
        for sheet_audit in audit_result.sheets:
            print(f"\nSheet: '{sheet_audit.sheet_name}' (Status: {sheet_audit.status})")
            print(f"  Rows: {sheet_audit.row_count}, Cols: {sheet_audit.col_count}")
            print(f"  Blocks Discovered: {len(sheet_audit.blocks)}")
            
            for b in sheet_audit.blocks:
                bbox = b.bounding_box
                print(f"    * Block ID: {b.block_id}")
                print(f"      - Category: {b.category}")
                print(f"      - Bounding Box: ({bbox.start_row}, {bbox.start_col}) -> ({bbox.end_row}, {bbox.end_col})")
                print(f"      - Parsing Confidence: {b.parsing_confidence:.2f}")
                print(f"      - Semantic Label: {b.semantic_label}")
                if b.decision_traces:
                    for dt in b.decision_traces:
                        print(f"      - Decision: {dt.decision} (Evidence: {dt.evidence})")

            # Evaluate Legacy vs New for this sheet
            legacy_status = "PASS" if sheet_audit.sheet_name in legacy_results else "FAIL/CRASH"
            new_status = "PASS"
            if sheet_audit.status == "EMPTY":
                new_status = "UNSUPPORTED"
            elif not sheet_audit.blocks:
                new_status = "UNSUPPORTED"

            ir_produced = "YES" if sheet_audit.blocks and sheet_audit.blocks[0].intermediate_representation else "NO"
            ml_compat = "YES" if sheet_audit.sheet_name in new_results else "NO"
            sem_type = sheet_audit.blocks[0].semantic_label if sheet_audit.blocks else "NONE"
            blocks_cnt = len(sheet_audit.blocks)
            warnings_str = "; ".join(sheet_audit.warnings + (sheet_audit.blocks[0].validation_warnings if sheet_audit.blocks else []))
            if not warnings_str:
                warnings_str = "None"

            matrix_rows.append({
                "Workbook": wb_name,
                "Sheet": sheet_audit.sheet_name,
                "Legacy Result": legacy_status,
                "New Parser Result": "SUCCESS" if new_status == "PASS" else new_status,
                "Detected Blocks": blocks_cnt,
                "Semantic Type": sem_type,
                "IR Produced?": ir_produced,
                "ML Compatible?": ml_compat,
                "Warnings": warnings_str,
                "Status": new_status,
            })

    print("\n\n" + "=" * 100)
    print("                    LEGACY VS NEW PARSER COMPATIBILITY MATRIX")
    print("=" * 100)
    df_matrix = pd.DataFrame(matrix_rows)
    print(df_matrix.to_string(index=False))

if __name__ == "__main__":
    run_real_workbook_verification()
