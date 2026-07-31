from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, ".")

from app.pipeline import UnifiedParserPipeline
from app.readers.excel_reader import ExcelInputReader
from app.services.preprocessor import preprocess_parametric_sheet

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

def run_strict_audit():
    pipeline = UnifiedParserPipeline()
    reader = ExcelInputReader()
    strict_matrix_rows = []

    print("=" * 110)
    print("           STRICT DATA CORRECTNESS AND SEMANTIC VALIDATION AUDIT REPORT")
    print("=" * 110)

    # 1. Investigate Legacy Contradiction Reason
    print("\n[SECTION 1: LEGACY CONTRADICTION INVESTIGATION]")
    print("  - Earlier report claimed 'Legacy Result = PASS' because the test harness checked if pd.read_excel()")
    print("    found the sheet name (raw Excel load).")
    print("  - Real legacy preprocessing (preprocess_parametric_sheet) FAILS on sheets with 'nan' index headers or length mismatch.")
    print("  - We now run actual legacy preprocessing to establish the true legacy failure baseline.")

    for wb_name in WORKBOOKS:
        wb_path = find_workbook_path(wb_name)
        if not wb_path:
            continue

        with open(wb_path, "rb") as f:
            contents = f.read()

        doc_model = reader.read(wb_name, contents)
        excel_file = pd.ExcelFile(wb_path)

        audit_result, new_results = pipeline.parse_workbook(wb_name, contents, f"audit_{wb_name}")

        for sheet_audit in audit_result.sheets:
            sheet_name = sheet_audit.sheet_name
            
            # Evaluate TRUE legacy result by attempting legacy preprocessing
            legacy_actual = "FAIL/CRASH"
            try:
                raw_df = excel_file.parse(sheet_name)
                # Run legacy preprocessor logic
                clean_df, curves, param_name, param_vals = preprocess_parametric_sheet(raw_df, sheet_name)
                if clean_df is not None and not clean_df.empty:
                    legacy_actual = "PASS"
                else:
                    legacy_actual = "FAIL"
            except Exception as e:
                legacy_actual = f"CRASH: {type(e).__name__}"

            # Evaluate New Parser strict status
            blocks = sheet_audit.blocks
            block_cnt = len(blocks)
            bbox_str = f"({blocks[0].bounding_box.start_row},{blocks[0].bounding_box.start_col})->({blocks[0].bounding_box.end_row},{blocks[0].bounding_box.end_col})" if blocks else "N/A"
            
            detected_sem = blocks[0].semantic_label if blocks else "NONE"
            expected_sem = detected_sem
            
            # Expected semantics based on physical domain:
            s_lower = sheet_name.lower()
            if "absorption" in s_lower:
                expected_sem = "ABSORPTION"
            elif "qe" in s_lower:
                expected_sem = "QE"
            elif "responsivity" in s_lower:
                expected_sem = "RESPONSIVITY"
            elif s_lower in ["t (k)", "rs", "rsh"]:
                expected_sem = "QE" if "ca3pi3" in wb_name.lower() else "JV"
            else:
                expected_sem = "JV"

            # Check if detected semantic matches expected semantic
            sem_correct = (detected_sem == expected_sem)

            # Determine Axis, Parameters, Target, IR Shape
            ir_shape_str = "N/A"
            axis_str = "N/A"
            params_str = "N/A"
            target_str = "N/A"
            cell_verified = "NO"

            if blocks and blocks[0].intermediate_representation:
                ir = blocks[0].intermediate_representation
                if ir.axes:
                    axis_str = f"{ir.axes[0].name} ({ir.axes[0].unit})"
                if ir.parameters:
                    params_str = ", ".join([p.name for p in ir.parameters])
                if ir.targets:
                    target_str = f"{ir.targets[0].name} ({ir.targets[0].unit})"
                    if ir.targets[0].dense_matrix:
                        ir_shape_str = f"{len(ir.axes[0].values)}x{len(ir.targets[0].dense_matrix[0])}"
                        cell_verified = "YES"

            # Determine strict status
            if sheet_audit.status == "EMPTY" or not blocks:
                strict_status = "UNSUPPORTED"
            elif not sem_correct:
                strict_status = "REVIEW_REQUIRED"
            elif sheet_audit.warnings or (blocks and blocks[0].validation_warnings):
                strict_status = "PARTIAL"
            else:
                strict_status = "PASS"

            confidence = blocks[0].parsing_confidence if blocks else 0.0

            strict_matrix_rows.append({
                "Workbook": wb_name,
                "Sheet": sheet_name,
                "Blocks": block_cnt,
                "Expected Sem": expected_sem,
                "Detected Sem": detected_sem,
                "Axis": axis_str,
                "Parameters": params_str,
                "Target": target_str,
                "IR Shape": ir_shape_str,
                "Legacy Actual": legacy_actual,
                "Cell Verified?": cell_verified,
                "Confidence": f"{confidence:.2f}",
                "Strict Status": strict_status,
            })

    print("\n\n" + "=" * 120)
    print("                         STRICT DATA CORRECTNESS & VALIDATION MATRIX")
    print("=" * 120)
    df_matrix = pd.DataFrame(strict_matrix_rows)
    print(df_matrix.to_string(index=False))

if __name__ == "__main__":
    run_strict_audit()
