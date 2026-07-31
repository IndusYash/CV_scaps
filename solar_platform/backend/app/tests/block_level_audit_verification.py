from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, ".")

from app.pipeline import UnifiedParserPipeline
from app.readers.excel_reader import ExcelInputReader

WORKBOOKS = [
    "sample_data.xlsx",
    "Ca3PI3.xlsx",
    "Data file.xlsx",
]

def find_path(fn: str) -> str:
    for p in [fn, os.path.join("..", fn), os.path.join("solar_platform", fn)]:
        if os.path.exists(p): return p
    return ""

def run_block_level_audit():
    pipeline = UnifiedParserPipeline()
    reader = ExcelInputReader()
    block_rows = []

    print("=" * 140)
    print("                    EXACT BLOCK-LEVEL DATA CORRECTNESS & SEMANTIC AUDIT MATRIX")
    print("=" * 140)

    for wb_name in WORKBOOKS:
        wb_path = find_path(wb_name)
        if not wb_path:
            continue

        with open(wb_path, "rb") as f:
            contents = f.read()

        audit_result, parsed_data = pipeline.parse_workbook(wb_name, contents, f"block_audit_{wb_name}")

        for sheet_audit in audit_result.sheets:
            sheet_name = sheet_audit.sheet_name
            s_lower = sheet_name.lower()

            if not sheet_audit.blocks:
                block_rows.append({
                    "Workbook": wb_name,
                    "Sheet": sheet_name,
                    "Block ID": "N/A",
                    "Bounding Box": "N/A",
                    "Category": "EMPTY",
                    "Axis": "N/A",
                    "Axis Unit": "N/A",
                    "Parameter(s)": "N/A",
                    "Param Unit(s)": "N/A",
                    "Target": "N/A",
                    "Target Unit": "N/A",
                    "Expected Sem": "NONE",
                    "Detected Sem": "NONE",
                    "Struct Conf": "0.00",
                    "Sem Conf": "0.00",
                    "Sci Validity": "1.00",
                    "IR Shape": "N/A",
                    "ML Result": "NO",
                    "Status": "UNSUPPORTED",
                    "Warnings": "Empty sheet",
                })
                continue

            for block in sheet_audit.blocks:
                bbox = block.bounding_box
                bbox_str = f"({bbox.start_row},{bbox.start_col})->({bbox.end_row},{bbox.end_col})"
                cat_str = str(block.category.value if hasattr(block.category, "value") else block.category)

                axis_name = "N/A"
                axis_unit = "N/A"
                params_str = "N/A"
                params_unit = "N/A"
                target_name = "N/A"
                target_unit = "N/A"
                ir_shape_str = "N/A"

                if block.intermediate_representation:
                    ir = block.intermediate_representation
                    if ir.summary_metrics:
                        category_str = "SUMMARY"
                        target_name = "Summary Metrics"
                        target_unit = "Various"
                        ir_shape_str = f"Scalars ({len(ir.summary_metrics)})"
                    else:
                        if ir.axes:
                            axis_name = ir.axes[0].name
                            axis_unit = ir.axes[0].unit
                        if ir.parameters:
                            params_str = ", ".join([p.name for p in ir.parameters])
                            params_unit = ", ".join([p.unit if p.unit else "scalar" for p in ir.parameters])
                        if ir.targets:
                            target_name = ir.targets[0].name
                            target_unit = ir.targets[0].unit
                            if ir.targets[0].dense_matrix:
                                r_cnt = len(ir.axes[0].values) if ir.axes else 0
                                c_cnt = len(ir.targets[0].dense_matrix[0]) if ir.targets[0].dense_matrix else 0
                                ir_shape_str = f"{r_cnt}x{c_cnt}"

                # Expected Semantic Deduction
                if "absorption" in s_lower:
                    exp_sem = "ABSORPTION"
                elif "qe" in s_lower:
                    exp_sem = "QE"
                elif "responsivity" in s_lower:
                    exp_sem = "RESPONSIVITY"
                elif s_lower in ["t+dd", "idd", "na", "t (k)", "rs", "rsh"] and "ca3pi3" in wb_name.lower() and cat_str == "SUMMARY":
                    exp_sem = "SUMMARY_METRICS"
                elif cat_str == "SUMMARY":
                    exp_sem = "SUMMARY_METRICS"
                else:
                    exp_sem = "JV"

                det_sem = block.semantic_label

                # Calibrated Confidence Scores
                struct_conf = block.parsing_confidence
                sem_conf = block.plugin_selection.selection_confidence if block.plugin_selection else 0.80
                sci_val = block.scientific_validity

                # Strict Parsing Status vs Scientific Validity
                if det_sem == exp_sem or (exp_sem == "SUMMARY_METRICS" and det_sem == "SUMMARY_METRICS"):
                    status_str = "PASS"
                elif det_sem != exp_sem:
                    status_str = "REVIEW_REQUIRED"
                else:
                    status_str = "PARTIAL"

                ml_res = "YES" if (sheet_name in parsed_data or cat_str == "SUMMARY") else "NO"
                warn_str = "; ".join(block.validation_warnings) if block.validation_warnings else "None"

                block_rows.append({
                    "Workbook": wb_name,
                    "Sheet": sheet_name,
                    "Block ID": block.block_id,
                    "Bounding Box": bbox_str,
                    "Category": cat_str,
                    "Axis": axis_name,
                    "Axis Unit": axis_unit,
                    "Parameter(s)": params_str,
                    "Param Unit(s)": params_unit,
                    "Target": target_name,
                    "Target Unit": target_unit,
                    "Expected Sem": exp_sem,
                    "Detected Sem": det_sem,
                    "Struct Conf": f"{struct_conf:.2f}",
                    "Sem Conf": f"{sem_conf:.2f}",
                    "Sci Validity": f"{sci_val:.2f}",
                    "IR Shape": ir_shape_str,
                    "ML Result": ml_res,
                    "Status": status_str,
                    "Warnings": warn_str,
                })

    df_res = pd.DataFrame(block_rows)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_block_level_audit()
