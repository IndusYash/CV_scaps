import sys
import openpyxl
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.pipeline import UnifiedParserPipeline
from app.services.experiment_grouper import ExperimentGrouperEngine

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

wb = openpyxl.load_workbook(wb_path, data_only=True)

print("==========================================================================================")
print("             1. RAW EXCEL STRUCTURAL INVENTORY FOR T(K), NA, Rs, Rsh                       ")
print("==========================================================================================")

target_sheets = ["T (K)", "NA", "Rs", "Rsh"]

for s_name in target_sheets:
    sheet = wb[s_name]
    print(f"\n--- SHEET: '{s_name}' (max_row={sheet.max_row}, max_column={sheet.max_column}) ---")
    for r in range(1, min(15, sheet.max_row + 1)):
        row_vals = []
        for c in range(1, min(25, sheet.max_column + 1)):
            v = sheet.cell(row=r, column=c).value
            if v is not None:
                row_vals.append(f"C{c-1}:'{v}'")
        if row_vals:
            print(f"Row {r-1:2d}: " + " | ".join(row_vals))

print("\n==========================================================================================")
print("             2. PIPELINE TRACE ACROSS ALL STAGES FOR T(K), NA, Rs, Rsh                    ")
print("==========================================================================================")

with open(wb_path, "rb") as f:
    contents = f.read()

doc_model = ExcelInputReader().read(wb_path.name, contents)
pipeline = UnifiedParserPipeline()
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "investigate_session")
grouper = ExperimentGrouperEngine()

for s_name in target_sheets:
    sheet_audit = next((s for s in audit_result.sheets if s.sheet_name == s_name), None)
    print(f"\n==================== SHEET TRACE: '{s_name}' ====================")
    if not sheet_audit:
        print("  [FAIL] Sheet not found in audit_result!")
        continue

    print(f"  [STAGE 1] Raw DocumentModel Sheet: {s_name} (rows={sheet_audit.row_count}, cols={sheet_audit.col_count})")
    print(f"  [STAGE 2] TableSegmenter DataBlocks Count: {len(sheet_audit.blocks)}")
    
    for idx, b in enumerate(sheet_audit.blocks):
        ir = b.intermediate_representation
        print(f"\n    Block {idx+1}: ID='{b.block_id}', Category={b.category}, Semantic={b.semantic_label}, BBox={b.bounding_box}")
        if not ir:
            print("      [FAIL] IntermediateRepresentation is None!")
            continue
            
        print(f"      IR Version: {ir.ir_version}")
        print(f"      Axes Count: {len(ir.axes)}")
        if ir.axes:
            print(f"        Axis 0: name='{ir.axes[0].name}', unit='{ir.axes[0].unit}', values_count={len(ir.axes[0].values)}")
        print(f"      Targets Count: {len(ir.targets)}")
        for t in ir.targets:
            print(f"        Target: name='{t.name}', unit='{t.unit}', shape={t.data_shape}")
        print(f"      Summary Metrics Present: {bool(ir.summary_metrics)}")
        if ir.summary_metrics:
            print(f"        Summary Metrics Keys ({len(ir.summary_metrics)}): {list(ir.summary_metrics.keys())[:10]}")

    exps = grouper.group_sheet_blocks_into_experiments(sheet_audit)
    print(f"\n  [STAGE 3] ExperimentGrouper ScientificExperiments Count: {len(exps)}")
    for exp in exps:
        print(f"    - Experiment ID: '{exp.experiment_id}'")
        print(f"      Target: '{exp.target_name}' ({exp.target_unit}), Axis: '{exp.axis_name}' ({exp.axis_unit})")
        print(f"      Combined DF Shape: {exp.combined_dataframe.shape}")
        print(f"      DF Columns: {list(exp.combined_dataframe.columns)}")
