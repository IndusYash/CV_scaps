import sys
import io
import pandas as pd
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.pipeline import UnifiedParserPipeline

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

with open(wb_path, "rb") as f:
    contents = f.read()

pipeline = UnifiedParserPipeline()
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "phase1_diag_session")

print("==========================================================================================")
print("                       PHASE 1 FORENSIC DIAGNOSTIC REPORT                                ")
print("==========================================================================================")

for sheet_name in ["Illumination", "QE (T+DD)", "QE (IDD)", "Optimized"]:
    sheet_audit = next((s for s in audit_result.sheets if s.sheet_name == sheet_name), None)
    if not sheet_audit: continue

    print(f"\nSHEET: {sheet_name}")
    print(f"  Block Count: {len(sheet_audit.blocks)}")

    for idx, b in enumerate(sheet_audit.blocks[:3]):
        ir = b.intermediate_representation
        if not ir: continue
        ax = ir.axes[0] if ir.axes else None
        
        print(f"\n  --- {b.block_id} ---")
        if ax:
            print(f"      Axis: Name='{ax.name}', Unit='{ax.unit}', Count={len(ax.values)}, Min={min(ax.values)}, Max={max(ax.values)}")
            print(f"            First 3 = {ax.values[:3]}, Last 3 = {ax.values[-3:]}")

        for p in ir.parameters:
            print(f"      Parameter: Name='{p.name}', Unit='{p.unit}', Values={p.values}")

        for t in ir.targets:
            print(f"      Target: Name='{t.name}', Unit='{t.unit}', Shape={t.data_shape}, Column Labels={t.column_labels}")
            if t.dense_matrix:
                flat_col0 = [row[0] for row in t.dense_matrix if row]
                valid_flt = [v for v in flat_col0 if v is not None and not pd.isna(v)]
                if valid_flt:
                    print(f"              Target Min={min(valid_flt)}, Max={max(valid_flt)}")
                    print(f"              First 3 = {valid_flt[:3]}, Last 3 = {valid_flt[-3:]}")
