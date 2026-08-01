import sys
import io
import pandas as pd
import numpy as np
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

reader = ExcelInputReader()
doc_model = reader.read(wb_path.name, contents)
doc_sheets = {s.sheet_name: s for s in doc_model.sheets}

pipeline = UnifiedParserPipeline()
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "file_forensic_003")

print("=== INVESTIGATION 9: SOURCE POINT ROUND-TRIP TEST ===")

# Test 10 specific source points from raw Excel cells down to API dataframe
test_points = [
    # (sheet, cell_row, cell_col, description)
    ("Optimized", 1, 0, "Optimized JV V0 (0.0 V)"),
    ("Optimized", 1, 1, "Optimized JV J0 (25.5144686 mA/cm2)"),
    ("Optimized", 61, 0, "Optimized JV V_max (1.24 V)"),
    ("Optimized", 61, 1, "Optimized JV J_max (-19.0077417 mA/cm2)"),
    ("Illumination", 1, 0, "Illumination V0 (0.0 V)"),
    ("Illumination", 1, 1, "Illumination Series 1 J0 (25.5144686)"),
    ("Illumination", 1, 2, "Illumination Series 2 J0 (32.277407)"),
    ("QE (T+DD)", 1, 0, "QE (T+DD) Block 1 Lambda 0 (300 nm)"),
    ("QE (T+DD)", 1, 1, "QE (T+DD) Block 1 QE 0 (89.29%)"),
    ("QE (IDD)", 1, 1, "QE (IDD) Block 1 QE 0 (90.11%)"),
]

print(f"{'Source Location':<35} | {'Raw Cell Value':<20} | {'IR Value':<20} | {'DF Value':<20} | {'Status':<6}")
print("-" * 110)

for sname, r, c, desc in test_points:
    doc_sheet = doc_sheets.get(sname)
    cell = doc_sheet.get_cell(r, c) if doc_sheet else None
    raw_val = cell.raw_value if cell else None
    
    # Trace to IR
    sheet_audit = next((s for s in audit_result.sheets if s.sheet_name == sname), None)
    ir_val = "N/A"
    df_val = "N/A"
    status = "FAIL"
    
    if sheet_audit and sheet_audit.blocks:
        b0 = sheet_audit.blocks[0]
        ir = b0.intermediate_representation
        if ir and ir.axes and len(ir.axes[0].values) > r - 1:
            ir_val = ir.axes[0].values[r - 1]
        
        # Check DF
        df = parsed_data.get(sname, {}).get(b0.block_id, pd.DataFrame())
        if not df.empty and len(df) > r - 1:
            df_val = df.iat[r - 1, 0]
            
    if raw_val is not None and (ir_val != "N/A" or df_val != "N/A"):
        # Check tolerance
        try:
            if abs(float(raw_val) - float(df_val)) < 1e-3 or abs(float(raw_val) - float(ir_val)) < 1e-3:
                status = "PASS"
        except:
            pass
            
    print(f"{desc:<35} | {str(raw_val):<20} | {str(ir_val):<20} | {str(df_val):<20} | {status:<6}")
