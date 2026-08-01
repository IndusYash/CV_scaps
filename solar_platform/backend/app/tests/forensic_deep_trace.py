import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.pipeline import UnifiedParserPipeline
from app.services.ml_engine import train_sheet_models, predict_curve_comparison, get_trained_session

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

with open(wb_path, "rb") as f:
    contents = f.read()

pipeline = UnifiedParserPipeline()
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "file_forensic_002")

print("=== DEEP TRACE ON TRANSFORMED DATAFRAMES AND PREDICTIONS ===")

for sheet_name in ["QE (T+DD)", "QE (IDD)", "Illumination", "Optimized"]:
    sheet_audit = next((s for s in audit_result.sheets if s.sheet_name == sheet_name), None)
    if not sheet_audit: continue
    
    print(f"\n==========================================================================================")
    print(f"SHEET: '{sheet_name}'")
    print(f"==========================================================================================")
    
    # Train models for this sheet
    raw_reader = ExcelInputReader()
    doc_model = raw_reader.read(wb_path.name, contents)
    
    sheet_dict = parsed_data.get(sheet_name, {})
    
    for idx, b in enumerate(sheet_audit.blocks[:3]): # Check first 3 blocks
        b_id = b.block_id
        df = sheet_dict.get(b_id, pd.DataFrame())
        
        print(f"\n--- Block: {b_id} (Semantic Label: {b.semantic_label}) ---")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check sorting and parameter grouping within this block's dataframe
        x_col = "Wavelength" if "Wavelength" in df.columns else ("Voltage" if "Voltage" in df.columns else df.columns[0])
        y_col = [c for c in df.columns if c not in [x_col, "Run", "Parameter_1", "Thickness", "Defect Density"]][0] if len(df.columns) > 1 else df.columns[-1]
        
        print(f"Primary X-col: '{x_col}', Y-col: '{y_col}'")
        
        # Inspect first 15 rows of raw DF
        print("First 15 DF rows:")
        print(df.head(15).to_string())
        
        # Grouping check: Does X reset or repeat?
        x_diffs = df[x_col].diff().dropna()
        resets = (x_diffs < 0).sum()
        print(f"X-axis value resets (where x_next < x_prev): {resets}")
        if resets > 0:
            reset_indices = df.index[df[x_col].diff() < 0].tolist()
            print(f"X-reset row indices: {reset_indices}")
            for r_idx in reset_indices[:3]:
                print(f"  Row {r_idx-1}: X={df.loc[r_idx-1, x_col]}, Y={df.loc[r_idx-1, y_col]}  -->  Row {r_idx}: X={df.loc[r_idx, x_col]}, Y={df.loc[r_idx, y_col]}")
