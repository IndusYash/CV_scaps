from __future__ import annotations

import os
import sys
import pandas as pd

sys.path.insert(0, ".")

from app.readers.excel_reader import ExcelInputReader

reader = ExcelInputReader()

def find_path(fn):
    for p in [fn, os.path.join("..", fn), os.path.join("solar_platform", fn)]:
        if os.path.exists(p): return p
    return ""

for wb_name, sheet_name in [("sample_data.xlsx", "ETL"), ("Data file.xlsx", "Defect Density"), ("Ca3PI3.xlsx", "Unoptimized")]:
    wb_path = find_path(wb_name)
    if not wb_path: continue
    with open(wb_path, "rb") as f:
        doc = reader.read(wb_name, f.read())
    sheet = next((s for s in doc.sheets if s.sheet_name == sheet_name), None)
    if not sheet: continue

    print(f"\n--- {wb_name} | Sheet: '{sheet_name}' ---")
    for r in range(min(4, sheet.row_count)):
        row_str = [sheet.get_cell(r, c).formatted_value for c in range(min(12, sheet.col_count))]
        print(f"  Row {r}: {row_str}")
