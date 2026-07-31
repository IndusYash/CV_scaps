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

sheets_to_inspect = [
    ("sample_data.xlsx", "Optimized"),
    ("Ca3PI3.xlsx", "T+DD"),
    ("Ca3PI3.xlsx", "IDD"),
    ("Ca3PI3.xlsx", "NA"),
    ("Ca3PI3.xlsx", "T (K)"),
    ("Ca3PI3.xlsx", "Rs"),
    ("Ca3PI3.xlsx", "Rsh"),
    ("Data file.xlsx", "Thickness_PVK"),
    ("Data file.xlsx", "Rs+Rsh"),
    ("Data file.xlsx", "Thick+DD(PVK)"),
    ("Data file.xlsx", "Optimized"),
    ("Data file.xlsx", "absorption"),
]

print("=" * 100)
print("              EXACT WORKBOOK CELL & HEADER GROUND TRUTH INSPECTION")
print("=" * 100)

for wb_name, sheet_name in sheets_to_inspect:
    wb_path = find_path(wb_name)
    if not wb_path: continue

    with open(wb_path, "rb") as f:
        doc = reader.read(wb_name, f.read())

    sheet = next((s for s in doc.sheets if s.sheet_name == sheet_name), None)
    if not sheet: continue

    print(f"\n--- {wb_name} | Sheet: '{sheet_name}' ({sheet.row_count} rows x {sheet.col_count} cols) ---")
    for r in range(min(5, sheet.row_count)):
        row_str = [sheet.get_cell(r, c).formatted_value for c in range(min(10, sheet.col_count))]
        print(f"  Row {r}: {row_str}")
