"""
Deep inspect for Interface density and ETL (all columns shown).
"""
from __future__ import annotations
import sys, io
import openpyxl
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

wb_path = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\Data_file.xlsx")
wb = openpyxl.load_workbook(wb_path, data_only=True)

def row_all(ws, row_1idx):
    return [(c-1, ws.cell(row=row_1idx, column=c).value)
            for c in range(1, ws.max_column + 1)
            if ws.cell(row=row_1idx, column=c).value is not None]

def empty_boundaries(ws):
    return [c-1 for c in range(1, ws.max_column+1)
            if all(ws.cell(row=r, column=c).value is None
                   for r in range(1, min(ws.max_row+1, 30)))]

for sheet_name in ["ETL", "Interface density"]:
    ws = wb[sheet_name]
    print(f"{'='*72}")
    print(f"SHEET: {sheet_name!r}  ({ws.max_row} rows x {ws.max_column} cols)")
    print("  Full header (row 0, non-empty):")
    for ci, v in row_all(ws, 1):
        print(f"    col {ci:2d}: {str(v)!r}")
    for r in range(2, min(ws.max_row+1, 7)):
        print(f"  row {r-1}: {row_all(ws, r)}")
    # also row at halfway and last 2 rows
    mid = ws.max_row // 2
    print(f"  row {mid} (mid): {row_all(ws, mid+1)}")
    print(f"  row {ws.max_row-2}: {row_all(ws, ws.max_row-1)}")
    print(f"  row {ws.max_row-1}: {row_all(ws, ws.max_row)}")
    print(f"  Empty col boundaries: {empty_boundaries(ws)}")
    if ws.merged_cells.ranges:
        print(f"  Merged cells: {[str(m) for m in ws.merged_cells.ranges]}")
    print()
