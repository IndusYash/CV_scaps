import sys
import openpyxl
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

wb = openpyxl.load_workbook(wb_path, data_only=True)
sheet = wb["QE (T+DD)"]

print("=== OPENPYXL DIRECT SHEET INSPECTION (ROWS 25 TO 32) ===")
for r in range(25, 33):
    row_cells = []
    for c in range(1, 20):
        val = sheet.cell(row=r, column=c).value
        if val is not None:
            row_cells.append(f"C{c-1}:{val}")
    if row_cells:
        print(f"Row {r:2d}: " + " | ".join(row_cells))
