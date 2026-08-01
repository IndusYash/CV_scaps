import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

with open(wb_path, "rb") as f:
    contents = f.read()

doc_model = ExcelInputReader().read(wb_path.name, contents)
sheet = next((s for s in doc_model.sheets if s.sheet_name == "QE (T+DD)"), None)

print("=== SEARCHING ROWS 0 TO 30 FOR BATCH PARAMETER TABLE ===")
for r in range(31):
    row_str = []
    for c in range(min(15, sheet.col_count)):
        cell = sheet.get_cell(r, c)
        if cell and cell.formatted_value:
            row_str.append(f"C{c}: '{cell.formatted_value}'")
    if row_str:
        print(f"Row {r:2d}: " + " | ".join(row_str))
