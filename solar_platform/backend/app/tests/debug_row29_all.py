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

print("=== ALL NON-NONE CELLS IN ROW 28 AND 29 OF QE (T+DD) ===")
for r in [28, 29]:
    for c in range(sheet.col_count):
        cell = sheet.get_cell(r, c)
        if cell and cell.formatted_value:
            print(f"Row {r}, Col {c}: formatted='{cell.formatted_value}'")
