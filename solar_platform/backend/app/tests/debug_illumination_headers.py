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
sheet = next(s for s in doc_model.sheets if s.sheet_name == "Illumination")

for r in range(5):
    row_cells = [f"c{c}:'{sheet.get_cell(r, c).formatted_value}'" for c in range(6) if sheet.get_cell(r, c)]
    print(f"Row {r}: " + " | ".join(row_cells))
