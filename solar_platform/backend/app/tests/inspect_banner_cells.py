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

print(f"=== INSPECTING QE (T+DD) CELLS IN ROWS 0 TO 10, COLS 0 TO 5 ===")
for r in range(12):
    row_str = []
    for c in range(6):
        cell = sheet.get_cell(r, c)
        row_str.append(f"[{r},{c}]={cell.formatted_value if cell else 'None'}")
    print(" | ".join(row_str))
