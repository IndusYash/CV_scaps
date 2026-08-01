import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.layout.segmenter import TableSegmenter

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

with open(wb_path, "rb") as f:
    contents = f.read()

doc_model = ExcelInputReader().read(wb_path.name, contents)
sheet = next((s for s in doc_model.sheets if s.sheet_name == "QE (T+DD)"), None)
b0 = TableSegmenter().segment_sheet(sheet)[0]

cols = list(range(b0.bounding_box.start_col, b0.bounding_box.end_col + 1))
print(f"Cols: {cols}")

for r in range(b0.bounding_box.start_row, 35):
    h_text = sheet.get_cell(r, cols[0]).formatted_value if sheet.get_cell(r, cols[0]) else ""
    c1 = sheet.get_cell(r+1, cols[0]).to_float() if sheet.get_cell(r+1, cols[0]) else None
    c2 = sheet.get_cell(r+2, cols[0]).to_float() if sheet.get_cell(r+2, cols[0]) else None
    c3 = sheet.get_cell(r+3, cols[0]).to_float() if sheet.get_cell(r+3, cols[0]) else None
    print(f"Row {r:2d}: Text[0]='{h_text[:20]}', r+1={c1}, r+2={c2}, r+3={c3}")
