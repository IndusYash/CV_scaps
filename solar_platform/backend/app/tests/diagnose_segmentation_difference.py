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
segmenter = TableSegmenter()

for s_name in ["T (K)", "NA", "Rs", "Rsh"]:
    sheet = next(s for s in doc_model.sheets if s.sheet_name == s_name)
    blocks = segmenter.segment_sheet(sheet)
    print(f"\nSheet: '{s_name}' -> Segmented Blocks Count: {len(blocks)}")
    for b in blocks:
        print(f"  Block ID='{b.block_id}', BBox: rows={b.bounding_box.start_row}..{b.bounding_box.end_row}, cols={b.bounding_box.start_col}..{b.bounding_box.end_col}")
    
    col_has_data = []
    for c in range(0, sheet.col_count):
        has_val = any(sheet.get_cell(r, c) is not None and sheet.get_cell(r, c).formatted_value for r in range(0, sheet.row_count))
        col_has_data.append((c, has_val))
    
    print("  Columns 7 to 13 data presence:", col_has_data[7:14])
