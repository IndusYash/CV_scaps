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
segmenter = TableSegmenter(max_adjacency_distance=1)

for s in doc_model.sheets:
    blocks = segmenter.segment_sheet(s)
    print(f"Sheet: '{s.sheet_name}' -> Blocks Count ({len(blocks)}):")
    for b in blocks:
        print(f"  Block ID='{b.block_id}', BBox: rows={b.bounding_box.start_row}..{b.bounding_box.end_row}, cols={b.bounding_box.start_col}..{b.bounding_box.end_col}")
