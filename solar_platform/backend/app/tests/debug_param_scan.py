import sys
import re
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.layout.segmenter import TableSegmenter
from app.domain.ir_generator import IRGeneratorEngine

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

with open(wb_path, "rb") as f:
    contents = f.read()

doc_model = ExcelInputReader().read(wb_path.name, contents)
sheet = next((s for s in doc_model.sheets if s.sheet_name == "QE (T+DD)"), None)
b0 = TableSegmenter().segment_sheet(sheet)[0]
engine = IRGeneratorEngine()

ir = engine.generate_ir_for_block(sheet, b0)

print(f"Block 0 bbox: {b0.bounding_box}")
print(f"IR parameters count: {len(ir.parameters)}")
for p in ir.parameters:
    print(f"  Param: name='{p.name}', unit='{p.unit}', values={p.values}")

# Manually test _extract_parameters
cols = list(range(b0.bounding_box.start_col, b0.bounding_box.end_col + 1))
header_row_idx = 31
header_texts = [sheet.get_cell(header_row_idx, c).formatted_value if sheet.get_cell(header_row_idx, c) else f"Col_{c}" for c in cols]

res = engine._extract_parameters(sheet, b0.bounding_box, header_row_idx, cols, cols[0], header_texts)
print(f"Direct _extract_parameters result: {[p.name for p in res]}")
