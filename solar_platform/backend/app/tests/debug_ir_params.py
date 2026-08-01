import sys
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
blocks = TableSegmenter().segment_sheet(sheet)
b0 = blocks[0]

print(f"Block 0 bbox: {b0.bounding_box}")

engine = IRGeneratorEngine()
ir = engine.generate_ir_for_block(sheet, b0)

print(f"IR Parameters count: {len(ir.parameters)}")
for p in ir.parameters:
    print(f"  Param: name='{p.name}', unit='{p.unit}', values={p.values}")
