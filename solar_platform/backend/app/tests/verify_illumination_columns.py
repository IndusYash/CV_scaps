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
sheet = next((s for s in doc_model.sheets if s.sheet_name == "Illumination"), None)

print("=== ILLUMINATION RAW EXCEL COLUMNS INSPECTION ===")
print(f"Sheet dimensions: {sheet.row_count} rows x {sheet.col_count} cols")

# Inspect rows 0 to 5 for all columns in sheet
for r in range(5):
    row_str = []
    for c in range(sheet.col_count):
        cell = sheet.get_cell(r, c)
        val = cell.formatted_value if cell else ""
        row_str.append(f"Col {c}: '{val}'")
    print(f"Row {r}: " + " | ".join(row_str[:6]))
    if sheet.col_count > 6:
        print(f"        " + " | ".join(row_str[6:]))

# Also inspect IR generated for block_Illumination_1
b0 = TableSegmenter().segment_sheet(sheet)[0]
ir = IRGeneratorEngine().generate_ir_for_block(sheet, b0)

print("\n=== GENERATED IR OBJECT ===")
print(f"Block bounding box: {b0.bounding_box}")
print(f"Independent Axis: {ir.axes[0].name} ({ir.axes[0].unit}), count={len(ir.axes[0].values)}")
print(f"Targets count: {len(ir.targets)}")
for t in ir.targets:
    print(f"Target: Name='{t.name}', Unit='{t.unit}', Shape={t.data_shape}")
    print(f"Target Column Labels ({len(t.column_labels)}): {t.column_labels}")
