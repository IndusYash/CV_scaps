import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.pipeline import UnifiedParserPipeline

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

with open(wb_path, "rb") as f:
    contents = f.read()

pipeline = UnifiedParserPipeline()
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "debug_p4_session")

for s in ["Illumination", "NA"]:
    sheet_audit = next(sa for sa in audit_result.sheets if sa.sheet_name == s)
    print(f"\nSheet '{s}': Blocks Count = {len(sheet_audit.blocks)}")
    for b in sheet_audit.blocks:
        ir = b.intermediate_representation
        if ir:
            print(f"  Block '{b.block_id}', BBox={b.bounding_box}, Category={b.category}")
            for t in ir.targets:
                print(f"    Target '{t.name}' ({t.unit}), shape={t.data_shape}, column_labels={t.column_labels}")
