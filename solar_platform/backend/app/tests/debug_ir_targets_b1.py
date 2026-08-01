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
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "debug_ir_session")

sheet = next(s for s in audit_result.sheets if s.sheet_name == "QE (T+DD)")
b1 = sheet.blocks[0]
ir = b1.intermediate_representation

print("IR targets count:", len(ir.targets))
for idx, t in enumerate(ir.targets):
    print(f"Target {idx}: name='{t.name}', unit='{t.unit}', shape={t.data_shape}, column_labels={t.column_labels}")
