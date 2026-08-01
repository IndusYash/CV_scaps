import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.pipeline import UnifiedParserPipeline
from app.domain.ir_generator import IRGeneratorEngine

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

with open(wb_path, "rb") as f:
    contents = f.read()

doc_model = ExcelInputReader().read(wb_path.name, contents)
sheet = next(s for s in doc_model.sheets if s.sheet_name == "QE (T+DD)")

pipeline = UnifiedParserPipeline()
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "debug_session")
sheet_audit = next(s for s in audit_result.sheets if s.sheet_name == "QE (T+DD)")
b1 = sheet_audit.blocks[0]

engine = IRGeneratorEngine()
ir = engine.generate_ir_for_block(sheet, b1)

print("Targets count:", len(ir.targets))
for t in ir.targets:
    print(f"  Target '{t.name}' ({t.unit}), shape={t.data_shape}, column_labels={t.column_labels}")
