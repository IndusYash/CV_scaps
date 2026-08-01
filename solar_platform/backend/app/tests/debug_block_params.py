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
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "test_session")

sheet = next((s for s in audit_result.sheets if s.sheet_name == "QE (T+DD)"), None)
for idx, b in enumerate(sheet.blocks[:10]):
    ir = b.intermediate_representation
    p_names = {p.name: p.values for p in ir.parameters} if ir else {}
    print(f"Block {idx+1:2d} ({b.block_id}): {p_names}")
