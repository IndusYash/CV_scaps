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

sheet_audit = next((s for s in audit_result.sheets if s.sheet_name == "QE (T+DD)"), None)
b0 = sheet_audit.blocks[0]

print(f"Block 0 ID: {b0.block_id}")
print(f"Block 0 decision traces:")
for t in b0.decision_traces:
    print(f"  {t.decision_type}: {t.decision} ({t.evidence})")

ir = b0.intermediate_representation
print(f"IR parameters: {[p.name for p in ir.parameters]}")
