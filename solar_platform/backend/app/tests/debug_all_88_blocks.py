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
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "debug_88_session")

sheet = next(s for s in audit_result.sheets if s.sheet_name == "QE (T+DD)")
print(f"Total blocks in {sheet.sheet_name}: {len(sheet.blocks)}")

params_list = []
for idx, b in enumerate(sheet.blocks):
    ir = b.intermediate_representation
    p_dict = {p.name: p.values[0] for p in ir.parameters} if ir else {}
    params_list.append((b.block_id, p_dict))

print("First 10 blocks:", params_list[:10])

# Check for duplicates
tup_list = [tuple(p.items()) for _, p in params_list]
print("Unique parameter tuples count:", len(set(tup_list)))
