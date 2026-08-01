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
doc_model = ExcelInputReader().read(wb_path.name, contents)
sheet = next(s for s in doc_model.sheets if s.sheet_name == "QE (T+DD)")

audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "debug_session")
sheet_audit = next(s for s in audit_result.sheets if s.sheet_name == "QE (T+DD)")
b1 = sheet_audit.blocks[0]

bbox = b1.bounding_box
cols = list(range(bbox.start_col, bbox.end_col + 1))
header_row_idx = 31

header_texts = [
    sheet.get_cell(header_row_idx, c).formatted_value if sheet.get_cell(header_row_idx, c) else f"Col_{c}"
    for c in cols
]

print("header_texts:", header_texts)

for col in cols:
    h_text = header_texts[cols.index(col)]
    h_lower = h_text.lower()
    cell_0 = sheet.get_cell(0, col)
    c0_text = cell_0.formatted_value.lower() if (cell_0 and cell_0.formatted_value) else ""
    print(f"Col {col}: h_text='{h_text}', c0_text='{c0_text}'")
