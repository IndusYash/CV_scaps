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

bbox = b0.bounding_box
cols = list(range(bbox.start_col, bbox.end_col + 1))

# Find header row
header_row_idx = bbox.start_row
for r in range(bbox.start_row, min(bbox.start_row + 40, bbox.end_row + 1)):
    numeric_count = sum(
        1 for c in cols
        if sheet.get_cell(r, c) and sheet.get_cell(r, c).is_numeric()
    )
    total = len(cols)
    if numeric_count / total >= 0.5:
        header_row_idx = max(bbox.start_row, r - 1)
        break

print(f"Detected header_row_idx: {header_row_idx}")

for r in range(bbox.start_row, header_row_idx):
    for c in [0] + cols:
        cell = sheet.get_cell(r, c)
        if cell and cell.formatted_value:
            txt = cell.formatted_value.strip()
            p_name, p_unit = engine._identify_param_name(txt)
            if p_name:
                print(f"Row {r}, Col {c}: text='{txt}' -> p_name='{p_name}', p_unit='{p_unit}'")
                
                num_val = None
                if ":" in txt or "=" in txt:
                    rhs = re.split(r"[:=]", txt)[-1].strip()
                    from app.services.preprocessor import extract_numeric_value
                    num_val = extract_numeric_value(rhs)
                    print(f"   rhs='{rhs}' -> extract_numeric_value={num_val}")
                
                if num_val is None:
                    for val_col in cols:
                        v_cell = sheet.get_cell(r, val_col)
                        if v_cell and v_cell.to_float() is not None:
                            num_val = v_cell.to_float()
                            print(f"   found num_val={num_val} at col {val_col}")
                            break
