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

doc_model = ExcelInputReader().read(wb_path.name, contents)
pipeline = UnifiedParserPipeline()
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "inspect_session")

print("==========================================================================================")
print("                       9. OPTIMIZED BAND DATA SOURCE INSPECTION                           ")
print("==========================================================================================")

sheet_opt = next(s for s in audit_result.sheets if s.sheet_name == "Optimized")
b4 = sheet_opt.blocks[3]
ir4 = b4.intermediate_representation

print(f"Block ID: {b4.block_id}")
print(f"Bounding Box: start_row={b4.bounding_box.start_row}, end_row={b4.bounding_box.end_row}, start_col={b4.bounding_box.start_col}, end_col={b4.bounding_box.end_col}")

# Get raw openpyxl/document headers and first/last rows
doc_sheet_opt = next(s for s in doc_model.sheets if s.sheet_name == "Optimized")
headers = [doc_sheet_opt.get_cell(b4.bounding_box.start_row, c).formatted_value for c in range(b4.bounding_box.start_col, b4.bounding_box.end_col + 1)]
print("Headers:", headers)
print("Col Count:", len(headers))
print("Axis:", ir4.axes[0].name, f"({ir4.axes[0].unit})", "Values Count:", len(ir4.axes[0].values))
print("First 5 X values:", ir4.axes[0].values[:5])
print("Last 5 X values:", ir4.axes[0].values[-5:])

for idx, t in enumerate(ir4.targets):
    print(f"Target {idx}: name='{t.name}', unit='{t.unit}', shape={t.data_shape}")
    print(f"  First 5 row Y values: {t.dense_matrix[:5]}")
    print(f"  Last 5 row Y values: {t.dense_matrix[-5:]}")


print("\n==========================================================================================")
print("                       10. STANDARDIZED SUMMARY DATA AVAILABILITY                         ")
print("==========================================================================================")

for sheet_name in ["T+DD", "IDD", "T (K)", "Rs", "Rsh", "NA"]:
    sheet_audit = next((s for s in audit_result.sheets if s.sheet_name == sheet_name), None)
    if not sheet_audit:
        continue
    
    print(f"\nSHEET: {sheet_name}")
    for idx, b in enumerate(sheet_audit.blocks):
        ir = b.intermediate_representation
        if ir and ir.summary_metrics:
            print(f"  Block {idx+1} ({b.block_id}) SUMMARY METRICS:")
            for m_name, m_dict in ir.summary_metrics.items():
                print(f"    - {m_name}: {m_dict}")
