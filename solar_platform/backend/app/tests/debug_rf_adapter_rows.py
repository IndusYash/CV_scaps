import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.pipeline import UnifiedParserPipeline
from app.adapters.rf_adapter import RandomForestMLAdapter

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

with open(wb_path, "rb") as f:
    contents = f.read()

pipeline = UnifiedParserPipeline()
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "debug_rf_session")

sheet = next(s for s in audit_result.sheets if s.sheet_name == "QE (T+DD)")
b1 = sheet.blocks[0]
ir = b1.intermediate_representation

print("Axis values count:", len(ir.axes[0].values))
print("Targets count:", len(ir.targets))
for idx, t in enumerate(ir.targets):
    print(f"  Target {idx}: name='{t.name}', shape={t.data_shape}, matrix_len={len(t.dense_matrix)}")
    df_long = RandomForestMLAdapter().transform_to_long_dataframe(ir, target_index=idx, block_id=b1.block_id)
    print(f"  --> Long DF shape for target {idx}: {df_long.shape}")
