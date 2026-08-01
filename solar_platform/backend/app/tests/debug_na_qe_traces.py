import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.pipeline import UnifiedParserPipeline
from app.services.experiment_grouper import ExperimentGrouperEngine

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

with open(wb_path, "rb") as f:
    contents = f.read()

pipeline = UnifiedParserPipeline()
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "debug_na_session")
grouper = ExperimentGrouperEngine()

sheet_audit = next(s for s in audit_result.sheets if s.sheet_name == "NA")
exps = grouper.group_sheet_blocks_into_experiments(sheet_audit)

for e in exps:
    print(f"\nExp ID: '{e.experiment_id}', Target: '{e.target_name}', Blocks Count: {len(e.blocks)}")
    df = e.combined_dataframe
    print("  DF shape:", df.shape)
    print("  DF columns:", list(df.columns))
    if "Trace_ID" in df.columns:
        print("  Trace_ID unique values:", df["Trace_ID"].unique())
