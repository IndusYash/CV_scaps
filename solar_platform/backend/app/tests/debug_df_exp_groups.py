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
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "debug_session")
grouper = ExperimentGrouperEngine()

sheet = next(s for s in audit_result.sheets if s.sheet_name == "QE (T+DD)")
exps = grouper.group_sheet_blocks_into_experiments(sheet)
exp_qe = next(e for e in exps if e.target_name == "QE")
df = exp_qe.combined_dataframe

print("Columns in df:", list(df.columns))
print("Unique Block_IDs:", len(df["Block_ID"].unique()))
print("Groupby Block_ID len:", len(df.groupby("Block_ID")))
print("Groupby ['Thickness', 'Defect_Density'] len:", len(df.groupby(["Thickness", "Defect_Density"])))
print("Groupby ['Temperature', 'Thickness', 'Defect_Density'] len:", len(df.groupby(["Temperature", "Thickness", "Defect_Density"])))
