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
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "inventory_session")

grouper = ExperimentGrouperEngine()

print("==========================================================================================")
print("                       EXPECTED SCIENTIFIC GRAPH CARD INVENTORY                           ")
print("==========================================================================================")

for sheet_audit in audit_result.sheets:
    sheet_name = sheet_audit.sheet_name
    exps = grouper.group_sheet_blocks_into_experiments(sheet_audit)
    print(f"\nSHEET: {sheet_name}")
    print(f"  Source Blocks Count: {len(sheet_audit.blocks)}")
    print(f"  Expected Graph Cards Count: {len(exps)}")
    
    for exp in exps:
        df = exp.combined_dataframe
        traces_count = len(df["Trace_ID"].unique()) if "Trace_ID" in df.columns else len(exp.blocks)
        params = [c for c in df.columns if c not in ["Voltage", "Wavelength", "Current", "QE", "Responsivity", "Detectivity", "Absorption", "Photon_Energy", "Trace_ID", "Block_ID", "Run"]]
        
        print(f"    - Experiment ID: '{exp.experiment_id}'")
        print(f"      Target: '{exp.target_name}' ({exp.target_unit}), Axis: '{exp.axis_name}' ({exp.axis_unit})")
        print(f"      Available Traces: {traces_count}")
        print(f"      Parameter Names: {params}")
        print(f"      Source Blocks: {len(exp.blocks)}")
