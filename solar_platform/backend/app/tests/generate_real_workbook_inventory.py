import sys
import json
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

artifact_dir = Path(r"C:\Users\yashv\.gemini\antigravity-ide\brain\a115f4dd-7d59-4d32-b143-068974c2dfa0")

inventory_lines = [
    "# Real Workbook Experiment Inventory — Post Phase 4",
    "",
    f"**Workbook:** `{wb_path.name}`",
    f"**Total Sheets Audited:** {len(audit_result.sheets)}",
    "",
    "---",
    "",
]

expectation_manifest = {}

for sheet_audit in audit_result.sheets:
    sheet_name = sheet_audit.sheet_name
    exps = grouper.group_sheet_blocks_into_experiments(sheet_audit)
    
    inventory_lines.append(f"## Sheet: `{sheet_name}`")
    inventory_lines.append(f"- **Segmented Blocks:** {len(sheet_audit.blocks)}")
    inventory_lines.append(f"- **Scientific Experiments:** {len(exps)}")
    inventory_lines.append("")
    
    sheet_manifest = []
    
    for e in exps:
        df = e.combined_dataframe
        n_rows = len(df)
        trace_ids = df["Trace_ID"].unique().tolist() if "Trace_ID" in df.columns else []
        n_traces = len(trace_ids) if trace_ids else (len(df.groupby(e.swept_parameter_names)) if e.swept_parameter_names else 1)
        
        inventory_lines.append(f"### Experiment: `{e.experiment_id}`")
        inventory_lines.append(f"- **Target Name:** `{e.target_name}` ({e.target_unit})")
        inventory_lines.append(f"- **Axis Name:** `{e.axis_name}` ({e.axis_unit})")
        inventory_lines.append(f"- **Swept Parameters:** {e.swept_parameter_names or 'None'}")
        inventory_lines.append(f"- **Total Rows:** {n_rows}")
        inventory_lines.append(f"- **Trace Count:** {n_traces}")
        inventory_lines.append(f"- **Sample Trace IDs:** {trace_ids[:5] if trace_ids else 'Standard'}")
        inventory_lines.append("")
        
        sheet_manifest.append({
            "experiment_id": e.experiment_id,
            "target_name": e.target_name,
            "target_unit": e.target_unit,
            "axis_name": e.axis_name,
            "axis_unit": e.axis_unit,
            "swept_parameters": e.swept_parameter_names,
            "total_rows": n_rows,
            "trace_count": n_traces,
        })
        
    expectation_manifest[sheet_name] = sheet_manifest
    inventory_lines.append("---")
    inventory_lines.append("")

# Write markdown inventory artifact
inv_path = artifact_dir / "real_workbook_experiment_inventory.md"
with open(inv_path, "w", encoding="utf-8") as f:
    f.write("\n".join(inventory_lines))

# Write json manifest artifact
man_path = artifact_dir / "graph_expectation_manifest.json"
with open(man_path, "w", encoding="utf-8") as f:
    json.dump(expectation_manifest, f, indent=2)

print(f"Inventory written to {inv_path}")
print(f"Manifest written to {man_path}")
