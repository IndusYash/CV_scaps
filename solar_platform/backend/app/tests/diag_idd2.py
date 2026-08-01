from __future__ import annotations
import sys, io, asyncio
from io import BytesIO
from pathlib import Path
from fastapi import UploadFile

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.api.endpoints import upload_file
from app.services.ml_engine import get_parsed_experiments
from app.pipeline import UnifiedParserPipeline

for name in ["Ca3PI3.xlsx", "Ca3PI3 copy.xlsx"]:
    wb_path = backend_dir.parent / name
    if wb_path.exists():
        break

with open(wb_path, "rb") as f:
    contents = f.read()

pipeline = UnifiedParserPipeline()
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "debug_idd")
sheet = next(s for s in audit_result.sheets if s.sheet_name == "IDD")
print(f"IDD: {len(sheet.blocks)} blocks")
for b in sheet.blocks[:5]:
    ir = b.intermediate_representation
    if ir is None:
        print(f"  block {b.block_id}: NO IR")
        continue
    print(f"  block {b.block_id}: category={b.category}")
    print(f"    axes: {[(a.name, a.unit, len(a.values)) for a in ir.axes]}")
    print(f"    targets: {[(t.name, t.unit, t.data_shape, t.column_labels) for t in ir.targets]}")
    print(f"    params: {[(p.name, p.values) for p in ir.parameters]}")
    if ir.targets:
        t = ir.targets[0]
        print(f"    first target dense_matrix rows: {len(t.dense_matrix) if t.dense_matrix else 0}")
