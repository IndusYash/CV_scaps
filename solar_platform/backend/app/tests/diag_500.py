"""
Diagnostic script: captures exact exceptions from /api/curves for every sheet.
Run from:  solar_platform/backend/
"""
from __future__ import annotations
import sys
import io
import traceback
import asyncio
from io import BytesIO
from pathlib import Path
from fastapi import UploadFile

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.api.endpoints import upload_file, get_curves
from app.models.schemas import TrainRequest

for name in ["Ca3PI3.xlsx", "Ca3PI3 copy.xlsx", "Ca3AsI3.xlsx"]:
    wb_path = backend_dir.parent / name
    if wb_path.exists():
        break

print(f"Workbook: {wb_path}")

with open(wb_path, "rb") as f:
    contents = f.read()

upload_file_obj = UploadFile(filename=wb_path.name, file=BytesIO(contents))
upload_res = asyncio.run(upload_file(upload_file_obj))
file_id = upload_res.file_id
print(f"file_id: {file_id}\n")

for s in upload_res.sheets:
    sheet_name = s.sheet_name
    print(f"=== Sheet: {sheet_name!r} ===")
    try:
        req = TrainRequest(file_id=file_id, sheet_name=sheet_name)
        res = asyncio.run(get_curves(req))
        print(f"  OK => {len(res.experiments)} experiments")
        for exp in res.experiments:
            print(f"    [{exp.experiment_id}] traces={len(exp.traces)}")
            if exp.traces:
                t0 = exp.traces[0]
                print(f"      first_trace label={t0.label!r} x={len(t0.x)} y={len(t0.y)}")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    print()
