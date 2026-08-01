import sys
import asyncio
from pathlib import Path
from io import BytesIO
from fastapi import UploadFile

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.api.endpoints import upload_file

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

with open(wb_path, "rb") as f:
    contents = f.read()

upload_file_obj = UploadFile(filename=wb_path.name, file=BytesIO(contents))
upload_res = asyncio.run(upload_file(upload_file_obj))

sheet_illum = next(s for s in upload_res.sheets if s.sheet_name == "Illumination")
exp = sheet_illum.experiments[0]
print(f"Exp ID: '{exp.experiment_id}', trace_count: {exp.trace_count}, params: {exp.parameter_names}")
