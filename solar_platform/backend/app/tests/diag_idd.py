from __future__ import annotations
import sys, io, asyncio
from io import BytesIO
from pathlib import Path
from fastapi import UploadFile
import pandas as pd

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.api.endpoints import upload_file
from app.services.ml_engine import get_parsed_experiments

for name in ["Ca3PI3.xlsx", "Ca3PI3 copy.xlsx"]:
    wb_path = backend_dir.parent / name
    if wb_path.exists():
        break

with open(wb_path, "rb") as f:
    contents = f.read()

upload_file_obj = UploadFile(filename=wb_path.name, file=BytesIO(contents))
upload_res = asyncio.run(upload_file(upload_file_obj))
file_id = upload_res.file_id

for sheet_name in ["IDD", "T+DD"]:
    print(f"\n=== {sheet_name} ===")
    exps = get_parsed_experiments(file_id, sheet_name)
    for exp in exps[:3]:
        df = exp.combined_dataframe
        print(f"  exp_id={exp.experiment_id!r}")
        print(f"  axis_name={exp.axis_name!r}  target_name={exp.target_name!r}")
        print(f"  df.columns={list(df.columns)}")
        print(f"  df shape={df.shape}")
        print(f"  df head:")
        print(df.head(3).to_string())
        print()
