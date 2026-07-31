from __future__ import annotations

import os
import sys
import pandas as pd

sys.path.insert(0, ".")

from app.pipeline import UnifiedParserPipeline
from app.readers.excel_reader import ExcelInputReader
from app.services.ml_engine import (
    _SESSION_STORE,
    get_trained_session,
    predict_point,
    train_sheet_models,
)

reader = ExcelInputReader()

def find_path(fn):
    for p in [fn, os.path.join("..", fn), os.path.join("solar_platform", fn)]:
        if os.path.exists(p): return p
    return ""

pipeline = UnifiedParserPipeline()
wb_path = find_path("Data file.xlsx")

if wb_path:
    with open(wb_path, "rb") as f:
        contents = f.read()

    file_id = "file_opt_001"
    audit_result, parsed_data = pipeline.parse_workbook("Data file.xlsx", contents, file_id)
    opt_parsed = parsed_data.get("Optimized", {})

    print("=" * 110)
    print("           MULTI-BLOCK MODEL CACHE COLLISION INVESTIGATION REPORT")
    print("=" * 110)

    print(f"\n1. Discovered blocks in sheet 'Optimized' ({len(opt_parsed)} block keys in parsed_data):")
    for k in opt_parsed.keys():
        df = opt_parsed[k]
        print(f"   Key '{k}': shape {df.shape}, columns {list(df.columns)}")

    # Train all models for 'Optimized' sheet
    trained_sessions = train_sheet_models(
        file_id=file_id,
        sheet_name="Optimized",
        df_raw=pd.DataFrame(),
        parsed_sheet_data=opt_parsed,
    )

    print("\n2. Actual _SESSION_STORE Cache Keys Created:")
    for key_tuple in _SESSION_STORE.keys():
        if key_tuple[0] == file_id:
            sess = _SESSION_STORE[key_tuple]
            print(f"   Key Tuple {key_tuple} -> Session Block ID: '{sess.block_id}', Curve: '{sess.curve_type}'")

    print("\n3. Trace Prediction Resolution per JV Block:")
    for b_id in ["block_Optimized_1", "block_Optimized_2", "block_Optimized_4"]:
        sess = get_trained_session(file_id, "Optimized", block_id=b_id)
        if sess:
            pred_val = predict_point(file_id, "Optimized", "jv", block_id=b_id, x_val=0.0)
            m_key = list(sess.models.keys())[0]
            model = sess.models[m_key]
            print(f"   Requested block_id '{b_id}':")
            print(f"      - Resolved Session Block ID: {sess.block_id}")
            print(f"      - Model Feature Names: {getattr(model, 'feature_names_in_', 'N/A')}")
            print(f"      - Training Dataset Rows: {len(sess.processed_data[m_key])}")
            print(f"      - Predicted Point Value at V=0: {pred_val:.4f}")
