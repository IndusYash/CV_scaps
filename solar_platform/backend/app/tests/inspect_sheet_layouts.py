from __future__ import annotations

import os
import sys
import pandas as pd

sys.path.insert(0, ".")

from app.readers.excel_reader import ExcelInputReader
from app.layout.segmenter import TableSegmenter

reader = ExcelInputReader()

ca3_path = "../Ca3PI3.xlsx"
if not os.path.exists(ca3_path):
    ca3_path = "solar_platform/Ca3PI3.xlsx"

if os.path.exists(ca3_path):
    with open(ca3_path, "rb") as f:
        doc = reader.read("Ca3PI3.xlsx", f.read())

    segmenter = TableSegmenter(max_adjacency_distance=2)

    for sheet in doc.sheets:
        if sheet.sheet_name in ["QE (T+DD)", "QE (IDD)", "T (K)", "Rs", "Rsh", "absorption"]:
            blocks = segmenter.segment_sheet(sheet)
            print(f"\nSheet '{sheet.sheet_name}': {sheet.row_count} rows x {sheet.col_count} cols -> Discovered {len(blocks)} blocks")
            for b in blocks[:5]:
                print(f"   Block {b.block_id}: ({b.bounding_box.start_row},{b.bounding_box.start_col}) -> ({b.bounding_box.end_row},{b.bounding_box.end_col})")
