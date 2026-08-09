"""
Generate Evidence Report for representative regions.
"""
from __future__ import annotations
import sys
import io
from pathlib import Path

from app.parser_v2.reader.raw_reader import read_workbook
from app.parser_v2.scanner.region_scanner import scan_sheet
from app.parser_v2.probes import (
    axis_probe,
    header_probe,
    numeric_probe,
    parameter_probe,
    repetition_probe,
    label_probe,
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

file_bytes = open(r'C:\Users\yashv\Desktop\REPOS\Solar_Project\Data_file.xlsx', 'rb').read()
raw_wb = read_workbook('Data_file.xlsx', file_bytes)

targets = [
    ('ETL', 0, '1. ETL J-V region'),
    ('ETL', 2, '2. ETL QE region'),
    ('Defect Density', 0, '3. Defect Density J-V region'),
    ('Rs', 0, '4. Rs J-V region'),
    ('Optimized', 0, '5. Optimized J-V region'),
    ('Sheet1', 0, '6. Sheet1 region'),
]

for sheet_name, reg_idx, desc in targets:
    sheet = raw_wb.get_sheet(sheet_name)
    regions = scan_sheet(sheet)
    if reg_idx >= len(regions):
        reg = max(regions, key=lambda r: r.total_cells)
    else:
        reg = regions[reg_idx]

    print('=' * 80)
    print(f"{desc} (Sheet: {sheet_name!r}, Region: {reg.region_id}, Bounds: row {reg.row_start}..{reg.row_end}, col {reg.col_start}..{reg.col_end})")
    print('=' * 80)

    items = []
    items.extend(axis_probe.probe_axis(reg, sheet))
    items.extend(header_probe.probe_headers(reg, sheet))
    items.extend(numeric_probe.probe_numerics(reg, sheet))
    items.extend(parameter_probe.probe_parameters(reg, sheet))
    items.extend(repetition_probe.probe_repetition(reg, sheet))
    items.extend(label_probe.probe_labels(reg, sheet))

    for it in items:
        cell_count = len(it.source_cells)
        print(f"  • [{it.probe_id}] {it.name:<32} | Polarity: {it.polarity.value:<8} | Conf: {it.confidence:.2f} | Sources: {cell_count} cells")
        print(f"    Rationale: {it.rationale}")
        if isinstance(it.value, dict):
            if 'patterns' in it.value:
                print(f"    Patterns: {it.value['patterns']}")
            if 'distinct_concepts' in it.value:
                print(f"    Concepts: {it.value['distinct_concepts']}")
            if 'range' in it.value:
                print(f"    Range: {it.value['range']}")
    print()
