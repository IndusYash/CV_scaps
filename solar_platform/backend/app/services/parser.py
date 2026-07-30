from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
import io
import pandas as pd


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Removes Unnamed columns, strips header spaces, drops completely empty rows."""
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    df.columns = df.columns.astype(str).str.strip()
    return df.dropna(how="all").reset_index(drop=True)


def detect_sheet_schema(df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
    """Inspects a single sheet DataFrame and auto-detects parameter sweep patterns,

    X-axis columns, and curve types (JV, QE, Responsivity, Detectivity).
    """
    df_clean = clean_dataframe(df)
    columns = [str(c) for c in df_clean.columns]

    # Check for Single-Curve / Optimized Mode
    col_str_lower = " ".join([c.lower() for c in columns])
    if "jtot" in col_str_lower or "d*" in col_str_lower or "detectivity" in col_str_lower:
        curve_types = []
        if any("v" in c.lower() for c in columns):
            curve_types.append("JV")
        if any("qe" in c.lower() for c in columns):
            curve_types.append("QE")
        if any("r[" in c.lower() or "responsivity" in c.lower() for c in columns):
            curve_types.append("Responsivity")
        if any("d*" in c.lower() for c in columns):
            curve_types.append("Detectivity")

        return {
            "sheet_name": sheet_name,
            "row_count": len(df_clean),
            "col_count": len(columns),
            "columns": columns,
            "detected_parameter_name": "Single Run (Optimized Device)",
            "detected_parameter_unit": "",
            "detected_x_axis": "Voltage / Wavelength",
            "curve_types_found": curve_types,
            "is_single_curve_mode": True,
        }

    # Detect parametric sweep patterns (t=, T=, 10^N)
    param_name = "Parameter"
    param_unit = ""
    curve_types = ["JV"]

    t_count = sum(1 for c in columns if str(c).startswith("t="))
    temp_count = sum(1 for c in columns if str(c).startswith("T="))
    exp_count = sum(1 for c in columns if str(c).startswith("10^"))

    if t_count > 0:
        param_name = f"{sheet_name} Thickness" if sheet_name in ["PVK", "ETL", "HTL"] else "Thickness"
        param_unit = "nm"
    elif temp_count > 0:
        param_name = "Temperature"
        param_unit = "K"
    elif exp_count > 0:
        if "defect" in sheet_name.lower():
            param_name = "Defect Density"
            param_unit = "cm^-3"
        elif "donor" in sheet_name.lower():
            param_name = "Donor Concentration"
            param_unit = "cm^-3"
        elif "interface" in sheet_name.lower():
            param_name = "Interface Density"
            param_unit = "cm^-2"
        else:
            param_name = "Density / Concentration"
            param_unit = "cm^-3"

    # Check for QE (.1 suffix) and Responsivity (.2 suffix)
    if any(".1" in c for c in columns):
        curve_types.append("QE")
    if any(".2" in c for c in columns):
        curve_types.append("Responsivity")

    x_axis = "Voltage / Wavelength"
    for c in columns:
        if c.lower() in ["v(v)", "voltage", "v"]:
            x_axis = "Voltage (V)"
            break
        elif "lambda" in c.lower() or "wavelength" in c.lower():
            x_axis = "Wavelength (nm)"
            break

    return {
        "sheet_name": sheet_name,
        "row_count": len(df_clean),
        "col_count": len(columns),
        "columns": columns,
        "detected_parameter_name": param_name,
        "detected_parameter_unit": param_unit,
        "detected_x_axis": x_axis,
        "curve_types_found": curve_types,
        "is_single_curve_mode": False,
    }


def parse_excel_workbook(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """Loads all sheets from an Excel binary file into a dictionary of DataFrames."""
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    return {sheet: excel_file.parse(sheet) for sheet in excel_file.sheet_names}
