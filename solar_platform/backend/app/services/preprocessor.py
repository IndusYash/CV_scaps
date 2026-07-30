from __future__ import annotations

import re
from typing import Any, Dict, Tuple
import pandas as pd
from app.services.parser import clean_dataframe


def preprocess_parametric_sheet(
    df: pd.DataFrame,
    sheet_name: str,
) -> Dict[str, pd.DataFrame]:
    """Preprocesses a parametric sweep sheet (e.g. PVK, HTL, ETL, Temp, Defect, Donor, Interface).

    Returns a dict with key long DataFrames: 'jv', 'qe', 'responsivity'.
    """
    df = clean_dataframe(df)
    columns = [str(c) for c in df.columns]

    # Detect parameter prefix rule (t=, T=, 10^)
    prefix = ""
    extract_regex = r"(\d+(?:\.\d+)?)"

    if any(c.startswith("t=") for c in columns):
        prefix = "t="
        extract_regex = r"t=(\d+)"
    elif any(c.startswith("T=") for c in columns):
        prefix = "T="
        extract_regex = r"T=(\d+)"
    elif any(c.startswith("10^") for c in columns):
        prefix = "10^"
        extract_regex = r"10\^(\d+)"

    param_col_name = "Parameter"
    if "thickness" in sheet_name.lower() or sheet_name in ["PVK", "ETL", "HTL"]:
        param_col_name = "Thickness"
    elif "temp" in sheet_name.lower():
        param_col_name = "Temperature"
    elif "defect" in sheet_name.lower():
        param_col_name = "Defect_Density"
    elif "donor" in sheet_name.lower():
        param_col_name = "Donor_Concentration"
    elif "interface" in sheet_name.lower():
        param_col_name = "Interface_Density"

    result = {}

    # 1. J-V Data Processing
    jv_x_col = next((c for c in columns if c.lower() in ["v(v)", "voltage", "v"]), None)
    if jv_x_col and jv_x_col in columns:
        jv_cols = [jv_x_col] + [
            c for c in columns
            if (c.startswith(prefix) or prefix == "") and ".1" not in c and ".2" not in c and c != jv_x_col
        ]
        if len(jv_cols) > 1:
            jv_df = df[jv_cols].copy()
            jv_long = jv_df.melt(id_vars=jv_x_col, var_name=param_col_name, value_name="Current")
            jv_long[param_col_name] = jv_long[param_col_name].astype(str).str.extract(extract_regex).astype(float)
            jv_long = jv_long.dropna().rename(columns={jv_x_col: "Voltage"}).reset_index(drop=True)
            jv_long["Voltage"] = pd.to_numeric(jv_long["Voltage"])
            jv_long["Current"] = pd.to_numeric(jv_long["Current"])
            result["jv"] = jv_long

    # 2. QE Data Processing
    qe_x_col = next((c for c in columns if "lambda" in c.lower() or "wavelength" in c.lower()), None)
    if qe_x_col and qe_x_col in columns:
        qe_cols = [qe_x_col] + [
            c for c in columns
            if (c.startswith(prefix) or prefix == "") and ".1" in c and c != qe_x_col
        ]
        if len(qe_cols) > 1:
            qe_df = df[qe_cols].copy()
            qe_long = qe_df.melt(id_vars=qe_x_col, var_name=param_col_name, value_name="QE")
            qe_long[param_col_name] = qe_long[param_col_name].astype(str).str.extract(extract_regex).astype(float)
            qe_long = qe_long.dropna().rename(columns={qe_x_col: "Wavelength"}).reset_index(drop=True)
            qe_long["Wavelength"] = pd.to_numeric(qe_long["Wavelength"])
            qe_long["QE"] = pd.to_numeric(qe_long["QE"])
            result["qe"] = qe_long

    # 3. Responsivity Data Processing
    r_x_col = next((c for c in columns if ".1" in c and ("lambda" in c.lower() or "wavelength" in c.lower())), None)
    if r_x_col and r_x_col in columns:
        r_cols = [r_x_col] + [
            c for c in columns
            if (c.startswith(prefix) or prefix == "") and ".2" in c and c != r_x_col
        ]
        if len(r_cols) > 1:
            r_df = df[r_cols].copy()
            r_long = r_df.melt(id_vars=r_x_col, var_name=param_col_name, value_name="Responsivity")
            r_long[param_col_name] = r_long[param_col_name].astype(str).str.extract(extract_regex).astype(float)
            r_long = r_long.dropna().rename(columns={r_x_col: "Wavelength"}).reset_index(drop=True)
            r_long["Wavelength"] = pd.to_numeric(r_long["Wavelength"])
            r_long["Responsivity"] = pd.to_numeric(r_long["Responsivity"])
            result["responsivity"] = r_long

    return result


def preprocess_single_curve_sheet(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Preprocesses a single-curve / Optimized sheet (e.g. Optimized device).

    Returns dataframes for JV, QE, Responsivity, and Detectivity.
    """
    df = clean_dataframe(df)
    columns = [str(c) for c in df.columns]

    result = {}

    # JV
    v_col = next((c for c in columns if "v(" in c.lower() or c.lower() == "voltage"), None)
    j_col = next((c for c in columns if "jtot" in c.lower() or "current" in c.lower()), None)
    if v_col and j_col:
        jv_df = df[[v_col, j_col]].copy().dropna()
        jv_df.columns = ["Voltage", "Current"]
        jv_df["Voltage"] = pd.to_numeric(jv_df["Voltage"])
        jv_df["Current"] = pd.to_numeric(jv_df["Current"])
        result["jv"] = jv_df

    # QE
    l_col = next((c for c in columns if "lambda" in c.lower() and ".1" not in c), None)
    qe_col = next((c for c in columns if "qe" in c.lower()), None)
    if l_col and qe_col:
        qe_df = df[[l_col, qe_col]].copy().dropna()
        qe_df.columns = ["Wavelength", "QE"]
        qe_df["Wavelength"] = pd.to_numeric(qe_df["Wavelength"])
        qe_df["QE"] = pd.to_numeric(qe_df["QE"])
        result["qe"] = qe_df

    # Responsivity
    l_col_r = next((c for c in columns if "lambda" in c.lower() and ".1" in c), None)
    r_col = next((c for c in columns if "r[" in c.lower() or "responsivity" in c.lower()), None)
    if l_col_r and r_col:
        r_df = df[[l_col_r, r_col]].copy().dropna()
        r_df.columns = ["Wavelength", "Responsivity"]
        r_df["Wavelength"] = pd.to_numeric(r_df["Wavelength"])
        r_df["Responsivity"] = pd.to_numeric(r_df["Responsivity"])
        result["responsivity"] = r_df

    # Detectivity (D*)
    vd_col = next((c for c in columns if "v(" in c.lower() and c != v_col), v_col)
    d_col = next((c for c in columns if "d*" in c.lower() or "detectivity" in c.lower()), None)
    if vd_col and d_col:
        d_df = df[[vd_col, d_col]].copy().dropna()
        d_df.columns = ["Voltage", "Detectivity"]
        d_df["Voltage"] = pd.to_numeric(d_df["Voltage"])
        d_df["Detectivity"] = pd.to_numeric(d_df["Detectivity"])
        result["detectivity"] = d_df

    return result
