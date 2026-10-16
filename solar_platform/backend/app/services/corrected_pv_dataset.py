"""
app.services.corrected_pv_dataset
---------------------------------
Builds the training table for the global (Layer B) Random Forest from the
*corrected* parsing pipeline's summary sweeps (PCE / Voc / Jsc / FF versus a
swept device parameter), instead of from the J-V-derived rows in the SQLite
database.

Why this exists
    The legacy database path feeds Layer B only a handful of rows derived from
    J-V curves, with almost every sweep feature equal to 0. The summary sweeps
    (NA, Rs, Rsh, Temperature, T+DD, ...) hold the real parameter -> performance
    data, and the corrected pipeline reconstructs them with the right axes and
    parameter names.

Design constraints
    * Purely additive. Nothing here changes the existing database tables or the
      existing builder (rf_pv_regressor.build_ml_dataset_from_db). It is only
      used when the training endpoint is called with data_source="corrected".
    * Output has the exact same columns Layer B already expects (workbook_name,
      material, group_id, j_sc, v_oc, ff, pce + FEATURE_KEYS) so the existing
      MLRegressorManager and validation modes work unchanged.
    * Conservative: a sweep whose parameter cannot be mapped to an existing
      feature key is excluded and reported in the audit log, never guessed.

Conventions (matching the existing builder)
    * Unspecified / not-swept features are 0.0.
    * FF is stored as a fraction (0.71), PCE in percent, Jsc in mA/cm^2.
    * Each row of a summary sweep is one independent simulation run, so each
      row gets its own group_id (experiment-level split == run-level split).
"""
from __future__ import annotations

import datetime
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from app.database import db_manager
from app.services.experiment_grouper import ExperimentGrouperEngine

logger = logging.getLogger(__name__)

# Feature keys are owned by rf_pv_regressor; imported lazily to avoid any
# chance of an import cycle and to guarantee we always match its definition.
def _feature_keys() -> List[str]:
    from app.services.rf_pv_regressor import FEATURE_KEYS
    return list(FEATURE_KEYS)


# Target (metric) name emitted by the pipeline -> Layer B column.
_METRIC_COLUMNS = {"Jsc": "j_sc", "Voc": "v_oc", "FF": "ff", "PCE": "pce"}

# Normalised parameter name (lowercase, alphanumerics only) -> Layer B feature.
# Only unambiguous physical parameters are mapped. Anything else (for example
# the interface-defect sweeps in IDD, which have no matching feature key) is
# excluded and reported rather than forced into a wrong feature.
_PARAMETER_TO_FEATURE = {
    "thickness": "thickness",
    "defectdensity": "defect_density",
    "temperature": "temperature",
    "seriesresistance": "series_resistance",
    "shuntresistance": "shunt_resistance",
    "acceptordensity": "doping_density",
    "donordensity": "doping_density",
}

_NON_PARAM_COLUMNS = {"Trace_ID", "Block_ID"}
_CURVE_AXES = {"Wavelength", "Voltage"}
_MIN_ILLUMINATED_JSC = 0.1  # same dark-current guard as the existing builder

# Physical sanity limits for a single-junction solar cell simulation. Anything
# outside these is a parsing artefact (for example a swept-parameter column read
# as a result), never a real device, and one such row is enough to wreck a
# Random Forest (a single 1e22 target ruins every prediction near it).
# Rows outside the limits are excluded and reported, never trained on.
_PLAUSIBLE_LIMITS = {
    "j_sc": (_MIN_ILLUMINATED_JSC, 100.0),  # mA/cm^2
    "v_oc": (0.0, 3.0),                     # V
    "ff":   (0.0, 1.0),                     # fraction
    "pce":  (0.0, 60.0),                    # %
}


def implausible_reason(values: Dict[str, Any]) -> str:
    """Returns a reason string if any metric is outside its physical range, else ''."""
    for key, (lo, hi) in _PLAUSIBLE_LIMITS.items():
        v = values.get(key)
        if v is None or pd.isna(v) or not (lo < float(v) <= hi):
            return f"Implausible {key}={v:.4g} (physical range {lo:g} to {hi:g})" if v is not None and not pd.isna(v) else f"Missing {key}"
    return ""

_TABLE = "corrected_pv_rows"


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def material_from_workbook(workbook_name: str) -> str:
    """Ca3AsBr3.xlsx -> 'Ca3asbr3' (same capitalisation style as the existing builder)."""
    stem = Path(workbook_name).stem.lower()
    stem = re.sub(r"\(\d+\)|\bcopy\b", "", stem)
    return re.sub(r"[^a-z0-9]", "", stem).capitalize() or "Solardevice"


def extract_rows(filename: str, audit_result: Any) -> List[Dict[str, Any]]:
    """Turns the corrected pipeline's summary-sweep experiments into row dicts.

    Every candidate row is returned with status ELIGIBLE or EXCLUDED (plus a
    reason) so the audit trail is complete.
    """
    feature_keys = _feature_keys()
    material = material_from_workbook(filename)
    grouper = ExperimentGrouperEngine()
    out: List[Dict[str, Any]] = []

    for sheet_audit in audit_result.sheets:
        sheet_name = sheet_audit.sheet_name
        experiments = grouper.group_sheet_blocks_into_experiments(sheet_audit)

        # (sheet, block, parameter columns) -> {metric column: (frame, unit)}
        sweeps: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Tuple[pd.DataFrame, str]]] = defaultdict(dict)

        for exp in experiments:
            metric_col = _METRIC_COLUMNS.get(exp.target_name)
            df = exp.combined_dataframe
            if metric_col is None or df is None or df.empty:
                continue
            if exp.axis_name in _CURVE_AXES:
                continue  # curve data, not a summary sweep

            param_cols = tuple(
                c for c in df.columns if c not in _NON_PARAM_COLUMNS and c != exp.target_name
            )
            if not param_cols or exp.target_name not in df.columns:
                continue

            unit = str(getattr(exp, "target_unit", "") or "")
            for block_id, block_df in df.groupby("Block_ID", sort=False) if "Block_ID" in df.columns else [("block", df)]:
                sweeps[(str(block_id), param_cols)][metric_col] = (
                    block_df[list(param_cols) + [exp.target_name]].rename(columns={exp.target_name: metric_col}),
                    unit,
                )

        for (block_id, param_cols), metrics in sweeps.items():
            # Need all four targets to form a complete training row.
            if set(metrics) != set(_METRIC_COLUMNS.values()):
                missing = sorted(set(_METRIC_COLUMNS.values()) - set(metrics))
                out.append(_excluded(filename, material, sheet_name, block_id, 0,
                                     f"Incomplete metric set (missing {', '.join(missing)})"))
                continue

            unmapped = [c for c in param_cols if _norm(c) not in _PARAMETER_TO_FEATURE]
            merged = None
            for col, (frame, _unit) in metrics.items():
                frame = frame.groupby(list(param_cols), as_index=False, sort=False)[col].first()
                merged = frame if merged is None else merged.merge(frame, on=list(param_cols), how="inner")
            if merged is None or merged.empty:
                continue

            ff_unit = metrics["ff"][1]
            ff_is_percent = ff_unit == "%" or (not ff_unit and float(merged["ff"].max()) > 1.5)

            for i, rec in enumerate(merged.to_dict("records")):
                if unmapped:
                    out.append(_excluded(
                        filename, material, sheet_name, block_id, i,
                        f"Sweep parameter '{unmapped[0]}' has no matching Layer B feature key",
                    ))
                    continue

                vals = {m: rec[m] for m in _METRIC_COLUMNS.values()}
                if any(pd.isna(v) for v in vals.values()) or any(pd.isna(rec[c]) for c in param_cols):
                    out.append(_excluded(filename, material, sheet_name, block_id, i,
                                         "Missing value in metrics or sweep parameter"))
                    continue
                if vals["j_sc"] < _MIN_ILLUMINATED_JSC:
                    out.append(_excluded(filename, material, sheet_name, block_id, i,
                                         "Non-illuminated / near-zero Jsc row"))
                    continue
                ff_val = float(vals["ff"]) / 100.0 if ff_is_percent else float(vals["ff"])
                bad = implausible_reason({"j_sc": vals["j_sc"], "v_oc": vals["v_oc"], "ff": ff_val, "pce": vals["pce"]})
                if bad:
                    out.append(_excluded(filename, material, sheet_name, block_id, i, bad))
                    continue

                row = {
                    "workbook_name": filename,
                    "sheet_name": sheet_name,
                    "block_id": block_id,
                    "row_index": i,
                    "dataset_id": f"{filename}_{sheet_name}_{block_id}",
                    "trace_id": f"{filename}_{sheet_name}_{block_id}_row{i}",
                    "material": material,
                    "group_id": f"grp_{material}_{sheet_name}_{block_id}_row{i}",
                    "status": "ELIGIBLE",
                    "reason": "Summary sweep row extracted from corrected pipeline",
                    "j_sc": float(vals["j_sc"]),
                    "v_oc": float(vals["v_oc"]),
                    "ff": float(vals["ff"]) / 100.0 if ff_is_percent else float(vals["ff"]),
                    "pce": float(vals["pce"]),
                }
                for key in feature_keys:
                    row[key] = 0.0
                for c in param_cols:
                    row[_PARAMETER_TO_FEATURE[_norm(c)]] = float(rec[c])
                out.append(row)

    return out


def _excluded(filename: str, material: str, sheet: str, block: str, idx: int, reason: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "workbook_name": filename, "sheet_name": sheet, "block_id": block, "row_index": idx,
        "dataset_id": f"{filename}_{sheet}_{block}", "trace_id": f"{filename}_{sheet}_{block}_row{idx}",
        "material": material, "group_id": f"grp_{material}_{sheet}_{block}_row{idx}",
        "status": "EXCLUDED", "reason": reason,
        "j_sc": None, "v_oc": None, "ff": None, "pce": None,
    }
    for key in _feature_keys():
        row[key] = 0.0
    return row


def _ensure_table(conn) -> None:
    feature_cols = ", ".join(f"{k} REAL" for k in _feature_keys())
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            workbook_name TEXT NOT NULL,
            sheet_name TEXT NOT NULL,
            block_id TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            dataset_id TEXT, trace_id TEXT, material TEXT, group_id TEXT,
            status TEXT, reason TEXT,
            j_sc REAL, v_oc REAL, ff REAL, pce REAL,
            {feature_cols},
            created_at TEXT,
            PRIMARY KEY (workbook_name, sheet_name, block_id, row_index)
        )
    """)


def persist_corrected_rows(filename: str, audit_result: Any) -> int:
    """Extracts and stores the corrected rows for one workbook.

    Re-uploading the same workbook replaces its rows (no duplicates, no stale
    leftovers). Returns the number of stored rows. Writes only to the new
    corrected_pv_rows table.
    """
    rows = extract_rows(filename, audit_result)
    feature_keys = _feature_keys()
    cols = ["workbook_name", "sheet_name", "block_id", "row_index", "dataset_id", "trace_id",
            "material", "group_id", "status", "reason", "j_sc", "v_oc", "ff", "pce",
            *feature_keys, "created_at"]
    now = datetime.datetime.utcnow().isoformat() + "Z"

    conn = db_manager.get_db_connection()
    try:
        _ensure_table(conn)
        conn.execute(f"DELETE FROM {_TABLE} WHERE workbook_name = ?", (filename,))
        conn.executemany(
            f"INSERT OR REPLACE INTO {_TABLE} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [tuple({**r, 'created_at': now}[c] for c in cols) for r in rows],
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def build_ml_dataset_from_corrected() -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Drop-in alternative to rf_pv_regressor.build_ml_dataset_from_db().

    Returns the same (dataframe, audit_log) shape. The dataframe holds only
    ELIGIBLE rows; the audit log lists every row, eligible or excluded.
    """
    conn = db_manager.get_db_connection()
    try:
        _ensure_table(conn)
        stored = pd.read_sql_query(f"SELECT * FROM {_TABLE} ORDER BY workbook_name, sheet_name, block_id, row_index", conn)
    finally:
        conn.close()

    # Re-check stored rows against the physical limits so rows saved before the
    # check existed are filtered without needing a re-upload.
    for idx, r in stored[stored["status"] == "ELIGIBLE"].iterrows():
        bad = implausible_reason({k: r[k] for k in ("j_sc", "v_oc", "ff", "pce")})
        if bad:
            stored.at[idx, "status"] = "EXCLUDED"
            stored.at[idx, "reason"] = bad

    audit_log = [
        {"dataset_id": r["dataset_id"], "trace_id": r["trace_id"], "status": r["status"], "reason": r["reason"]}
        for r in stored.to_dict("records")
    ]
    eligible = stored[stored["status"] == "ELIGIBLE"]
    keep = ["dataset_id", "trace_id", "workbook_name", "material", "group_id",
            "j_sc", "v_oc", "ff", "pce", *_feature_keys()]
    return eligible[keep].reset_index(drop=True), audit_log


def list_stored_workbooks() -> List[Dict[str, Any]]:
    """One entry per workbook currently stored in the corrected table, with the
    number of rows usable for training and the number excluded (with reasons in
    the audit log). Read-only."""
    conn = db_manager.get_db_connection()
    try:
        _ensure_table(conn)
        rows = conn.execute(
            f"""SELECT workbook_name,
                       SUM(CASE WHEN status = 'ELIGIBLE' THEN 1 ELSE 0 END) AS eligible_rows,
                       SUM(CASE WHEN status = 'EXCLUDED' THEN 1 ELSE 0 END) AS excluded_rows
                FROM {_TABLE} GROUP BY workbook_name ORDER BY workbook_name"""
        ).fetchall()
    finally:
        conn.close()
    return [
        {"workbook_name": r[0], "eligible_rows": int(r[1] or 0), "excluded_rows": int(r[2] or 0)}
        for r in rows
    ]


def delete_stored_workbooks(names: Any = None) -> int:
    """Removes rows from the corrected table only (never any other table).

    names=None removes every workbook; otherwise only the given workbook names.
    Returns the number of rows removed. Re-uploading a workbook refills it.
    """
    conn = db_manager.get_db_connection()
    try:
        _ensure_table(conn)
        if names is None:
            cur = conn.execute(f"DELETE FROM {_TABLE}")
        else:
            cur = conn.executemany(f"DELETE FROM {_TABLE} WHERE workbook_name = ?", [(n,) for n in names])
        conn.commit()
        return cur.rowcount if cur.rowcount is not None else 0
    finally:
        conn.close()