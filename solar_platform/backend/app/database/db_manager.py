"""
app.database.db_manager
-----------------------
Lightweight, standard sqlite3 persistent database layer for Canonical ScientificDataset objects.
Ensures session data survives server restarts.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.canonical.model import (
    CanonicalParameter,
    CanonicalScalarRecord,
    CanonicalTrace,
    CellRefDTO,
    ScientificDataset,
)

# Standard SQLite DB file path
DB_PATH = Path(__file__).resolve().parents[2] / "solar_platform.db"


def get_db_connection() -> sqlite3.Connection:
    """Establish and return a connection to the SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    """Create database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Table for ScientificDataset
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scientific_datasets (
        dataset_id TEXT PRIMARY KEY,
        workbook_name TEXT NOT NULL,
        sheet_name TEXT NOT NULL,
        region_id TEXT NOT NULL,
        parent_region_id TEXT,
        scientific_type TEXT NOT NULL,
        structural_source TEXT NOT NULL,
        axis_name TEXT,
        axis_unit TEXT,
        target_name TEXT,
        target_unit TEXT,
        validation_status TEXT NOT NULL,
        acceptance_state TEXT NOT NULL,
        is_ml_ready INTEGER NOT NULL,
        confidence_score REAL NOT NULL,
        ambiguity_gap REAL NOT NULL,
        global_parameters TEXT NOT NULL,   -- Serialized JSON list of parameters
        x_values TEXT NOT NULL,            -- Serialized JSON list of floats
        x_provenance TEXT NOT NULL,        -- Serialized JSON list of CellRefDTO
        traces TEXT NOT NULL,              -- Serialized JSON list of CanonicalTrace
        scalar_records TEXT NOT NULL,      -- Serialized JSON list of CanonicalScalarRecord
        validation_issues TEXT NOT NULL,   -- Serialized JSON list of warnings
        raw_dataset_reference TEXT         -- Serialized JSON dict of raw metadata
    );
    """)

    # 2. Table for derived JV parameters (PV metrics)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jv_derived_metrics (
        dataset_id TEXT NOT NULL,
        trace_id TEXT NOT NULL,
        j_sc REAL,
        v_oc REAL,
        p_max REAL,
        ff REAL,
        pce REAL,
        extracted_at TEXT NOT NULL,
        provenance_notes TEXT,
        PRIMARY KEY (dataset_id, trace_id),
        FOREIGN KEY (dataset_id) REFERENCES scientific_datasets(dataset_id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()


def insert_dataset(ds: ScientificDataset):
    """Insert a Canonical ScientificDataset object into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO scientific_datasets (
            dataset_id, workbook_name, sheet_name, region_id, parent_region_id,
            scientific_type, structural_source, axis_name, axis_unit, target_name, target_unit,
            validation_status, acceptance_state, is_ml_ready, confidence_score, ambiguity_gap,
            global_parameters, x_values, x_provenance, traces, scalar_records,
            validation_issues, raw_dataset_reference
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            ds.dataset_id,
            ds.workbook_name,
            ds.sheet_name,
            ds.region_id,
            ds.parent_region_id,
            ds.scientific_type,
            ds.structural_source,
            ds.axis_name,
            ds.axis_unit,
            ds.target_name,
            ds.target_unit,
            ds.validation_status,
            ds.acceptance_state,
            1 if ds.is_ml_ready else 0,
            ds.confidence_score,
            ds.ambiguity_gap,
            json.dumps([p.to_dict() for p in ds.global_parameters]),
            json.dumps(ds.x_values),
            json.dumps([p.to_dict() for p in ds.x_provenance]),
            json.dumps([t.to_dict() for t in ds.traces]),
            json.dumps([s.to_dict() for s in ds.scalar_records]),
            json.dumps(ds.validation_issues),
            json.dumps(ds.raw_dataset_reference) if ds.raw_dataset_reference else None,
        ),
    )

    conn.commit()
    conn.close()


def get_dataset(dataset_id: str) -> Optional[ScientificDataset]:
    """Retrieve a Canonical ScientificDataset object from the database by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM scientific_datasets WHERE dataset_id = ?;", (dataset_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    # Parse JSON fields back to objects
    global_params = [CanonicalParameter.from_dict(p) for p in json.loads(row["global_parameters"])]
    x_prov = [CellRefDTO.from_dict(p) for p in json.loads(row["x_provenance"])]
    traces = [CanonicalTrace.from_dict(t) for t in json.loads(row["traces"])]
    scalar_recs = [CanonicalScalarRecord.from_dict(s) for s in json.loads(row["scalar_records"])]

    return ScientificDataset(
        dataset_id=row["dataset_id"],
        workbook_name=row["workbook_name"],
        sheet_name=row["sheet_name"],
        region_id=row["region_id"],
        parent_region_id=row["parent_region_id"],
        scientific_type=row["scientific_type"],
        structural_source=row["structural_source"],
        axis_name=row["axis_name"],
        axis_unit=row["axis_unit"],
        target_name=row["target_name"],
        target_unit=row["target_unit"],
        global_parameters=global_params,
        x_values=json.loads(row["x_values"]),
        x_provenance=x_prov,
        traces=traces,
        scalar_records=scalar_recs,
        validation_status=row["validation_status"],
        acceptance_state=row["acceptance_state"],
        is_ml_ready=bool(row["is_ml_ready"]),
        confidence_score=row["confidence_score"],
        ambiguity_gap=row["ambiguity_gap"],
        validation_issues=json.loads(row["validation_issues"]),
        raw_dataset_reference=json.loads(row["raw_dataset_reference"]) if row["raw_dataset_reference"] else None,
    )


def list_datasets(scientific_type: Optional[str] = None) -> List[ScientificDataset]:
    """Retrieve list of datasets, optionally filtered by scientific_type."""
    conn = get_db_connection()
    cursor = conn.cursor()

    if scientific_type:
        cursor.execute("SELECT dataset_id FROM scientific_datasets WHERE scientific_type = ?;", (scientific_type.upper(),))
    else:
        cursor.execute("SELECT dataset_id FROM scientific_datasets;")

    rows = cursor.fetchall()
    conn.close()

    datasets = []
    for r in rows:
        ds = get_dataset(r["dataset_id"])
        if ds:
            datasets.append(ds)
    return datasets


def count_datasets() -> int:
    """Return total number of datasets in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM scientific_datasets;")
    row = cursor.fetchone()
    conn.close()
    return int(row["count"]) if row else 0


def clear_database():
    """Reset the database for development (drop all rows)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM jv_derived_metrics;")
    cursor.execute("DELETE FROM scientific_datasets;")
    conn.commit()
    conn.close()


# --- Derived metrics operations ---

def insert_derived_metrics(
    dataset_id: str,
    trace_id: str,
    j_sc: Optional[float],
    v_oc: Optional[float],
    p_max: Optional[float],
    ff: Optional[float],
    pce: Optional[float],
    extracted_at: str,
    provenance_notes: Optional[str] = None,
):
    """Insert or replace derived metrics for a specific curve trace."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO jv_derived_metrics (
            dataset_id, trace_id, j_sc, v_oc, p_max, ff, pce, extracted_at, provenance_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (dataset_id, trace_id, j_sc, v_oc, p_max, ff, pce, extracted_at, provenance_notes),
    )

    conn.commit()
    conn.close()


def get_derived_metrics(dataset_id: str, trace_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve derived metrics for a curve trace."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM jv_derived_metrics WHERE dataset_id = ? AND trace_id = ?;",
        (dataset_id, trace_id),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "dataset_id": row["dataset_id"],
        "trace_id": row["trace_id"],
        "j_sc": row["j_sc"],
        "v_oc": row["v_oc"],
        "p_max": row["p_max"],
        "ff": row["ff"],
        "pce": row["pce"],
        "extracted_at": row["extracted_at"],
        "provenance_notes": row["provenance_notes"],
    }


def list_all_derived_metrics() -> List[Dict[str, Any]]:
    """List all derived metrics in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jv_derived_metrics;")
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "dataset_id": r["dataset_id"],
            "trace_id": r["trace_id"],
            "j_sc": r["j_sc"],
            "v_oc": r["v_oc"],
            "p_max": r["p_max"],
            "ff": r["ff"],
            "pce": r["pce"],
            "extracted_at": r["extracted_at"],
            "provenance_notes": r["provenance_notes"],
        }
        for r in rows
    ]


def parse_and_persist_workbook_bytes(filename: str, file_bytes: bytes) -> List[ScientificDataset]:
    """
    Parse a workbook using parser_v2, convert to canonical,
    persist to SQLite, and automatically derive/persist JV parameters.
    """
    from app.parser_v2.reader.raw_reader import read_workbook
    from app.parser_v2.scanner.region_scanner import scan_sheet
    from app.parser_v2.reasoner.experiment_classifier import classify_sheet_regions
    from app.parser_v2.extractor.extract import extract_region
    from app.parser_v2.validator.validator import validate_dataset
    from app.canonical.converter import build_canonical_workbook
    from app.services.pv_extractor import extract_and_persist_pv_metrics

    raw_wb = read_workbook(filename, file_bytes)
    validated_ds_list = []

    for sname in raw_wb.sheet_names():
        sheet = raw_wb.get_sheet(sname)
        if not sheet:
            continue
        regions = scan_sheet(sheet)
        if not regions:
            continue
        interps = classify_sheet_regions(sheet, regions)
        for reg, interp in zip(regions, interps):
            ext_list = extract_region(sheet, reg, interp)
            for ext in ext_list:
                val = validate_dataset(ext)
                validated_ds_list.append(val)

    canon_wb = build_canonical_workbook(filename, validated_ds_list)
    for ds in canon_wb.datasets:
        insert_dataset(ds)
        if ds.scientific_type.upper() == "JV":
            extract_and_persist_pv_metrics(ds)

    return canon_wb.datasets


# Auto-initialize database on module import
initialize_database()


