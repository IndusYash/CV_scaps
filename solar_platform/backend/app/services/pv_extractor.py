"""
app.services.pv_extractor
-------------------------
Deterministic service that extracts derived PV metrics (Jsc, Voc, FF, PCE)
from actual extracted JV curves and persists them to the SQLite database.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, Optional

from app.canonical.model import ScientificDataset
from app.database import db_manager
from app.ml.modeling.pv_metrics import extract_pv_parameters


def extract_and_persist_pv_metrics(ds: ScientificDataset) -> int:
    """
    Extract PV parameters from each trace (JV curve) in a JV dataset,
    and persist them to the SQLite database.
    
    Returns:
        int: Number of traces successfully extracted and stored.
    """
    if ds.scientific_type.upper() != "JV" or not ds.x_values:
        return 0

    count = 0
    now_str = datetime.datetime.utcnow().isoformat() + "Z"

    for trace in ds.traces:
        # 1. Basic length validation
        if len(ds.x_values) < 2 or len(trace.y_values) != len(ds.x_values):
            db_manager.insert_derived_metrics(
                dataset_id=ds.dataset_id,
                trace_id=trace.trace_id,
                j_sc=None,
                v_oc=None,
                p_max=None,
                ff=None,
                pce=None,
                extracted_at=now_str,
                provenance_notes=f"Insufficient trace data: len(x)={len(ds.x_values)}, len(y)={len(trace.y_values)}",
            )
            continue

        # 2. Extract PV metrics
        pv = extract_pv_parameters(ds.x_values, trace.y_values)
        if pv is None:
            db_manager.insert_derived_metrics(
                dataset_id=ds.dataset_id,
                trace_id=trace.trace_id,
                j_sc=None,
                v_oc=None,
                p_max=None,
                ff=None,
                pce=None,
                extracted_at=now_str,
                provenance_notes="pv_metrics extraction returned None",
            )
            continue

        # 3. Store in DB
        db_manager.insert_derived_metrics(
            dataset_id=ds.dataset_id,
            trace_id=trace.trace_id,
            j_sc=pv.j_sc,
            v_oc=pv.v_oc,
            p_max=pv.p_max,
            ff=pv.ff,
            pce=pv.pce,
            extracted_at=now_str,
            provenance_notes="Deterministic PV metric extraction from JV curve trace",
        )
        count += 1

    return count
