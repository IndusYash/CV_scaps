"""
app.ml.deduplication
--------------------
Duplicate Family Resolution and Cross-Workbook Deduplication layer.

Calculates stable experiment_family_id for physical experiments.
Resolves EXACT_REPLICA families by selecting one canonical training representative
without mutating canonical datasets or losing provenance.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.canonical.model import ScientificDataset


class DuplicateStatus(str, Enum):
    EXACT_REPLICA = "EXACT_REPLICA"
    LIKELY_REPLICA = "LIKELY_REPLICA"
    RELATED_BUT_DISTINCT = "RELATED_BUT_DISTINCT"
    UNRESOLVED = "UNRESOLVED"
    UNIQUE = "UNIQUE"


@dataclass
class DuplicateFamily:
    """Group of datasets sharing a physical experiment family identity."""
    experiment_family_id: str
    material: str
    scientific_type: str
    duplicate_status: DuplicateStatus
    datasets: List[ScientificDataset]
    representative_dataset_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_family_id": self.experiment_family_id,
            "material": self.material,
            "scientific_type": self.scientific_type,
            "duplicate_status": self.duplicate_status.value,
            "datasets_count": len(self.datasets),
            "dataset_ids": [d.dataset_id for d in self.datasets],
            "representative_dataset_id": self.representative_dataset_id,
        }


def _derive_material_key(workbook_name: str, sheet_name: str) -> str:
    wb_clean = Path(workbook_name).stem if workbook_name else ""
    for mat in ["Ca3AsBr3", "Ca3PBr3", "Ca3PI3"]:
        if mat.lower() in wb_clean.lower() or mat.lower() in sheet_name.lower():
            return mat
    if wb_clean:
        # Standardize Data file vs Data File (1) to same material identity
        if "data file" in wb_clean.lower():
            return "SolarDevice_DataFile"
        return wb_clean
    return "SolarDevice"


def compute_experiment_family_id(dataset: ScientificDataset) -> str:
    """
    Construct a deterministic experiment_family_id using physical content signature.
    Combines material, scientific_type, sheet/context, trace counts, x-grid limits, and parameter values.
    """
    mat = _derive_material_key(dataset.workbook_name, dataset.sheet_name)
    parent_or_region = dataset.parent_region_id or dataset.region_id

    # Collect parameter signature
    glob_p = {p.name: p.value for p in dataset.global_parameters if p.name}
    trace_p = []
    for tr in dataset.traces:
        if tr.parameter and tr.parameter.name:
            trace_p.append(f"{tr.parameter.name}={tr.parameter.value}")

    x_sig = (
        round(dataset.x_values[0], 4),
        round(dataset.x_values[-1], 4),
        len(dataset.x_values),
    ) if dataset.x_values else ("no_x",)

    y_sig = len(dataset.traces)

    sig_payload = {
        "material": mat,
        "scientific_type": dataset.scientific_type,
        "sheet_name": dataset.sheet_name,
        "parent_or_region": parent_or_region,
        "global_params": glob_p,
        "trace_params": sorted(trace_p),
        "x_sig": x_sig,
        "y_sig": y_sig,
    }

    payload_json = json.dumps(sig_payload, sort_keys=True)
    hash_str = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:12]
    return f"fam_{mat}_{dataset.sheet_name}_{dataset.scientific_type}_{hash_str}"


def resolve_duplicate_families(
    datasets: List[ScientificDataset],
) -> List[DuplicateFamily]:
    """
    Group datasets by experiment_family_id and classify duplicate status.
    Selects one representative dataset_id for each EXACT_REPLICA family.
    """
    family_map = defaultdict(list)
    for ds in datasets:
        fam_id = compute_experiment_family_id(ds)
        family_map[fam_id].append(ds)

    results: List[DuplicateFamily] = []

    for fam_id, ds_list in family_map.items():
        mat = _derive_material_key(ds_list[0].workbook_name, ds_list[0].sheet_name)
        stype = ds_list[0].scientific_type

        if len(ds_list) == 1:
            status = DuplicateStatus.UNIQUE
            rep_id = ds_list[0].dataset_id
        else:
            # Check if datasets come from replica workbooks (e.g. Data file vs Data File (1))
            wb_names = set(d.workbook_name for d in ds_list)
            if len(wb_names) > 1 and ("Data File (1).xlsx" in wb_names and "Data file.xlsx" in wb_names):
                status = DuplicateStatus.EXACT_REPLICA
            else:
                status = DuplicateStatus.LIKELY_REPLICA

            # Deterministically pick representative: prefer non-"(1)" workbook or first dataset
            rep_ds = sorted(ds_list, key=lambda d: ("(1)" in d.workbook_name, d.dataset_id))[0]
            rep_id = rep_ds.dataset_id

        results.append(
            DuplicateFamily(
                experiment_family_id=fam_id,
                material=mat,
                scientific_type=stype,
                duplicate_status=status,
                datasets=ds_list,
                representative_dataset_id=rep_id,
            )
        )

    return results
