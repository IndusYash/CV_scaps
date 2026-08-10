"""
app.ml.curation_pipeline
------------------------
ML Dataset Curation Pipeline.

Assembles:
  - FULL_AUDIT_DATASET view
  - TRAINING_DATASET view
  - REGIME_A_IN_DOMAIN split (70/15/15 by experiment_family_id)
  - REGIME_B_CROSS_MATERIAL split (Train on Ca3PI3/Ca3PBr3/DataFile, Test on Ca3AsBr3)
  - Training manifest export (JSON & Markdown)

Preserves 100% native x/y values, 100% canonical source objects, and 0 data mutation.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.canonical.model import CanonicalParameter, CanonicalWorkbook, ScientificDataset
from app.ml.builder import MLDatasetBuilder
from app.ml.deduplication import (
    DuplicateFamily,
    DuplicateStatus,
    compute_experiment_family_id,
    resolve_duplicate_families,
)
from app.ml.model import MLCurveRecord, MLDatasetCollection, MLScalarRecord
from app.ml.parameter_curation import CuratedParameter, curate_parameter


@dataclass
class ManifestEntry:
    """Individual dataset entry for ML training manifest."""
    dataset_id: str
    experiment_family_id: str
    group_id: str
    material: str
    scientific_type: str
    structural_source: str
    parameters: Dict[str, Any]
    traces_count: int
    points_count: int
    scalar_records_count: int
    validation_status: str
    regime_a_split: str                     # "train", "val", "test"
    regime_b_split: str                     # "train", "test"
    curation_status: str
    duplicate_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "experiment_family_id": self.experiment_family_id,
            "group_id": self.group_id,
            "material": self.material,
            "scientific_type": self.scientific_type,
            "structural_source": self.structural_source,
            "parameters": self.parameters,
            "traces_count": self.traces_count,
            "points_count": self.points_count,
            "scalar_records_count": self.scalar_records_count,
            "validation_status": self.validation_status,
            "regime_a_split": self.regime_a_split,
            "regime_b_split": self.regime_b_split,
            "curation_status": self.curation_status,
            "duplicate_status": self.duplicate_status,
        }


@dataclass
class MLCurationResult:
    """Container for curated dataset views, split assignments, and manifests."""
    full_audit_collection: MLDatasetCollection
    training_collection: MLDatasetCollection
    duplicate_families: List[DuplicateFamily]
    manifest_entries: List[ManifestEntry]

    regime_a_train_family_ids: List[str]
    regime_a_val_family_ids: List[str]
    regime_a_test_family_ids: List[str]

    regime_b_train_family_ids: List[str]
    regime_b_test_family_ids: List[str]

    is_regime_a_leakage_free: bool
    is_regime_b_material_isolated: bool

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "full_audit_datasets_count": len(self.full_audit_collection.metadata.get("datasets", [])),
            "training_datasets_count": len(self.training_collection.metadata.get("datasets", [])),
            "duplicate_families_count": len(self.duplicate_families),
            "exact_replicas_removed_count": sum(1 for f in self.duplicate_families if f.duplicate_status == DuplicateStatus.EXACT_REPLICA),
            "training_curve_points_count": len(self.training_collection.curve_records),
            "training_scalar_records_count": len(self.training_collection.scalar_records),
            "regime_a": {
                "train_families": len(self.regime_a_train_family_ids),
                "val_families": len(self.regime_a_val_family_ids),
                "test_families": len(self.regime_a_test_family_ids),
                "is_leakage_free": self.is_regime_a_leakage_free,
            },
            "regime_b": {
                "train_families": len(self.regime_b_train_family_ids),
                "test_families": len(self.regime_b_test_family_ids),
                "is_material_isolated": self.is_regime_b_material_isolated,
            },
        }


class MLCurationPipeline:
    """
    ML Dataset Curation Pipeline building FULL_AUDIT_DATASET and TRAINING_DATASET views
    with non-mutating parameter curation, cross-workbook deduplication, and split regimes.
    """

    def __init__(
        self,
        builder: Optional[MLDatasetBuilder] = None,
        seed: int = 42,
    ):
        if builder is None:
            builder = MLDatasetBuilder(include_warnings=True, include_uncertain=False, include_unresolved=False)
        self.builder = builder
        self.seed = seed

    def execute(self, canonical_workbooks: List[CanonicalWorkbook]) -> MLCurationResult:
        all_ml_ready_ds: List[ScientificDataset] = []
        for wb in canonical_workbooks:
            for ds in wb.datasets:
                if self.builder.is_dataset_included(ds):
                    all_ml_ready_ds.append(ds)

        # 1. Deduplication & Duplicate Family Resolution
        families = resolve_duplicate_families(all_ml_ready_ds)
        family_lookup = {}
        for fam in families:
            for ds in fam.datasets:
                family_lookup[ds.dataset_id] = fam

        # Collect training representative datasets (1 per EXACT_REPLICA family)
        training_ds_list: List[ScientificDataset] = []
        full_audit_ds_list: List[ScientificDataset] = all_ml_ready_ds

        for fam in families:
            if fam.duplicate_status == DuplicateStatus.EXACT_REPLICA:
                rep = next(d for d in fam.datasets if d.dataset_id == fam.representative_dataset_id)
                training_ds_list.append(rep)
            else:
                training_ds_list.extend(fam.datasets)

        # 2. Build FULL_AUDIT_DATASET collection
        full_curves: List[MLCurveRecord] = []
        full_scalars: List[MLScalarRecord] = []

        for wb in canonical_workbooks:
            col = self.builder.build_from_canonical_workbook(wb)
            full_curves.extend(col.curve_records)
            full_scalars.extend(col.scalar_records)

        full_audit_collection = MLDatasetCollection(
            curve_records=full_curves,
            scalar_records=full_scalars,
            metadata={"datasets": [d.dataset_id for d in full_audit_ds_list]},
        )

        # 3. Build TRAINING_DATASET collection with Curated Parameters
        train_curves: List[MLCurveRecord] = []
        train_scalars: List[MLScalarRecord] = []

        for ds in training_ds_list:
            col = self.builder.build_from_canonical_dataset(ds)

            # Apply parameter curation to curve records
            for cr in col.curve_records:
                curated_params = {}
                for p_name, p_val in cr.parameters.items():
                    c_param = curate_parameter(
                        param=CanonicalParameter(name=p_name, value=p_val),
                        workbook_name=ds.workbook_name,
                        sheet_name=ds.sheet_name,
                        region_id=ds.region_id,
                        scientific_type=ds.scientific_type,
                    )
                    curated_params[c_param.curated_name] = c_param.curated_value
                cr.parameters = curated_params
                train_curves.append(cr)

            # Apply parameter curation to scalar records
            for sr in col.scalar_records:
                curated_params = {}
                for p_name, p_val in sr.parameters.items():
                    c_param = curate_parameter(
                        param=CanonicalParameter(name=p_name, value=p_val),
                        workbook_name=ds.workbook_name,
                        sheet_name=ds.sheet_name,
                        region_id=ds.region_id,
                        scientific_type=ds.scientific_type,
                    )
                    curated_params[c_param.curated_name] = c_param.curated_value
                sr.parameters = curated_params
                train_scalars.append(sr)

        training_collection = MLDatasetCollection(
            curve_records=train_curves,
            scalar_records=train_scalars,
            metadata={"datasets": [d.dataset_id for d in training_ds_list]},
        )

        # 4. Construct REGIME_A_IN_DOMAIN Split by experiment_family_id
        all_family_ids = sorted(list(set(f.experiment_family_id for f in families)))
        rng = random.Random(self.seed)
        shuffled_fams = list(all_family_ids)
        rng.shuffle(shuffled_fams)

        n_fams = len(shuffled_fams)
        n_train = max(1, int(n_fams * 0.70))
        n_val = max(1, int(n_fams * 0.15))

        reg_a_train = set(shuffled_fams[:n_train])
        reg_a_val = set(shuffled_fams[n_train:n_train + n_val])
        reg_a_test = set(shuffled_fams[n_train + n_val:])

        is_reg_a_leakage_free = (
            len(reg_a_train.intersection(reg_a_val)) == 0
            and len(reg_a_train.intersection(reg_a_test)) == 0
            and len(reg_a_val.intersection(reg_a_test)) == 0
        )

        # 5. Construct REGIME_B_CROSS_MATERIAL Split
        reg_b_train_fams = set()
        reg_b_test_fams = set()

        for fam in families:
            if "Ca3AsBr3" in fam.material or "Ca3AsBr3" in fam.datasets[0].workbook_name:
                reg_b_test_fams.add(fam.experiment_family_id)
            else:
                reg_b_train_fams.add(fam.experiment_family_id)

        is_reg_b_material_isolated = len(reg_b_train_fams.intersection(reg_b_test_fams)) == 0

        # 6. Build Manifest Entries
        manifest_entries: List[ManifestEntry] = []
        for ds in training_ds_list:
            fam = family_lookup[ds.dataset_id]
            fam_id = fam.experiment_family_id

            split_a = "train" if fam_id in reg_a_train else ("val" if fam_id in reg_a_val else "test")
            split_b = "test" if fam_id in reg_b_test_fams else "train"

            # Curate parameters summary
            param_dict = {}
            for p in ds.global_parameters:
                if p.name:
                    cp = curate_parameter(p, ds.workbook_name, ds.sheet_name, ds.region_id, ds.scientific_type)
                    param_dict[cp.curated_name] = cp.curated_value

            pts_cnt = len(ds.x_values) * len(ds.traces) if ds.x_values else 0
            scal_cnt = len(ds.scalar_records)

            manifest_entries.append(
                ManifestEntry(
                    dataset_id=ds.dataset_id,
                    experiment_family_id=fam_id,
                    group_id=f"grp_{ds.workbook_name.replace('.xlsx', '')}_{ds.sheet_name}_{ds.region_id}",
                    material=ds.workbook_name.replace(".xlsx", ""),
                    scientific_type=ds.scientific_type,
                    structural_source=ds.structural_source,
                    parameters=param_dict,
                    traces_count=len(ds.traces),
                    points_count=pts_cnt,
                    scalar_records_count=scal_cnt,
                    validation_status=ds.validation_status,
                    regime_a_split=split_a,
                    regime_b_split=split_b,
                    curation_status="CURATED",
                    duplicate_status=fam.duplicate_status.value,
                )
            )

        return MLCurationResult(
            full_audit_collection=full_audit_collection,
            training_collection=training_collection,
            duplicate_families=families,
            manifest_entries=manifest_entries,
            regime_a_train_family_ids=sorted(list(reg_a_train)),
            regime_a_val_family_ids=sorted(list(reg_a_val)),
            regime_a_test_family_ids=sorted(list(reg_a_test)),
            regime_b_train_family_ids=sorted(list(reg_b_train_fams)),
            regime_b_test_family_ids=sorted(list(reg_b_test_fams)),
            is_regime_a_leakage_free=is_reg_a_leakage_free,
            is_regime_b_material_isolated=is_reg_b_material_isolated,
        )


def export_training_manifest_json(result: MLCurationResult) -> str:
    """Export ML training manifest as formatted JSON string."""
    payload = {
        "summary": result.to_summary_dict(),
        "manifest_entries": [e.to_dict() for e in result.manifest_entries],
    }
    return json.dumps(payload, indent=2)


def export_training_manifest_md(result: MLCurationResult) -> str:
    """Export ML training manifest as Markdown report."""
    lines = [
        "# ML Training Manifest",
        "",
        "## Summary",
        f"- **Full Audit Datasets**: {len(result.full_audit_collection.metadata.get('datasets', []))}",
        f"- **Training Representative Datasets**: {len(result.training_collection.metadata.get('datasets', []))}",
        f"- **Total Experiment Families**: {len(result.duplicate_families)}",
        f"- **Exact Replica Families Removed**: {sum(1 for f in result.duplicate_families if f.duplicate_status == DuplicateStatus.EXACT_REPLICA)}",
        f"- **Training Native Curve Points**: {len(result.training_collection.curve_records)}",
        f"- **Training Scalar Records**: {len(result.training_collection.scalar_records)}",
        "",
        "## Regime A (In-Domain 70/15/15 Split)",
        f"- **Train Families**: {len(result.regime_a_train_family_ids)}",
        f"- **Validation Families**: {len(result.regime_a_val_family_ids)}",
        f"- **Test Families**: {len(result.regime_a_test_family_ids)}",
        f"- **Leakage Free Status**: {'100% LEAKAGE FREE' if result.is_regime_a_leakage_free else 'LEAKAGE DETECTED'}",
        "",
        "## Regime B (Cross-Material Holdout)",
        f"- **Train Families (Ca3PI3 + Ca3PBr3 + DataFile)**: {len(result.regime_b_train_family_ids)}",
        f"- **Test Families (Ca3AsBr3 Holdout)**: {len(result.regime_b_test_family_ids)}",
        f"- **Material Isolated Status**: {'100% ISOLATED' if result.is_regime_b_material_isolated else 'CROSS-TALK DETECTED'}",
        "",
        "## Manifest Dataset Entries",
        "| Dataset ID | Family ID | Material | Scientific Type | Traces | Points | Regime A Split | Regime B Split |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for e in result.manifest_entries[:20]:
        lines.append(
            f"| `{e.dataset_id[:25]}` | `{e.experiment_family_id[:18]}` | {e.material} | {e.scientific_type} | {e.traces_count} | {e.points_count} | {e.regime_a_split} | {e.regime_b_split} |"
        )
    if len(result.manifest_entries) > 20:
        lines.append(f"| ... and {len(result.manifest_entries) - 20} more datasets | | | | | | | |")

    return "\n".join(lines)
