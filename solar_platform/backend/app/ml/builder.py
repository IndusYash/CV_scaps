"""
app.ml.builder
--------------
ML Dataset Builder layer.

Consumes ONLY canonical ScientificDataset / CanonicalWorkbook objects from app.canonical.
Contains NO Excel layout logic, NO openpyxl references, and NO parser structural hypothesis dependencies.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.canonical.model import CanonicalWorkbook, ScientificDataset
from app.ml.model import MLCurveRecord, MLDatasetCollection, MLScalarRecord


def _derive_material_name(workbook_name: str, sheet_name: str) -> str:
    """Extract material / compound identity from workbook or sheet name."""
    wb_clean = Path(workbook_name).stem if workbook_name else ""
    for mat in ["Ca3AsBr3", "Ca3PBr3", "Ca3PI3"]:
        if mat.lower() in wb_clean.lower() or mat.lower() in sheet_name.lower():
            return mat
    if wb_clean:
        return wb_clean
    return "SolarDevice"


def _derive_group_id(dataset: ScientificDataset) -> str:
    """
    Generate experiment / sweep group ID for leakage-proof data splitting.
    All related traces belonging to the same sheet / parent sweep belong to the same group_id.
    """
    mat = _derive_material_name(dataset.workbook_name, dataset.sheet_name)
    parent = dataset.parent_region_id or dataset.region_id
    clean_parent = re.sub(r"[^\w\-.]", "_", parent)
    return f"grp_{mat}_{dataset.sheet_name}_{clean_parent}_{dataset.scientific_type}"


class MLDatasetBuilder:
    """
    ML Dataset Builder converting canonical ScientificDataset collections into MLDatasetCollection.

    Configuration:
      - include_warnings: bool = True (include ACCEPTED_WITH_WARNINGS datasets)
      - include_uncertain: bool = False (exclude UNCERTAIN datasets by default)
      - include_unresolved: bool = False (exclude UNRESOLVED datasets by default)
    """

    def __init__(
        self,
        include_warnings: bool = True,
        include_uncertain: bool = False,
        include_unresolved: bool = False,
    ):
        self.include_warnings = include_warnings
        self.include_uncertain = include_uncertain
        self.include_unresolved = include_unresolved

    def is_dataset_included(self, dataset: ScientificDataset) -> bool:
        """Filter dataset based on validation status and flags."""
        st = dataset.validation_status
        if st == "ACCEPTED_VALID":
            return True
        if st == "ACCEPTED_WITH_WARNINGS":
            return self.include_warnings
        if st == "UNCERTAIN":
            return self.include_uncertain
        if st in ["UNRESOLVED", "INVALID_STRUCTURE"]:
            return self.include_unresolved
        return False

    def build_from_canonical_dataset(self, dataset: ScientificDataset) -> MLDatasetCollection:
        """Convert a single ScientificDataset into ML curve and scalar records."""
        curve_records: List[MLCurveRecord] = []
        scalar_records: List[MLScalarRecord] = []

        if not self.is_dataset_included(dataset):
            return MLDatasetCollection(curve_records=[], scalar_records=[], metadata={"dataset_id": dataset.dataset_id, "status": "excluded"})

        material = _derive_material_name(dataset.workbook_name, dataset.sheet_name)
        group_id = _derive_group_id(dataset)

        glob_params: Dict[str, Any] = {
            p.name: p.value for p in dataset.global_parameters if p.name and p.value is not None
        }

        if dataset.traces and dataset.x_values:
            for trace in dataset.traces:
                trace_params = dict(glob_params)
                if trace.parameter and trace.parameter.name and trace.parameter.value is not None:
                    trace_params[trace.parameter.name] = trace.parameter.value

                axis_name = dataset.axis_name or "x"
                target_name = dataset.target_name or "y"

                for pt_idx, (x_val, y_val) in enumerate(zip(dataset.x_values, trace.y_values)):
                    rec_id = f"{dataset.dataset_id}_{trace.trace_id}_pt_{pt_idx + 1}"

                    x_prov = dataset.x_provenance[pt_idx] if pt_idx < len(dataset.x_provenance) else None
                    y_prov = trace.y_provenance[pt_idx] if pt_idx < len(trace.y_provenance) else None

                    curve_records.append(
                        MLCurveRecord(
                            record_id=rec_id,
                            dataset_id=dataset.dataset_id,
                            group_id=group_id,
                            material=material,
                            scientific_type=dataset.scientific_type,
                            axis_name=axis_name,
                            axis_unit=dataset.axis_unit,
                            target_name=target_name,
                            target_unit=dataset.target_unit,
                            trace_id=trace.trace_id,
                            parameters=trace_params,
                            x=float(x_val),
                            y=float(y_val),
                            x_provenance=x_prov,
                            y_provenance=y_prov,
                            validation_status=dataset.validation_status,
                        )
                    )

        if dataset.scalar_records:
            for rec_idx, s_rec in enumerate(dataset.scalar_records):
                scalar_params = dict(glob_params)
                if s_rec.parameter and s_rec.parameter.name and s_rec.parameter.value is not None:
                    scalar_params[s_rec.parameter.name] = s_rec.parameter.value

                rec_id = f"{dataset.dataset_id}_{s_rec.record_id}"

                scalar_records.append(
                    MLScalarRecord(
                        record_id=rec_id,
                        dataset_id=dataset.dataset_id,
                        group_id=group_id,
                        material=material,
                        scientific_type=dataset.scientific_type,
                        parameters=scalar_params,
                        target_name=s_rec.target_name,
                        target_value=s_rec.target_value,
                        target_unit=s_rec.target_unit,
                        target_provenance=s_rec.target_provenance,
                        validation_status=dataset.validation_status,
                    )
                )

        return MLDatasetCollection(
            curve_records=curve_records,
            scalar_records=scalar_records,
            metadata={
                "dataset_id": dataset.dataset_id,
                "material": material,
                "group_id": group_id,
                "status": dataset.validation_status,
            },
        )

    def build_from_canonical_workbook(self, workbook: CanonicalWorkbook) -> MLDatasetCollection:
        """Convert an entire CanonicalWorkbook into a unified MLDatasetCollection."""
        all_curves: List[MLCurveRecord] = []
        all_scalars: List[MLScalarRecord] = []
        dataset_summaries: List[Dict[str, Any]] = []

        for ds in workbook.datasets:
            col = self.build_from_canonical_dataset(ds)
            all_curves.extend(col.curve_records)
            all_scalars.extend(col.scalar_records)
            dataset_summaries.append(col.metadata)

        return MLDatasetCollection(
            curve_records=all_curves,
            scalar_records=all_scalars,
            metadata={
                "workbook_name": workbook.workbook_name,
                "total_canonical_datasets": len(workbook.datasets),
                "ml_ready_curves_count": len(all_curves),
                "ml_ready_scalars_count": len(all_scalars),
                "dataset_summaries": dataset_summaries,
            },
        )
