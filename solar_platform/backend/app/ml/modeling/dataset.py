"""
app.ml.modeling.dataset
-----------------------
PyData Dataset container layer for ML training and evaluation.

Stores preprocessed input feature vectors, target values, and full metadata pointers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.ml.model import MLCurveRecord, MLScalarRecord
from app.ml.modeling.preprocessing import MLPreprocessor


class CurvePointDataset:
    """Dataset container for curve conditioning tasks f(material, params, x) -> y."""

    def __init__(self, records: List[MLCurveRecord], preprocessor: MLPreprocessor):
        self.records = records
        self.preprocessor = preprocessor
        self.features: List[List[float]] = []
        self.targets: List[float] = []

        for r in records:
            feat, target = preprocessor.transform_curve_record(r)
            self.features.append(feat)
            self.targets.append(target)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        return {
            "features": self.features[idx],
            "target": self.targets[idx],
            "record_id": rec.record_id,
            "dataset_id": rec.dataset_id,
            "group_id": rec.group_id,
            "material": rec.material,
            "scientific_type": rec.scientific_type,
            "trace_id": rec.trace_id,
            "x_native": rec.x,
            "y_native": rec.y,
            "x_provenance": rec.x_provenance.to_dict() if rec.x_provenance else None,
            "y_provenance": rec.y_provenance.to_dict() if rec.y_provenance else None,
        }

    def filter_by_scientific_types(self, scientific_types: List[str]) -> "CurvePointDataset":
        """Filter dataset records by scientific type with physical value bounds filtering."""
        stypes = set(scientific_types)
        filtered_recs = []
        for r in self.records:
            if r.scientific_type not in stypes:
                continue
            # Apply physical range plausibility checks
            if r.scientific_type == "JV" and not (-50.0 <= r.y <= 150.0):
                continue
            if r.scientific_type == "QE" and not (-5.0 <= r.y <= 150.0):
                continue
            if r.scientific_type == "RESPONSIVITY" and not (-1.0 <= r.y <= 20.0):
                continue
            filtered_recs.append(r)
        return CurvePointDataset(filtered_recs, self.preprocessor)


class ScalarRecordDataset:
    """Dataset container for scalar prediction tasks f(material, params) -> target_value."""

    def __init__(self, records: List[MLScalarRecord], preprocessor: MLPreprocessor):
        self.records = records
        self.preprocessor = preprocessor
        self.features: List[List[float]] = []
        self.targets: List[float] = []

        for r in records:
            mat_vec = preprocessor.transform_material_one_hot(r.material)
            param_vec = preprocessor.transform_parameters(r.parameters)
            feat = mat_vec + param_vec
            self.features.append(feat)
            self.targets.append(r.target_value)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        return {
            "features": self.features[idx],
            "target": self.targets[idx],
            "record_id": rec.record_id,
            "dataset_id": rec.dataset_id,
            "group_id": rec.group_id,
            "material": rec.material,
            "scientific_type": rec.scientific_type,
            "target_name": rec.target_name,
            "target_value": rec.target_value,
            "target_provenance": rec.target_provenance.to_dict() if rec.target_provenance else None,
        }
