"""
app.ml.model
------------
Data models for ML Dataset Preparation layer.

Consumes ONLY canonical ScientificDataset objects from app.canonical.
Independent of Excel layouts or parser structural hypotheses.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.canonical.model import CellRefDTO


@dataclass
class MLCurveRecord:
    """Long-format native curve point record for ML preprocessing."""
    record_id: str                          # {dataset_id}_{trace_id}_{point_idx}
    dataset_id: str                         # Canonical ScientificDataset ID
    group_id: str                           # Experiment/sweep group ID for leakage-proof splitting
    material: str                           # Material / workbook identity (e.g. "Ca3PI3", "Ca3PBr3")
    scientific_type: str                    # "JV", "QE", "RESPONSIVITY", "DETECTIVITY"
    axis_name: str                          # e.g. "Voltage", "Wavelength"
    axis_unit: Optional[str]                # e.g. "V", "nm"
    target_name: Optional[str]              # e.g. "CurrentDensity", "QuantumEfficiency"
    target_unit: Optional[str]              # e.g. "mA/cm2", "%"
    trace_id: str
    parameters: Dict[str, Any]              # Dict of arbitrary parameter_name -> parameter_value
    x: float                                # Native x coordinate
    y: float                                # Native y coordinate
    x_provenance: Optional[CellRefDTO] = None
    y_provenance: Optional[CellRefDTO] = None
    validation_status: str = "ACCEPTED_VALID"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "dataset_id": self.dataset_id,
            "group_id": self.group_id,
            "material": self.material,
            "scientific_type": self.scientific_type,
            "axis_name": self.axis_name,
            "axis_unit": self.axis_unit,
            "target_name": self.target_name,
            "target_unit": self.target_unit,
            "trace_id": self.trace_id,
            "parameters": self.parameters,
            "x": self.x,
            "y": self.y,
            "x_provenance": self.x_provenance.to_dict() if self.x_provenance else None,
            "y_provenance": self.y_provenance.to_dict() if self.y_provenance else None,
            "validation_status": self.validation_status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MLCurveRecord:
        x_prov = CellRefDTO.from_dict(data["x_provenance"]) if data.get("x_provenance") else None
        y_prov = CellRefDTO.from_dict(data["y_provenance"]) if data.get("y_provenance") else None
        return cls(
            record_id=data.get("record_id", ""),
            dataset_id=data.get("dataset_id", ""),
            group_id=data.get("group_id", ""),
            material=data.get("material", "Unknown"),
            scientific_type=data.get("scientific_type", "UNKNOWN"),
            axis_name=data.get("axis_name", "x"),
            axis_unit=data.get("axis_unit"),
            target_name=data.get("target_name", "y"),
            target_unit=data.get("target_unit"),
            trace_id=data.get("trace_id", ""),
            parameters=data.get("parameters", {}),
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            x_provenance=x_prov,
            y_provenance=y_prov,
            validation_status=data.get("validation_status", "ACCEPTED_VALID"),
        )


@dataclass
class MLScalarRecord:
    """Long-format scalar metric mapping record for ML preprocessing."""
    record_id: str                          # {dataset_id}_scalar_{rec_idx}
    dataset_id: str
    group_id: str
    material: str
    scientific_type: str                    # "SCALAR_SWEEP"
    parameters: Dict[str, Any]              # Dict of arbitrary parameter_name -> parameter_value
    target_name: str                        # "Voc", "Jsc", "FF", "PCE", "Rsh", "Rs"
    target_value: float                     # Native target float value
    target_unit: Optional[str] = None
    target_provenance: Optional[CellRefDTO] = None
    validation_status: str = "ACCEPTED_VALID"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "dataset_id": self.dataset_id,
            "group_id": self.group_id,
            "material": self.material,
            "scientific_type": self.scientific_type,
            "parameters": self.parameters,
            "target_name": self.target_name,
            "target_value": self.target_value,
            "target_unit": self.target_unit,
            "target_provenance": self.target_provenance.to_dict() if self.target_provenance else None,
            "validation_status": self.validation_status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MLScalarRecord:
        prov = CellRefDTO.from_dict(data["target_provenance"]) if data.get("target_provenance") else None
        return cls(
            record_id=data.get("record_id", ""),
            dataset_id=data.get("dataset_id", ""),
            group_id=data.get("group_id", ""),
            material=data.get("material", "Unknown"),
            scientific_type=data.get("scientific_type", "SCALAR_SWEEP"),
            parameters=data.get("parameters", {}),
            target_name=data.get("target_name", ""),
            target_value=float(data.get("target_value", 0.0)),
            target_unit=data.get("target_unit"),
            target_provenance=prov,
            validation_status=data.get("validation_status", "ACCEPTED_VALID"),
        )


@dataclass
class MLDatasetCollection:
    """Collection containing long-format curve points and scalar records for ML systems."""
    curve_records: List[MLCurveRecord] = field(default_factory=list)
    scalar_records: List[MLScalarRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "curve_records_count": len(self.curve_records),
            "scalar_records_count": len(self.scalar_records),
            "curve_records": [r.to_dict() for r in self.curve_records],
            "scalar_records": [s.to_dict() for s in self.scalar_records],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MLDatasetCollection:
        curves = [MLCurveRecord.from_dict(r) for r in data.get("curve_records", [])]
        scalars = [MLScalarRecord.from_dict(s) for s in data.get("scalar_records", [])]
        return cls(
            curve_records=curves,
            scalar_records=scalars,
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> MLDatasetCollection:
        return cls.from_dict(json.loads(json_str))

    def to_csv_curves(self) -> str:
        """Export long-format curve records to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "record_id",
            "dataset_id",
            "group_id",
            "material",
            "scientific_type",
            "trace_id",
            "parameters_json",
            "axis_name",
            "axis_unit",
            "x",
            "target_name",
            "target_unit",
            "y",
            "validation_status",
        ])
        for r in self.curve_records:
            writer.writerow([
                r.record_id,
                r.dataset_id,
                r.group_id,
                r.material,
                r.scientific_type,
                r.trace_id,
                json.dumps(r.parameters),
                r.axis_name,
                r.axis_unit or "",
                r.x,
                r.target_name or "",
                r.target_unit or "",
                r.y,
                r.validation_status,
            ])
        return output.getvalue()

    def to_csv_scalars(self) -> str:
        """Export long-format scalar records to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "record_id",
            "dataset_id",
            "group_id",
            "material",
            "scientific_type",
            "parameters_json",
            "target_name",
            "target_value",
            "target_unit",
            "validation_status",
        ])
        for s in self.scalar_records:
            writer.writerow([
                s.record_id,
                s.dataset_id,
                s.group_id,
                s.material,
                s.scientific_type,
                json.dumps(s.parameters),
                s.target_name,
                s.target_value,
                s.target_unit or "",
                s.validation_status,
            ])
        return output.getvalue()
