"""
app.canonical.model
-------------------
Unified Canonical Scientific Dataset representation.

Standardized external representation between parser_v2 and future ML systems.
Independent of whether the source structure was NumericCurveGrid, ScalarSweepTable,
StackedCurveBlocks, InterleavedCurveGrid, or CategoryTable.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class CellRefDTO:
    """CellRef Data Transfer Object for complete provenance tracking."""
    sheet_name: str
    row: int
    col: int
    raw_value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sheet_name": self.sheet_name,
            "row": self.row,
            "col": self.col,
            "raw_value": self.raw_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CellRefDTO:
        return cls(
            sheet_name=data.get("sheet_name", ""),
            row=int(data.get("row", 0)),
            col=int(data.get("col", 0)),
            raw_value=data.get("raw_value"),
        )

    @classmethod
    def from_cell_ref(cls, cell_ref: Any) -> Optional[CellRefDTO]:
        if cell_ref is None:
            return None
        return cls(
            sheet_name=getattr(cell_ref, "sheet_name", ""),
            row=getattr(cell_ref, "row", 0),
            col=getattr(cell_ref, "col", 0),
            raw_value=getattr(cell_ref, "raw_value", None),
        )


@dataclass
class CanonicalParameter:
    """Typed parameter key-value pair with unit, raw text, and CellRef provenance."""
    name: str
    value: Union[float, int, str, None]
    unit: Optional[str] = None
    raw_text: Optional[str] = None
    provenance: Optional[CellRefDTO] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "raw_text": self.raw_text,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanonicalParameter:
        prov = CellRefDTO.from_dict(data["provenance"]) if data.get("provenance") else None
        return cls(
            name=data.get("name", ""),
            value=data.get("value"),
            unit=data.get("unit"),
            raw_text=data.get("raw_text"),
            provenance=prov,
        )


@dataclass
class CanonicalTrace:
    """One y-trace in a curve dataset retaining parameter identity and provenance."""
    trace_index: int
    trace_id: str
    parameter: Optional[CanonicalParameter] = None
    y_values: List[float] = field(default_factory=list)
    y_provenance: List[CellRefDTO] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_index": self.trace_index,
            "trace_id": self.trace_id,
            "parameter": self.parameter.to_dict() if self.parameter else None,
            "y_values": self.y_values,
            "y_provenance": [p.to_dict() for p in self.y_provenance],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanonicalTrace:
        param = CanonicalParameter.from_dict(data["parameter"]) if data.get("parameter") else None
        prov = [CellRefDTO.from_dict(p) for p in data.get("y_provenance", [])]
        return cls(
            trace_index=int(data.get("trace_index", 0)),
            trace_id=data.get("trace_id", ""),
            parameter=param,
            y_values=[float(v) for v in data.get("y_values", [])],
            y_provenance=prov,
            metadata=data.get("metadata", {}),
        )


@dataclass
class CanonicalScalarRecord:
    """A scalar sweep mapping (e.g. thickness=100nm -> Jsc=23.4 mA/cm2)."""
    record_id: str
    parameter: CanonicalParameter
    target_name: str
    target_value: float
    target_unit: Optional[str] = None
    target_provenance: Optional[CellRefDTO] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "parameter": self.parameter.to_dict(),
            "target_name": self.target_name,
            "target_value": self.target_value,
            "target_unit": self.target_unit,
            "target_provenance": self.target_provenance.to_dict() if self.target_provenance else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanonicalScalarRecord:
        param = CanonicalParameter.from_dict(data["parameter"])
        target_prov = CellRefDTO.from_dict(data["target_provenance"]) if data.get("target_provenance") else None
        return cls(
            record_id=data.get("record_id", ""),
            parameter=param,
            target_name=data.get("target_name", ""),
            target_value=float(data.get("target_value", 0.0)),
            target_unit=data.get("target_unit"),
            target_provenance=target_prov,
        )


@dataclass
class ScientificDataset:
    """
    Unified Canonical Scientific Dataset.

    Normalizes all parser_v2 structural forms (NumericCurveGrid, ScalarSweepTable,
    StackedCurveBlocks, InterleavedCurveGrid, CategoryTable) into a clean,
    standardized representation for ML systems while preserving complete provenance.
    """
    dataset_id: str
    workbook_name: str
    sheet_name: str
    region_id: str
    parent_region_id: Optional[str]
    scientific_type: str                  # "JV", "QE", "RESPONSIVITY", "DETECTIVITY", "SCALAR_SWEEP", "UNKNOWN"
    structural_source: str                # "NumericCurveGrid", "ScalarSweepTable", "StackedCurveBlocks", "InterleavedCurveGrid", "CategoryTable"

    # Metadata & Concepts
    axis_name: Optional[str] = None
    axis_unit: Optional[str] = None
    target_name: Optional[str] = None
    target_unit: Optional[str] = None
    global_parameters: List[CanonicalParameter] = field(default_factory=list)

    # X-Axis Data (for curve datasets)
    x_values: List[float] = field(default_factory=list)
    x_provenance: List[CellRefDTO] = field(default_factory=list)

    # Multi-Trace Data (for curve datasets)
    traces: List[CanonicalTrace] = field(default_factory=list)

    # Scalar Records (for scalar sweep datasets)
    scalar_records: List[CanonicalScalarRecord] = field(default_factory=list)

    # Validation & Reasoner Metadata
    validation_status: str = "ACCEPTED_VALID"   # "ACCEPTED_VALID", "ACCEPTED_WITH_WARNINGS", "UNCERTAIN", "UNRESOLVED"
    acceptance_state: str = "ACCEPTED"         # "ACCEPTED", "UNCERTAIN", "UNRESOLVED"
    is_ml_ready: bool = True
    confidence_score: float = 1.0
    ambiguity_gap: float = 1.0
    validation_issues: List[str] = field(default_factory=list)

    # Raw Data Reference (100% raw cell grid recoverable)
    raw_dataset_reference: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "workbook_name": self.workbook_name,
            "sheet_name": self.sheet_name,
            "region_id": self.region_id,
            "parent_region_id": self.parent_region_id,
            "scientific_type": self.scientific_type,
            "structural_source": self.structural_source,
            "axis_name": self.axis_name,
            "axis_unit": self.axis_unit,
            "target_name": self.target_name,
            "target_unit": self.target_unit,
            "global_parameters": [p.to_dict() for p in self.global_parameters],
            "x_values": self.x_values,
            "x_provenance": [p.to_dict() for p in self.x_provenance],
            "traces": [t.to_dict() for t in self.traces],
            "scalar_records": [s.to_dict() for s in self.scalar_records],
            "validation_status": self.validation_status,
            "acceptance_state": self.acceptance_state,
            "is_ml_ready": self.is_ml_ready,
            "confidence_score": self.confidence_score,
            "ambiguity_gap": self.ambiguity_gap,
            "validation_issues": self.validation_issues,
            "raw_dataset_reference": self.raw_dataset_reference,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScientificDataset:
        glob_params = [CanonicalParameter.from_dict(p) for p in data.get("global_parameters", [])]
        x_prov = [CellRefDTO.from_dict(p) for p in data.get("x_provenance", [])]
        traces = [CanonicalTrace.from_dict(t) for t in data.get("traces", [])]
        scalars = [CanonicalScalarRecord.from_dict(s) for s in data.get("scalar_records", [])]
        return cls(
            dataset_id=data.get("dataset_id", ""),
            workbook_name=data.get("workbook_name", ""),
            sheet_name=data.get("sheet_name", ""),
            region_id=data.get("region_id", ""),
            parent_region_id=data.get("parent_region_id"),
            scientific_type=data.get("scientific_type", "UNKNOWN"),
            structural_source=data.get("structural_source", "NumericCurveGrid"),
            axis_name=data.get("axis_name"),
            axis_unit=data.get("axis_unit"),
            target_name=data.get("target_name"),
            target_unit=data.get("target_unit"),
            global_parameters=glob_params,
            x_values=[float(v) for v in data.get("x_values", [])],
            x_provenance=x_prov,
            traces=traces,
            scalar_records=scalars,
            validation_status=data.get("validation_status", "ACCEPTED_VALID"),
            acceptance_state=data.get("acceptance_state", "ACCEPTED"),
            is_ml_ready=bool(data.get("is_ml_ready", True)),
            confidence_score=float(data.get("confidence_score", 1.0)),
            ambiguity_gap=float(data.get("ambiguity_gap", 1.0)),
            validation_issues=data.get("validation_issues", []),
            raw_dataset_reference=data.get("raw_dataset_reference"),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ScientificDataset:
        return cls.from_dict(json.loads(json_str))

    def to_csv(self) -> str:
        """Export dataset to tabular CSV string suitable for downstream ML ingestion."""
        output = io.StringIO()
        writer = csv.writer(output)

        if self.scalar_records:
            writer.writerow([
                "dataset_id",
                "scientific_type",
                "parameter_name",
                "parameter_value",
                "parameter_unit",
                "target_name",
                "target_value",
                "target_unit",
            ])
            for rec in self.scalar_records:
                writer.writerow([
                    self.dataset_id,
                    self.scientific_type,
                    rec.parameter.name if rec.parameter else "",
                    rec.parameter.value if rec.parameter else "",
                    rec.parameter.unit if rec.parameter else "",
                    rec.target_name,
                    rec.target_value,
                    rec.target_unit or "",
                ])
        else:
            x_col = self.axis_name or "x"
            writer.writerow([
                "dataset_id",
                "scientific_type",
                "trace_id",
                "parameter_name",
                "parameter_value",
                "parameter_unit",
                x_col,
                "y_value",
            ])
            for trace in self.traces:
                param_name = trace.parameter.name if trace.parameter else ""
                param_val = trace.parameter.value if trace.parameter else ""
                param_unit = trace.parameter.unit if trace.parameter else ""
                for x_val, y_val in zip(self.x_values, trace.y_values):
                    writer.writerow([
                        self.dataset_id,
                        self.scientific_type,
                        trace.trace_id,
                        param_name,
                        param_val,
                        param_unit,
                        x_val,
                        y_val,
                    ])

        return output.getvalue()


@dataclass
class CanonicalWorkbook:
    """Container holding all canonical scientific datasets for a complete workbook."""
    workbook_name: str
    datasets: List[ScientificDataset] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workbook_name": self.workbook_name,
            "datasets_count": len(self.datasets),
            "datasets": [d.to_dict() for d in self.datasets],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanonicalWorkbook:
        datasets = [ScientificDataset.from_dict(d) for d in data.get("datasets", [])]
        return cls(
            workbook_name=data.get("workbook_name", ""),
            datasets=datasets,
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> CanonicalWorkbook:
        return cls.from_dict(json.loads(json_str))

    def to_csv_folder(self, folder_path: Union[str, Path]) -> List[Path]:
        """Write every ML-ready dataset to an individual CSV file in folder_path."""
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        created_files: List[Path] = []
        for ds in self.datasets:
            if not ds.is_ml_ready:
                continue
            filepath = folder / f"{ds.dataset_id}.csv"
            filepath.write_text(ds.to_csv(), encoding="utf-8")
            created_files.append(filepath)
        return created_files
