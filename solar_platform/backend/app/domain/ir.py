from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IndependentAxis(BaseModel):
    axis_id: str
    name: str  # E.g. "Voltage", "Wavelength"
    unit: str  # E.g. "V", "nm"
    is_regular_sampling: bool = True
    values: List[float] = Field(default_factory=list)


class SweptParameter(BaseModel):
    parameter_id: str
    name: str  # E.g. "Thickness", "Temperature", "Defect_Density"
    unit: str  # E.g. "nm", "K", "cm^-3"
    values: List[float] = Field(default_factory=list)


class SparseEntry(BaseModel):
    axis_indices: List[int]
    param_indices: List[int]
    value: float


class DependentVariable(BaseModel):
    target_id: str
    name: str  # E.g. "Current", "QE", "Responsivity", "Detectivity"
    unit: str  # E.g. "mA/cm^2", "%", "A/W", "Jones"
    is_sparse: bool = False
    data_shape: List[int] = Field(default_factory=list)  # E.g. [N_axis, N_param1]
    dense_matrix: Optional[List[List[float]]] = None
    sparse_entries: Optional[List[SparseEntry]] = None
    column_labels: List[str] = Field(default_factory=list)  # E.g. ["AM1.5G", "400 nm", "500 nm", "600 nm"]


class IntermediateRepresentationV2(BaseModel):
    ir_version: str = "v2.0"
    axes: List[IndependentAxis] = Field(default_factory=list)
    parameters: List[SweptParameter] = Field(default_factory=list)
    targets: List[DependentVariable] = Field(default_factory=list)
    summary_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    metadata_attributes: Dict[str, Any] = Field(default_factory=dict)
