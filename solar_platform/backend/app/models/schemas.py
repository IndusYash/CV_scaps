from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BlockSummary(BaseModel):
    block_id: str
    category: str
    semantic_label: str
    human_label: str
    axis_name: str
    axis_unit: str
    target_name: str
    target_unit: str
    parameters: List[str]
    row_count: int
    confidence: float
    scientific_validity: float
    warnings: List[str]
    summary_metrics: Optional[Dict[str, Dict[str, float]]] = None


class SheetSummary(BaseModel):
    sheet_name: str
    row_count: int
    col_count: int
    columns: List[str]
    detected_parameter_name: str
    detected_parameter_unit: str
    detected_x_axis: str
    curve_types_found: List[str]
    is_single_curve_mode: bool = False
    blocks: List[BlockSummary] = []


class UploadResponse(BaseModel):
    filename: str
    file_id: str
    sheets: List[SheetSummary]


class ColumnMapping(BaseModel):
    x_axis_col: str
    param_regex: str = r"(\d+(?:\.\d+)?)"
    param_name: str = "Parameter"
    param_unit: str = ""
    is_single_curve_mode: bool = False


class TrainRequest(BaseModel):
    file_id: str
    sheet_name: str
    block_id: Optional[str] = None
    mapping: Optional[ColumnMapping] = None
    test_size: float = 0.2
    n_estimators: int = 200
    random_state: int = 42


class MetricPair(BaseModel):
    r2: float
    mae: float


class TrainResponse(BaseModel):
    sheet_name: str
    metrics: Dict[str, MetricPair]
    available_curve_types: List[str]
    parameter_name: str
    parameter_values: List[float]
    x_ranges: Dict[str, Dict[str, float]]


class PointPredictRequest(BaseModel):
    file_id: str
    sheet_name: str
    curve_type: str
    block_id: Optional[str] = None
    x_val: float
    param_val: Optional[float] = None
    param_vals: Optional[Dict[str, float]] = None


class PointPredictResponse(BaseModel):
    curve_type: str
    predicted_value: float


class CurvePredictRequest(BaseModel):
    file_id: str
    sheet_name: str
    curve_type: str
    block_id: Optional[str] = None
    param_val: Optional[float] = None
    param_vals: Optional[Dict[str, float]] = None


class CurvePredictResponse(BaseModel):
    curve_type: str
    param_val: float
    actual_x: List[float]
    actual_y: List[float]
    pred_x: List[float]
    pred_y: List[float]


class CurvesDataResponse(BaseModel):
    sheet_name: str
    curves: Dict[str, List[Dict[str, Any]]]
