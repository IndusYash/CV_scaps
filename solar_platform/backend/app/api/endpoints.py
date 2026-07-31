from __future__ import annotations

import uuid
from typing import Dict, List
from fastapi import APIRouter, File, HTTPException, UploadFile
import pandas as pd

from app.models.schemas import (
    CurvePredictRequest,
    CurvePredictResponse,
    CurvesDataResponse,
    PointPredictRequest,
    PointPredictResponse,
    SheetSummary,
    TrainRequest,
    TrainResponse,
    UploadResponse,
)
from app.pipeline import UnifiedParserPipeline
from app.services.ml_engine import (
    get_trained_session,
    predict_curve_comparison,
    predict_point,
    train_sheet_models,
)
from app.services.parser import detect_sheet_schema, parse_excel_workbook

router = APIRouter(prefix="/api")

# In-memory stores keyed by file_id
_WORKBOOK_STORE: Dict[str, Dict[str, pd.DataFrame]] = {}
_PARSED_DATA_STORE: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
_PIPELINE = UnifiedParserPipeline()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Uploads an Excel file, executes production-ready architectural pipeline, and returns sheet summaries."""
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an Excel (.xlsx or .xls) file.")

    try:
        contents = await file.read()
        file_id = str(uuid.uuid4())

        # Execute Production Architecture Pipeline
        audit_result, parsed_data = _PIPELINE.parse_workbook(file.filename, contents, file_id)
        _PARSED_DATA_STORE[file_id] = parsed_data

        # Fallback raw pandas sheets for legacy compatibility
        sheets_dict = parse_excel_workbook(contents)
        _WORKBOOK_STORE[file_id] = sheets_dict

        sheet_summaries: List[SheetSummary] = []
        for sheet_audit in audit_result.sheets:
            sheet_name = sheet_audit.sheet_name
            df_raw = sheets_dict.get(sheet_name, pd.DataFrame())
            schema_info = detect_sheet_schema(df_raw, sheet_name)
            sheet_summaries.append(SheetSummary(**schema_info))

        return UploadResponse(
            filename=file.filename,
            file_id=file_id,
            sheets=sheet_summaries,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {str(exc)}")


@router.post("/train", response_model=TrainResponse)
async def train_model(req: TrainRequest):
    """Trains Random Forest models for all curves in the specified sheet."""
    if req.file_id not in _WORKBOOK_STORE:
        raise HTTPException(status_code=404, detail="File ID not found. Please re-upload the dataset.")

    sheets = _WORKBOOK_STORE[req.file_id]
    if req.sheet_name not in sheets:
        raise HTTPException(status_code=404, detail=f"Sheet '{req.sheet_name}' not found in workbook.")

    try:
        df_raw = sheets[req.sheet_name]
        session = train_sheet_models(
            file_id=req.file_id,
            sheet_name=req.sheet_name,
            df_raw=df_raw,
            test_size=req.test_size,
            n_estimators=req.n_estimators,
            random_state=req.random_state,
        )

        available_curves = list(session.models.keys())
        parameter_name = session.param_col or "Parameter"
        parameter_values = session.param_values

        x_ranges = {}
        for curve_type, df in session.processed_data.items():
            x_col = "Voltage" if "Voltage" in df.columns else "Wavelength"
            if not df.empty:
                x_ranges[curve_type] = {
                    "min": float(df[x_col].min()),
                    "max": float(df[x_col].max()),
                }

        return TrainResponse(
            sheet_name=req.sheet_name,
            metrics=session.metrics,
            available_curve_types=available_curves,
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            x_ranges=x_ranges,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(exc)}")


@router.post("/curves", response_model=CurvesDataResponse)
async def get_curves(req: TrainRequest):
    """Returns long-format data records for all processed curves to render Plotly figures."""
    session = get_trained_session(req.file_id, req.sheet_name)
    if not session:
        raise HTTPException(status_code=400, detail="Sheet models not trained yet. Call /train first.")

    curves_output = {}
    for curve_type, df in session.processed_data.items():
        curves_output[curve_type] = df.to_dict(orient="records")

    return CurvesDataResponse(
        sheet_name=req.sheet_name,
        curves=curves_output,
    )


@router.post("/predict/point", response_model=PointPredictResponse)
async def predict_point_endpoint(req: PointPredictRequest):
    """Generates single point prediction using trained Random Forest model."""
    try:
        pred_val = predict_point(
            file_id=req.file_id,
            sheet_name=req.sheet_name,
            curve_type=req.curve_type,
            x_val=req.x_val,
            param_val=req.param_val,
        )
        return PointPredictResponse(
            curve_type=req.curve_type,
            predicted_value=pred_val,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Point prediction failed: {str(exc)}")


@router.post("/predict/curve", response_model=CurvePredictResponse)
async def predict_curve_endpoint(req: CurvePredictRequest):
    """Generates full curve prediction vs actual overlay data for Plotly."""
    try:
        act_x, act_y, prd_x, prd_y = predict_curve_comparison(
            file_id=req.file_id,
            sheet_name=req.sheet_name,
            curve_type=req.curve_type,
            param_val=req.param_val,
        )
        return CurvePredictResponse(
            curve_type=req.curve_type,
            param_val=req.param_val,
            actual_x=act_x,
            actual_y=act_y,
            pred_x=prd_x,
            pred_y=prd_y,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Curve comparison failed: {str(exc)}")
