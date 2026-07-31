from __future__ import annotations

import uuid
from typing import Dict, List
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
import pandas as pd

from app.exceptions import (
    FileCorruptedException,
    SCAPSParserBaseException,
    UnsupportedFormatException,
)
from app.models.schemas import (
    BlockSummary,
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

HUMAN_LABELS = {
    "JV": "Current-Voltage (J-V) Characteristic",
    "QE": "Quantum Efficiency (QE) Spectrum",
    "RESPONSIVITY": "Spectral Responsivity (A/W)",
    "DETECTIVITY": "Specific Detectivity (Jones)",
    "ABSORPTION": "Optical Absorption Coefficient (1/cm)",
    "SUMMARY_METRICS": "Device Performance Summary Metrics",
}


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Uploads an Excel or CSV file, executes production pipeline asynchronously, and returns sheet & block summaries."""
    filename_lower = file.filename.lower()
    if not filename_lower.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=400, detail="Invalid file format. Supported extensions: .xlsx, .xls, .csv")

    try:
        contents = await file.read()
        file_id = str(uuid.uuid4())

        # Offload CPU-bound parsing pipeline to FastAPI threadpool
        audit_result, parsed_data = await run_in_threadpool(
            _PIPELINE.parse_workbook, file.filename, contents, file_id
        )
        _PARSED_DATA_STORE[file_id] = parsed_data

        # Fallback raw pandas sheets for legacy compatibility
        if filename_lower.endswith((".xlsx", ".xls")):
            sheets_dict = parse_excel_workbook(contents)
        else:
            sheets_dict = {"CSV_Data": pd.read_csv(pd.io.common.BytesIO(contents))}
        _WORKBOOK_STORE[file_id] = sheets_dict

        sheet_summaries: List[SheetSummary] = []
        for sheet_audit in audit_result.sheets:
            sheet_name = sheet_audit.sheet_name
            df_raw = sheets_dict.get(sheet_name, pd.DataFrame())
            schema_info = detect_sheet_schema(df_raw, sheet_name)

            block_summaries: List[BlockSummary] = []
            for b in sheet_audit.blocks:
                ir = b.intermediate_representation
                cat_str = str(b.category.value if hasattr(b.category, "value") else b.category)

                axis_name = ir.axes[0].name if (ir and ir.axes) else "N/A"
                axis_unit = ir.axes[0].unit if (ir and ir.axes) else "N/A"
                target_name = ir.targets[0].name if (ir and ir.targets) else ("Summary Metrics" if ir and ir.summary_metrics else "N/A")
                target_unit = ir.targets[0].unit if (ir and ir.targets) else ("Various" if ir and ir.summary_metrics else "N/A")
                params = [p.name for p in ir.parameters] if (ir and ir.parameters) else []

                r_count = len(ir.axes[0].values) if (ir and ir.axes and ir.axes[0].values) else (len(df_raw) if not df_raw.empty else 0)

                human_lbl = HUMAN_LABELS.get(b.semantic_label.upper(), b.semantic_label)

                block_summaries.append(
                    BlockSummary(
                        block_id=b.block_id,
                        category=cat_str,
                        semantic_label=b.semantic_label,
                        human_label=human_lbl,
                        axis_name=axis_name,
                        axis_unit=axis_unit,
                        target_name=target_name,
                        target_unit=target_unit,
                        parameters=params,
                        row_count=r_count,
                        confidence=b.parsing_confidence,
                        scientific_validity=b.scientific_validity,
                        warnings=b.validation_warnings or [],
                        summary_metrics=ir.summary_metrics if (ir and ir.summary_metrics) else None,
                    )
                )

            schema_info["blocks"] = block_summaries
            sheet_summaries.append(SheetSummary(**schema_info))

        return UploadResponse(
            filename=file.filename,
            file_id=file_id,
            sheets=sheet_summaries,
        )
    except UnsupportedFormatException as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except SCAPSParserBaseException as exc:
        raise HTTPException(status_code=422, detail=f"Parser Error: {str(exc)}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {str(exc)}")


@router.post("/train", response_model=TrainResponse)
async def train_model(req: TrainRequest):
    """Trains Random Forest models for all curves/blocks in the specified sheet using multi-parameter features."""
    if req.file_id not in _WORKBOOK_STORE:
        raise HTTPException(status_code=404, detail="File ID not found. Please re-upload the dataset.")

    sheets = _WORKBOOK_STORE[req.file_id]
    if req.sheet_name not in sheets:
        raise HTTPException(status_code=404, detail=f"Sheet '{req.sheet_name}' not found in workbook.")

    try:
        df_raw = sheets[req.sheet_name]
        parsed_sheet_data = _PARSED_DATA_STORE.get(req.file_id, {}).get(req.sheet_name)

        trained_sessions = train_sheet_models(
            file_id=req.file_id,
            sheet_name=req.sheet_name,
            df_raw=df_raw,
            parsed_sheet_data=parsed_sheet_data,
            test_size=req.test_size,
            n_estimators=req.n_estimators,
            random_state=req.random_state,
        )

        available_curves = list(trained_sessions.keys())
        first_session = list(trained_sessions.values())[0] if trained_sessions else None
        parameter_name = first_session.param_col if first_session and first_session.param_col else "Parameter"
        parameter_values = first_session.param_values if first_session else []

        aggregated_metrics = {}
        x_ranges = {}
        for bkey, sess in trained_sessions.items():
            for ctype, mpair in sess.metrics.items():
                aggregated_metrics[bkey] = mpair
                df = sess.processed_data.get(ctype, pd.DataFrame())
                x_col = "Voltage" if "Voltage" in df.columns else "Wavelength"
                if not df.empty and x_col in df.columns:
                    x_ranges[bkey] = {
                        "min": float(df[x_col].min()),
                        "max": float(df[x_col].max()),
                    }

        return TrainResponse(
            sheet_name=req.sheet_name,
            metrics=aggregated_metrics,
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
    parsed_sheet_data = _PARSED_DATA_STORE.get(req.file_id, {}).get(req.sheet_name, {})
    if not parsed_sheet_data:
        session = get_trained_session(req.file_id, req.sheet_name)
        if session:
            parsed_sheet_data = session.processed_data
        else:
            parsed_sheet_data = {}

    curves_output = {}
    for block_key, df in parsed_sheet_data.items():
        if isinstance(df, pd.DataFrame):
            curves_output[block_key] = df.to_dict(orient="records")
        elif isinstance(df, list):
            curves_output[block_key] = df

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
            block_id=req.block_id,
            x_val=req.x_val,
            param_val=req.param_val,
            param_vals=req.param_vals,
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
            block_id=req.block_id,
            param_val=req.param_val or 1.0,
            param_vals=req.param_vals,
        )
        return CurvePredictResponse(
            curve_type=req.curve_type,
            param_val=req.param_val or 1.0,
            actual_x=act_x,
            actual_y=act_y,
            pred_x=prd_x,
            pred_y=prd_y,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Curve comparison failed: {str(exc)}")
