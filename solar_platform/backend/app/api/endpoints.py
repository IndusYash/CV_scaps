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
    ExperimentPayload,
    ExperimentSummary,
    ParameterDefinition,
    PointPredictRequest,
    PointPredictResponse,
    SheetSummary,
    TracePayload,
    TrainRequest,
    TrainResponse,
    UploadResponse,
)
from app.pipeline import UnifiedParserPipeline
from app.services.experiment_grouper import ExperimentGrouperEngine, ScientificExperiment
from app.services.ml_engine import (
    get_parsed_experiments,
    get_trained_session,
    predict_curve_comparison,
    predict_point,
    register_parsed_experiments,
    train_sheet_models,
)
from app.services.parser import detect_sheet_schema, parse_excel_workbook

router = APIRouter(prefix="/api")

# In-memory stores keyed by file_id
_WORKBOOK_STORE: Dict[str, Dict[str, pd.DataFrame]] = {}
_PARSED_DATA_STORE: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
_PIPELINE = UnifiedParserPipeline()
_GROUPER = ExperimentGrouperEngine()

HUMAN_LABELS = {
    "JV": "Current-Voltage (J-V) Characteristic",
    "QE": "Quantum Efficiency (QE) Spectrum",
    "RESPONSIVITY": "Spectral Responsivity (A/W)",
    "DETECTIVITY": "Specific Detectivity (Jones)",
    "ABSORPTION": "Optical Absorption Coefficient (1/cm)",
    "PHOTON_ENERGY": "Photon Energy Coordinate (eV)",
    "SUMMARY_METRICS": "Device Performance Summary Metrics",
}


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Uploads an Excel or CSV file, executes production pipeline asynchronously, and returns sheet & experiment summaries."""
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
        register_parsed_experiments(file_id, audit_result)

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

            # Build Phase 2 Experiment Summaries
            exps = get_parsed_experiments(file_id, sheet_name)
            exp_summaries: List[ExperimentSummary] = []
            for exp in exps:
                df_exp = exp.combined_dataframe
                
                param_names = [
                    c for c in df_exp.columns
                    if c not in [exp.axis_name, exp.target_name, "Run", "Trace_ID", "Block_ID"]
                ]
                
                if param_names:
                    traces_cnt = len(df_exp.groupby(param_names))
                elif "Trace_ID" in df_exp.columns and len(df_exp["Trace_ID"].unique()) > 0:
                    traces_cnt = len(df_exp["Trace_ID"].unique())
                elif "Block_ID" in df_exp.columns:
                    traces_cnt = len(df_exp["Block_ID"].unique())
                else:
                    traces_cnt = 1

                human_lbl = HUMAN_LABELS.get(exp.target_name.upper(), f"{exp.target_name} ({exp.target_unit})")

                exp_summaries.append(
                    ExperimentSummary(
                        experiment_id=exp.experiment_id,
                        sheet_name=exp.sheet_name,
                        semantic_type=exp.target_name,
                        human_label=human_lbl,
                        axis_name=exp.axis_name,
                        axis_unit=exp.axis_unit,
                        target_name=exp.target_name,
                        target_unit=exp.target_unit,
                        parameter_names=param_names,
                        trace_count=traces_cnt,
                        source_block_count=len(exp.blocks),
                    )
                )

            schema_info["blocks"] = block_summaries
            schema_info["experiments"] = exp_summaries
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
    """Trains Random Forest models for all ScientificExperiments on the specified sheet."""
    if req.file_id not in _WORKBOOK_STORE:
        raise HTTPException(status_code=404, detail="File ID not found in active session. Please re-upload your dataset.")

    sheets = _WORKBOOK_STORE[req.file_id]

    matched_sheet_name = None
    if req.sheet_name in sheets:
        matched_sheet_name = req.sheet_name
    else:
        s_target = req.sheet_name.strip().lower()
        for s_key in sheets.keys():
            if s_key.strip().lower() == s_target:
                matched_sheet_name = s_key
                break

    if not matched_sheet_name:
        raise HTTPException(status_code=404, detail=f"Sheet '{req.sheet_name}' not found in workbook. Available sheets: {list(sheets.keys())}")

    try:
        df_raw = sheets[matched_sheet_name]
        parsed_sheet_data = _PARSED_DATA_STORE.get(req.file_id, {}).get(matched_sheet_name)

        trained_sessions = train_sheet_models(
            file_id=req.file_id,
            sheet_name=matched_sheet_name,
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
        for exp_key, sess in trained_sessions.items():
            for ctype, mpair in sess.metrics.items():
                aggregated_metrics[exp_key] = mpair
                df = sess.processed_data.get(ctype, pd.DataFrame())
                x_col = sess.experiment.axis_name if sess.experiment else ("Voltage" if "Voltage" in df.columns else "Wavelength")
                if not df.empty and x_col in df.columns:
                    x_ranges[exp_key] = {
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
    """Returns scientific experiment payloads containing clean trace-oriented structures for Plotly visualization."""
    exps = get_parsed_experiments(req.file_id, req.sheet_name)
    exp_payloads: List[ExperimentPayload] = []

    for exp in exps:
        df_exp = exp.combined_dataframe

        # Skip experiments that have no data
        if df_exp is None or df_exp.empty:
            continue

        x_col = exp.axis_name
        target_col = exp.target_name

        # Guard: x_col and target_col must actually exist in the dataframe
        if x_col not in df_exp.columns or target_col not in df_exp.columns:
            continue

        param_cols = [
            c for c in df_exp.columns
            if c not in [x_col, target_col, "Run", "Trace_ID", "Block_ID"]
        ]

        param_defs: List[ParameterDefinition] = []
        for p_col in param_cols:
            unique_vals = sorted([float(v) for v in df_exp[p_col].unique() if pd.notna(v)])
            p_unit = ""
            for b in exp.blocks:
                ir = b.intermediate_representation
                if ir:
                    for p in ir.parameters:
                        if p.name == p_col:
                            p_unit = p.unit
                            break
            param_defs.append(
                ParameterDefinition(name=p_col, unit=p_unit, values=unique_vals)
            )

        traces: List[TracePayload] = []
        trace_counter = 0

        if param_cols:
            group_cols = param_cols
        elif "Trace_ID" in df_exp.columns and len(df_exp["Trace_ID"].unique()) > 0:
            group_cols = ["Trace_ID"]
        elif "Block_ID" in df_exp.columns:
            group_cols = ["Block_ID"]
        else:
            group_cols = []

        if group_cols:
            grouped = df_exp.groupby(group_cols, sort=False)
        else:
            grouped = [(("default",), df_exp)]

        for grp_keys, grp_df in grouped:
            # Skip empty groups
            if grp_df.empty:
                continue
            # Guard: both axis and target columns must exist in this group slice
            if x_col not in grp_df.columns or target_col not in grp_df.columns:
                continue

            trace_counter += 1
            grp_sorted = grp_df.sort_values(x_col)
            
            x_arr = [float(v) for v in grp_sorted[x_col].tolist() if pd.notna(v)]
            y_arr = [float(v) for v in grp_sorted[target_col].tolist() if pd.notna(v)]
            
            p_dict: Dict[str, float] = {}
            for p_col in param_cols:
                p_val = grp_sorted[p_col].iloc[0]
                if pd.notna(p_val):
                    p_dict[p_col] = float(p_val)

            if param_cols and p_dict:
                lbl_parts = [f"{k}: {v}" for k, v in p_dict.items()]
                lbl_str = ", ".join(lbl_parts)
            elif "Trace_ID" in grp_sorted.columns and pd.notna(grp_sorted["Trace_ID"].iloc[0]):
                lbl_str = str(grp_sorted["Trace_ID"].iloc[0])
            else:
                lbl_str = f"{exp.target_name}"

            source_b_id = str(grp_sorted["Block_ID"].iloc[0]) if "Block_ID" in grp_sorted.columns else f"block_{exp.experiment_id}"

            traces.append(
                TracePayload(
                    trace_id=f"trace_{trace_counter:03d}",
                    label=lbl_str,
                    parameters=p_dict,
                    x=x_arr,
                    y=y_arr,
                    source_block_id=source_b_id,
                )
            )

        human_lbl = HUMAN_LABELS.get(exp.target_name.upper(), f"{exp.target_name} ({exp.target_unit})")

        exp_payloads.append(
            ExperimentPayload(
                experiment_id=exp.experiment_id,
                sheet_name=exp.sheet_name,
                semantic_type=exp.target_name,
                human_label=human_lbl,
                axis={"name": exp.axis_name, "unit": exp.axis_unit},
                target={"name": exp.target_name, "unit": exp.target_unit},
                parameters=param_defs,
                traces=traces,
                source_block_ids=[b.block_id for b in exp.blocks],
            )
        )

    # Fallback legacy dictionary for backwards compatibility
    parsed_sheet_data = _PARSED_DATA_STORE.get(req.file_id, {}).get(req.sheet_name, {})
    curves_output = {}
    for block_key, df in parsed_sheet_data.items():
        if isinstance(df, pd.DataFrame):
            curves_output[block_key] = df.to_dict(orient="records")

    return CurvesDataResponse(
        sheet_name=req.sheet_name,
        curves=curves_output,
        experiments=exp_payloads,
    )


@router.post("/predict/point", response_model=PointPredictResponse)
async def predict_point_endpoint(req: PointPredictRequest):
    """Generates single point prediction using trained Random Forest model."""
    try:
        pred_val = predict_point(
            file_id=req.file_id,
            sheet_name=req.sheet_name,
            experiment_id=req.experiment_id,
            target_name=req.target_name,
            curve_type=req.curve_type,
            block_id=req.block_id,
            x_val=req.x_val,
            param_val=req.param_val,
            param_vals=req.param_vals,
        )
        return PointPredictResponse(
            curve_type=req.experiment_id or req.target_name or req.curve_type or "Prediction",
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
            experiment_id=req.experiment_id,
            target_name=req.target_name,
            curve_type=req.curve_type,
            block_id=req.block_id,
            param_val=req.param_val or 1.0,
            param_vals=req.param_vals,
        )
        return CurvePredictResponse(
            curve_type=req.experiment_id or req.target_name or req.curve_type or "Comparison",
            param_val=req.param_val or 1.0,
            actual_x=act_x,
            actual_y=act_y,
            pred_x=prd_x,
            pred_y=prd_y,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Curve comparison failed: {str(exc)}")
