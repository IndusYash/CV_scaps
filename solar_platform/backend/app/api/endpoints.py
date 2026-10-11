from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
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

# Global cache for classical RandomForest regressor results
_GLOBAL_RF_METRICS: Dict[str, Any] = {}
_GLOBAL_RF_PREDICTIONS: Dict[str, Any] = {}

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

        # Persist to SQLite persistent DB and extract PV parameters automatically
        from app.database import db_manager
        await run_in_threadpool(
            db_manager.parse_and_persist_workbook_bytes, file.filename, contents
        )

        # Additive: store the corrected summary-sweep rows for the opt-in
        # Layer B data source (data_source="corrected"). Writes only to its own
        # table and must never be able to affect the upload itself.
        try:
            from app.services import corrected_pv_dataset
            await run_in_threadpool(
                corrected_pv_dataset.persist_corrected_rows, file.filename, audit_result
            )
        except Exception as exc:  # pragma: no cover - defensive
            import logging
            logging.getLogger(__name__).warning(
                "Corrected PV rows were not persisted for %s: %s", file.filename, exc
            )

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
                        semantic_type=exp.semantic_type,
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
                semantic_type=exp.semantic_type,
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


# --- V1 Persistent Database & Classical ML Endpoints ---

@router.get("/database/status")
def get_db_status():
    """Returns total record counts in SQLite persistent database."""
    try:
        from app.database import db_manager
        conn = db_manager.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM jv_derived_metrics;")
        metrics_count = cursor.fetchone()["count"]
        conn.close()

        return {
            "datasets_count": db_manager.count_datasets(),
            "derived_metrics_count": metrics_count,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database check failed: {str(exc)}")


@router.get("/datasets")
def list_datasets_endpoint(scientific_type: Optional[str] = None):
    """Lists all canonical datasets currently stored in SQLite DB."""
    try:
        from app.database import db_manager
        datasets = db_manager.list_datasets(scientific_type)
        return [ds.to_dict() for ds in datasets]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(exc)}")


@router.get("/datasets/{dataset_id}")
def get_dataset_endpoint(dataset_id: str):
    """Retrieves a single canonical dataset from SQLite DB."""
    try:
        from app.database import db_manager
        ds = db_manager.get_dataset(dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Include any derived metrics for this dataset
        conn = db_manager.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jv_derived_metrics WHERE dataset_id = ?;", (dataset_id,))
        metrics = [dict(r) for r in cursor.fetchall()]
        conn.close()

        res = ds.to_dict()
        res["derived_metrics"] = metrics
        return res
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dataset: {str(exc)}")


@router.post("/ml/train/random-forest")
async def train_global_random_forest(
    material: Optional[str] = None,
    validation_mode: str = "interpolation",
    test_workbook: Optional[str] = None,
    test_size: float = 0.25,
    random_state: int = 42,
    data_source: str = "db"
):
    """
    Curates a tabular dataset from database, trains RandomForestRegressors for Jsc, Voc, FF, PCE,
    compares them to mean and linear baselines, and returns performance metrics.
    """
    try:
        from app.services.rf_pv_regressor import build_ml_dataset_from_db, MLRegressorManager
        
        if data_source == "corrected":
            from app.services.corrected_pv_dataset import build_ml_dataset_from_corrected
            df, audit_log = build_ml_dataset_from_corrected()
        else:
            df, audit_log = build_ml_dataset_from_db()
        if df.empty:
            return {
                "status": "NO_DATA",
                "message": "SQLite database is currently empty. Please upload SCAPS Excel datasets first.",
                "audit": audit_log
            }

        manager = MLRegressorManager(df)
        targets = ["j_sc", "v_oc", "ff", "pce"]
        
        global _GLOBAL_RF_METRICS, _GLOBAL_RF_PREDICTIONS
        _GLOBAL_RF_METRICS.clear()
        _GLOBAL_RF_PREDICTIONS.clear()
        
        results = {}
        for target in targets:
            metrics_dict = manager.train_and_evaluate(
                target_name=target,
                material_filter=material,
                test_size=test_size,
                random_state=random_state,
                validation_mode=validation_mode,
                test_workbook=test_workbook
            )
            results[target] = metrics_dict

        _GLOBAL_RF_METRICS.update(results)
        _GLOBAL_RF_PREDICTIONS.update(manager.predictions)

        # Generate audit report on disk
        from pathlib import Path
        from app.services.rf_pv_regressor import generate_ml_audit_report
        report_name = "v1_ml_dataset_audit_corrected.md" if data_source == "corrected" else "v1_ml_dataset_audit.md"
        report_path = Path(__file__).resolve().parents[2] / report_name
        generate_ml_audit_report(audit_log, report_path)

        # Find first success target to extract top-level metadata
        success_target = next((t for t in targets if results[t].get("status") != "INSUFFICIENT_DATA"), None)
        first_res = results[success_target] if success_target else results[targets[0]]

        return {
            "status": "SUCCESS" if success_target else "INSUFFICIENT_DATA",
            "model": "RandomForestRegressor",
            "validation_mode": validation_mode,
            "data_source": data_source,
            "test_workbook": test_workbook,
            "train_workbooks": first_res.get("train_workbooks", []),
            "test_workbooks": first_res.get("test_workbooks", []),
            "is_statistically_meaningful": first_res.get("is_statistically_meaningful", False),
            "targets": {
                target: {
                    "actual": manager.predictions[target]["actuals"] if target in manager.predictions else [],
                    "predicted": manager.predictions[target]["predicted_rf"] if target in manager.predictions else [],
                    "metrics": {
                        "r2": results[target]["random_forest"]["r2"] if "random_forest" in results[target] else float("nan"),
                        "mae": results[target]["random_forest"]["mae"] if "random_forest" in results[target] else float("nan"),
                        "rmse": results[target]["random_forest"]["rmse"] if "random_forest" in results[target] else float("nan"),
                        "n_samples": len(manager.predictions[target]["actuals"]) if target in manager.predictions else 0
                    },
                    "validation": results[target]
                } for target in targets
            },
            "metrics": results,
            "audit_log_summary": {
                "total_scanned": len(audit_log),
                "eligible": sum(1 for r in audit_log if r["status"] == "ELIGIBLE"),
                "excluded": sum(1 for r in audit_log if r["status"] == "EXCLUDED"),
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Global RF training failed: {str(exc)}")


@router.get("/ml/results")
def get_global_ml_results():
    """Returns parity plot actual vs predicted curves and comparison baselines."""
    if not _GLOBAL_RF_METRICS or not _GLOBAL_RF_PREDICTIONS:
        raise HTTPException(
            status_code=400,
            detail="No Random Forest models trained yet. Please POST to /api/ml/train/random-forest first."
        )
    targets = ["j_sc", "v_oc", "ff", "pce"]
    first_res = _GLOBAL_RF_METRICS.get(targets[0], {})
    return {
        "model": "RandomForestRegressor",
        "validation_mode": first_res.get("validation_mode", "interpolation"),
        "test_workbook": first_res.get("test_workbook", None),
        "train_workbooks": first_res.get("train_workbooks", []),
        "test_workbooks": first_res.get("test_workbooks", []),
        "is_statistically_meaningful": first_res.get("is_statistically_meaningful", False),
        "targets": {
            t: {
                "actual": _GLOBAL_RF_PREDICTIONS[t]["actuals"],
                "predicted": _GLOBAL_RF_PREDICTIONS[t]["predicted_rf"],
                "metrics": {
                    "r2": _GLOBAL_RF_METRICS[t]["random_forest"]["r2"] if "random_forest" in _GLOBAL_RF_METRICS[t] else float("nan"),
                    "mae": _GLOBAL_RF_METRICS[t]["random_forest"]["mae"] if "random_forest" in _GLOBAL_RF_METRICS[t] else float("nan"),
                    "rmse": _GLOBAL_RF_METRICS[t]["random_forest"]["rmse"] if "random_forest" in _GLOBAL_RF_METRICS[t] else float("nan"),
                    "n_samples": len(_GLOBAL_RF_PREDICTIONS[t]["actuals"])
                },
                "validation": _GLOBAL_RF_METRICS[t]
            } for t in targets
        },
        "metrics": _GLOBAL_RF_METRICS,
        "predictions": _GLOBAL_RF_PREDICTIONS
    }


# --- Corrected-pipeline data management (used by the V1 ML Reg. tab) ---

def _reset_global_rf_cache() -> None:
    """Drops the last training result so the page cannot show numbers for workbooks that were removed."""
    _GLOBAL_RF_METRICS.clear()
    _GLOBAL_RF_PREDICTIONS.clear()


@router.get("/ml/corrected/workbooks")
def list_corrected_workbooks():
    """Lists the workbooks stored for the corrected-pipeline data source."""
    try:
        from app.services import corrected_pv_dataset
        return {"workbooks": corrected_pv_dataset.list_stored_workbooks()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list corrected workbooks: {str(exc)}")


@router.delete("/ml/corrected/workbook")
def delete_corrected_workbook(name: str):
    """Removes one workbook from the corrected-pipeline data source (re-upload to restore it)."""
    try:
        from app.services import corrected_pv_dataset
        known = {w["workbook_name"] for w in corrected_pv_dataset.list_stored_workbooks()}
        if name not in known:
            raise HTTPException(status_code=404, detail=f"Workbook '{name}' is not stored in the corrected data.")
        removed = corrected_pv_dataset.delete_stored_workbooks([name])
        _reset_global_rf_cache()
        return {"status": "OK", "removed_rows": removed, "workbook_name": name}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to remove workbook: {str(exc)}")


@router.delete("/ml/corrected/workbooks")
def clear_corrected_workbooks():
    """Removes every workbook from the corrected-pipeline data source (re-upload to restore)."""
    try:
        from app.services import corrected_pv_dataset
        removed = corrected_pv_dataset.delete_stored_workbooks(None)
        _reset_global_rf_cache()
        return {"status": "OK", "removed_rows": removed}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to clear corrected data: {str(exc)}")