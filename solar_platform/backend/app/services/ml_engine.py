from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, LeaveOneOut

from app.models.schemas import MetricPair
from app.services.experiment_grouper import ExperimentGrouperEngine, ScientificExperiment
from app.services.preprocessor import preprocess_parametric_sheet, preprocess_single_curve_sheet

# Below this many rows, a single random train/test holdout split is not
# statistically meaningful: an 80/20 split can leave as few as 1 test
# point, for which R^2 is undefined (sklearn returns NaN), or 2 points,
# where a single unlucky held-out value can swing R^2 to wildly negative
# numbers that say nothing real about the model. Experiments with fewer
# rows than this use leave-one-out cross-validation instead (see
# _fit_and_evaluate below); experiments at or above it keep the existing
# random-split behavior unchanged.
_SMALL_SAMPLE_THRESHOLD = 20


def _fit_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,
    random_state: int,
    test_size: float,
) -> Tuple[RandomForestRegressor, float, float]:
    """Fits the production RandomForestRegressor on all available data and
    returns it alongside an R^2/MAE pair evaluated the way appropriate for
    the sample size.

    - len(X) < _SMALL_SAMPLE_THRESHOLD: leave-one-out cross-validation.
      Each row is held out once, predicted by a model trained on every
      other row, and R^2/MAE are computed over the full set of N
      out-of-fold predictions. This uses every point as a held-out test
      case exactly once, rather than gambling the whole evaluation on a
      single random 1-2 point split.
    - len(X) >= _SMALL_SAMPLE_THRESHOLD: unchanged -- a single random
      train/test split, exactly as before, since there is enough data
      for that split to be meaningful (this is the curve-data case,
      already verified to produce good R^2 scores).

    The returned model is always fit on the complete dataset, since for
    actual prediction use there is no reason to withhold data from the
    deployed model once evaluation is done.
    """
    n = len(X)

    if n < _SMALL_SAMPLE_THRESHOLD:
        oof_pred = np.empty(n, dtype=float)
        oof_true = y.to_numpy(dtype=float)
        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(X):
            fold_model = RandomForestRegressor(
                n_estimators=n_estimators, random_state=random_state
            )
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof_pred[test_idx[0]] = fold_model.predict(X.iloc[test_idx])[0]

        r2 = float(r2_score(oof_true, oof_pred))
        mae = float(mean_absolute_error(oof_true, oof_pred))

        final_model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
        final_model.fit(X, y)
        return final_model, r2, mae

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    return model, r2, mae


class TrainedSheetSession:
    def __init__(
        self,
        sheet_name: str,
        curve_type: str,
        block_id: str,
        processed_data: Dict[str, pd.DataFrame],
        models: Dict[str, RandomForestRegressor],
        metrics: Dict[str, MetricPair],
        is_single_curve: bool = False,
        experiment: Optional[ScientificExperiment] = None,
    ):
        self.sheet_name = sheet_name
        self.curve_type = curve_type
        self.block_id = block_id
        self.processed_data = processed_data
        self.models = models
        self.metrics = metrics
        self.is_single_curve = is_single_curve
        self.experiment = experiment

        self.param_cols: List[str] = []
        self.param_col = None
        self.param_values = []

        for df in processed_data.values():
            for col in df.columns:
                if col not in ["Voltage", "Wavelength", "Current", "QE", "Responsivity", "Detectivity", "Absorption", "Photon_Energy", "Run", "Trace_ID", "Block_ID"]:
                    if col not in self.param_cols:
                        self.param_cols.append(col)

        if self.param_cols:
            self.param_col = self.param_cols[0]
            first_df = list(processed_data.values())[0]
            self.param_values = sorted([float(v) for v in first_df[self.param_col].unique() if pd.notna(v)])


# Session store keyed by (file_id, sheet_name, experiment_id_or_curve_type)
_SESSION_STORE: Dict[Tuple[str, str, str], TrainedSheetSession] = {}
_EXPERIMENT_STORE: Dict[Tuple[str, str], List[ScientificExperiment]] = {}


def register_parsed_experiments(file_id: str, audit_result: Any):
    """Registers Phase 2 ScientificExperiments into in-memory store."""
    grouper = ExperimentGrouperEngine()
    for sheet_audit in audit_result.sheets:
        exps = grouper.group_sheet_blocks_into_experiments(sheet_audit)
        _EXPERIMENT_STORE[(file_id, sheet_audit.sheet_name)] = exps


def get_parsed_experiments(file_id: str, sheet_name: str) -> List[ScientificExperiment]:
    """Retrieves parsed ScientificExperiments for a sheet."""
    return _EXPERIMENT_STORE.get((file_id, sheet_name), [])


def train_sheet_models(
    file_id: str,
    sheet_name: str,
    df_raw: pd.DataFrame,
    parsed_sheet_data: Optional[Dict[str, pd.DataFrame]] = None,
    test_size: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42,
) -> Dict[str, TrainedSheetSession]:
    """Trains RandomForestRegressor models for all ScientificExperiments on a sheet."""
    exps = get_parsed_experiments(file_id, sheet_name)
    trained_sessions: Dict[str, TrainedSheetSession] = {}

    if exps:
        for exp in exps:
            df = exp.combined_dataframe
            if df.empty or len(df) < 5:
                continue

            x_col = exp.axis_name
            target_col = exp.target_name

            param_cols = [
                c for c in df.columns
                if c not in [x_col, target_col, "Run", "Trace_ID", "Block_ID"]
            ]
            feature_cols = [x_col] + param_cols

            X = df[feature_cols]
            y = df[target_col]

            if len(X) < 5:
                continue

            model, r2, mae = _fit_and_evaluate(
                X, y, n_estimators, random_state, test_size
            )

            exp_key = exp.experiment_id
            models = {exp_key: model}
            metrics = {exp_key: MetricPair(r2=r2, mae=mae)}

            session = TrainedSheetSession(
                sheet_name=sheet_name,
                curve_type=exp_key,
                block_id=exp_key,
                processed_data={exp_key: df},
                models=models,
                metrics=metrics,
                is_single_curve=False,
                experiment=exp,
            )

            _SESSION_STORE[(file_id, sheet_name, exp_key)] = session
            _SESSION_STORE[(file_id, sheet_name, target_col)] = session
            trained_sessions[exp_key] = session
    else:
        # Fallback raw pandas sheets for legacy compatibility
        processed_data = parsed_sheet_data or {}
        for key, df in processed_data.items():
            if df.empty or len(df) < 5 or key.lower() in ["summary_metrics", "summary"]:
                continue

            x_col = "Voltage" if "Voltage" in df.columns else "Wavelength"
            if x_col not in df.columns:
                continue

            param_cols = [
                c for c in df.columns
                if c not in ["Voltage", "Wavelength", "Current", "QE", "Responsivity", "Detectivity", "Absorption", "Photon_Energy", "Run", "Trace_ID", "Block_ID"]
            ]
            feature_cols = [x_col] + param_cols
            target_cols = [c for c in df.columns if c not in feature_cols and c not in ["Run", "Trace_ID", "Block_ID"]]
            if not target_cols:
                continue
            target_col = target_cols[0]

            X = df[feature_cols]
            y = df[target_col]

            if len(X) < 5:
                continue

            model, r2, mae = _fit_and_evaluate(
                X, y, n_estimators, random_state, test_size
            )

            models = {key: model}
            metrics = {key: MetricPair(r2=r2, mae=mae)}

            session = TrainedSheetSession(
                sheet_name=sheet_name,
                curve_type=key,
                block_id=key,
                processed_data={key: df},
                models=models,
                metrics=metrics,
                is_single_curve=False,
            )

            _SESSION_STORE[(file_id, sheet_name, key)] = session
            trained_sessions[key] = session

    return trained_sessions


def get_trained_session(
    file_id: str,
    sheet_name: str,
    experiment_id: Optional[str] = None,
    target_name: Optional[str] = None,
    curve_type: Optional[str] = None,
    block_id: Optional[str] = None,
) -> Optional[TrainedSheetSession]:
    """Retrieves trained session matching exact experiment_id first, then fallback keys."""
    for key in [experiment_id, target_name, block_id, curve_type]:
        if key and (file_id, sheet_name, key) in _SESSION_STORE:
            return _SESSION_STORE[(file_id, sheet_name, key)]

    # Fallback search matching file_id and sheet_name
    for (fid, sname, bkey), session in _SESSION_STORE.items():
        if fid == file_id and sname == sheet_name:
            return session
    return None


def predict_point(
    file_id: str,
    sheet_name: str,
    experiment_id: Optional[str] = None,
    target_name: Optional[str] = None,
    curve_type: Optional[str] = None,
    block_id: Optional[str] = None,
    x_val: float = 0.0,
    param_val: Optional[float] = None,
    param_vals: Optional[Dict[str, float]] = None,
) -> float:
    session = get_trained_session(file_id, sheet_name, experiment_id, target_name, curve_type, block_id)
    if not session:
        raise ValueError(f"No trained model found for sheet '{sheet_name}' and experiment '{experiment_id or target_name or curve_type or block_id}'")

    model_key = list(session.models.keys())[0]
    model = session.models[model_key]
    df_proc = session.processed_data[model_key]

    feature_names = getattr(model, "feature_names_in_", [c for c in df_proc.columns if c not in ["Run", "Trace_ID", "Block_ID"]])
    x_col = feature_names[0]

    input_dict = {x_col: [x_val]}

    for p_col in feature_names[1:]:
        if param_vals and p_col in param_vals:
            input_dict[p_col] = [param_vals[p_col]]
        elif param_val is not None:
            input_dict[p_col] = [param_val]
        else:
            input_dict[p_col] = [float(df_proc[p_col].mean()) if p_col in df_proc.columns else 1.0]

    input_df = pd.DataFrame(input_dict)[feature_names]
    pred = model.predict(input_df)[0]
    return float(pred)


def predict_curve_comparison(
    file_id: str,
    sheet_name: str,
    experiment_id: Optional[str] = None,
    target_name: Optional[str] = None,
    curve_type: Optional[str] = None,
    block_id: Optional[str] = None,
    param_val: float = 1.0,
    param_vals: Optional[Dict[str, float]] = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    session = get_trained_session(file_id, sheet_name, experiment_id, target_name, curve_type, block_id)
    if not session:
        raise ValueError(f"No trained model found for sheet '{sheet_name}' and experiment '{experiment_id or target_name or curve_type or block_id}'")

    model_key = list(session.models.keys())[0]
    model = session.models[model_key]
    df_proc = session.processed_data[model_key]

    feature_names = getattr(model, "feature_names_in_", [c for c in df_proc.columns if c not in ["Run", "Trace_ID", "Block_ID"]])
    x_col = feature_names[0]

    target_cols = [c for c in df_proc.columns if c not in feature_names and c not in ["Run", "Trace_ID", "Block_ID"]]
    if not target_cols:
        target_cols = [df_proc.columns[-1]]
    target_col = target_cols[0]

    # Filter actual dataframe by exact parameter conditions
    actual_df = df_proc.copy()
    if param_vals:
        for p_name, p_val in param_vals.items():
            if p_name in actual_df.columns:
                filtered = actual_df[actual_df[p_name] == p_val]
                if not filtered.empty:
                    actual_df = filtered

    if actual_df.empty or len(actual_df) == len(df_proc):
        # Pick the first trace group if available
        if "Trace_ID" in df_proc.columns:
            first_trace = df_proc["Trace_ID"].iloc[0]
            actual_df = df_proc[df_proc["Trace_ID"] == first_trace]
        else:
            actual_df = df_proc.iloc[:61]

    actual_df = actual_df.sort_values(x_col)
    actual_x = actual_df[x_col].tolist()
    actual_y = actual_df[target_col].tolist()

    input_dict = {x_col: actual_x}
    for p_col in feature_names[1:]:
        if param_vals and p_col in param_vals:
            input_dict[p_col] = [param_vals[p_col]] * len(actual_x)
        elif p_col in actual_df.columns:
            input_dict[p_col] = [float(actual_df[p_col].iloc[0])] * len(actual_x)
        else:
            input_dict[p_col] = [float(df_proc[p_col].mean()) if p_col in df_proc.columns else 1.0] * len(actual_x)

    input_df = pd.DataFrame(input_dict)[feature_names]
    pred_y = model.predict(input_df).tolist()

    return actual_x, actual_y, actual_x, pred_y