from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from app.models.schemas import MetricPair
from app.services.preprocessor import preprocess_parametric_sheet, preprocess_single_curve_sheet


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
    ):
        self.sheet_name = sheet_name
        self.curve_type = curve_type
        self.block_id = block_id
        self.processed_data = processed_data
        self.models = models
        self.metrics = metrics
        self.is_single_curve = is_single_curve

        self.param_cols: List[str] = []
        self.param_col = None
        self.param_values = []

        for df in processed_data.values():
            for col in df.columns:
                if col not in ["Voltage", "Wavelength", "Current", "QE", "Responsivity", "Detectivity", "Absorption", "Run"]:
                    if col not in self.param_cols:
                        self.param_cols.append(col)

        if self.param_cols:
            self.param_col = self.param_cols[0]
            first_df = list(processed_data.values())[0]
            self.param_values = sorted([float(v) for v in first_df[self.param_col].unique() if pd.notna(v)])


# Multi-block Model Cache Store keyed by (file_id, sheet_name, block_id_or_curve_type)
_SESSION_STORE: Dict[Tuple[str, str, str], TrainedSheetSession] = {}


def train_sheet_models(
    file_id: str,
    sheet_name: str,
    df_raw: pd.DataFrame,
    parsed_sheet_data: Optional[Dict[str, pd.DataFrame]] = None,
    test_size: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42,
) -> Dict[str, TrainedSheetSession]:
    """Trains RandomForestRegressor models for all available blocks/curves on a sheet, preserving each block independently."""
    if parsed_sheet_data:
        processed_data = parsed_sheet_data
    else:
        col_str_lower = " ".join([str(c).lower() for c in df_raw.columns])
        is_single_curve = "jtot" in col_str_lower or "d*" in col_str_lower
        if is_single_curve:
            processed_data = preprocess_single_curve_sheet(df_raw)
        else:
            processed_data = preprocess_parametric_sheet(df_raw, sheet_name)

    trained_sessions: Dict[str, TrainedSheetSession] = {}

    for key, df in processed_data.items():
        if df.empty or len(df) < 5 or key.lower() in ["summary_metrics", "summary"]:
            continue

        x_col = "Voltage" if "Voltage" in df.columns else "Wavelength"
        if x_col not in df.columns:
            continue

        # Extract semantic_label and block_id from DataFrame attributes or key
        block_id = df.attrs.get("block_id", key)
        curve_type = df.attrs.get("semantic_label", key).lower()

        # Exclude 'Run' and dependent target variables from feature matrix X
        param_cols = [
            c for c in df.columns
            if c not in ["Voltage", "Wavelength", "Current", "QE", "Responsivity", "Detectivity", "Absorption", "Run"]
        ]
        feature_cols = [x_col] + param_cols

        target_cols = [c for c in df.columns if c not in feature_cols and c != "Run"]
        if not target_cols:
            continue
        target_col = target_cols[0]

        X = df[feature_cols]
        y = df[target_col]

        if len(X) < 5:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))

        models = {curve_type: model}
        metrics = {curve_type: MetricPair(r2=r2, mae=mae)}

        session = TrainedSheetSession(
            sheet_name=sheet_name,
            curve_type=curve_type,
            block_id=block_id,
            processed_data={curve_type: df},
            models=models,
            metrics=metrics,
            is_single_curve=False,
        )

        # Store model keyed by BOTH block_id AND curve_type to prevent overwriting
        _SESSION_STORE[(file_id, sheet_name, block_id)] = session
        _SESSION_STORE[(file_id, sheet_name, curve_type)] = session
        trained_sessions[block_id] = session

    return trained_sessions


def get_trained_session(
    file_id: str,
    sheet_name: str,
    curve_type: Optional[str] = None,
    block_id: Optional[str] = None,
) -> Optional[TrainedSheetSession]:
    """Retrieves trained session matching exact block_id first, then curve_type fallback."""
    if block_id and (file_id, sheet_name, block_id) in _SESSION_STORE:
        return _SESSION_STORE[(file_id, sheet_name, block_id)]
    if curve_type and (file_id, sheet_name, curve_type) in _SESSION_STORE:
        return _SESSION_STORE[(file_id, sheet_name, curve_type)]
    if curve_type and (file_id, sheet_name, curve_type.lower()) in _SESSION_STORE:
        return _SESSION_STORE[(file_id, sheet_name, curve_type.lower())]
    
    # Fallback search matching file_id and sheet_name
    for (fid, sname, bkey), session in _SESSION_STORE.items():
        if fid == file_id and sname == sheet_name:
            return session
    return None


def predict_point(
    file_id: str,
    sheet_name: str,
    curve_type: str,
    block_id: Optional[str] = None,
    x_val: float = 0.0,
    param_val: Optional[float] = None,
    param_vals: Optional[Dict[str, float]] = None,
) -> float:
    session = get_trained_session(file_id, sheet_name, curve_type, block_id)
    if not session:
        raise ValueError(f"No trained model found for sheet '{sheet_name}' and block/curve '{block_id or curve_type}'")

    model_key = list(session.models.keys())[0]
    model = session.models[model_key]
    df_proc = session.processed_data[model_key]

    feature_names = getattr(model, "feature_names_in_", [c for c in df_proc.columns if c != "Run"][:-1])
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
    curve_type: str,
    block_id: Optional[str] = None,
    param_val: float = 1.0,
    param_vals: Optional[Dict[str, float]] = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    session = get_trained_session(file_id, sheet_name, curve_type, block_id)
    if not session:
        raise ValueError(f"No trained model found for sheet '{sheet_name}' and block/curve '{block_id or curve_type}'")

    model_key = list(session.models.keys())[0]
    model = session.models[model_key]
    df_proc = session.processed_data[model_key]

    feature_names = getattr(model, "feature_names_in_", [c for c in df_proc.columns if c != "Run"][:-1])
    x_col = feature_names[0]
    param_col = feature_names[1] if len(feature_names) > 1 else None

    target_cols = [c for c in df_proc.columns if c not in feature_names and c != "Run"]
    if not target_cols:
        target_cols = [df_proc.columns[-1]]
    target_col = target_cols[0]

    if param_col and param_col in df_proc.columns:
        actual_df = df_proc[df_proc[param_col] == param_val].sort_values(x_col)
        if actual_df.empty and session.param_values:
            closest_val = min(session.param_values, key=lambda v: abs(v - param_val))
            actual_df = df_proc[df_proc[param_col] == closest_val].sort_values(x_col)
    else:
        actual_df = df_proc.sort_values(x_col)

    if actual_df.empty:
        actual_df = df_proc.sort_values(x_col)

    actual_x = actual_df[x_col].tolist()
    actual_y = actual_df[target_col].tolist()

    input_dict = {x_col: actual_x}
    
    for p_col in feature_names[1:]:
        if param_vals and p_col in param_vals:
            input_dict[p_col] = [param_vals[p_col]] * len(actual_x)
        elif p_col == param_col:
            input_dict[p_col] = [param_val] * len(actual_x)
        else:
            input_dict[p_col] = [float(actual_df[p_col].iloc[0]) if p_col in actual_df.columns else float(df_proc[p_col].mean())] * len(actual_x)

    input_df = pd.DataFrame(input_dict)[feature_names]
    pred_y = model.predict(input_df).tolist()

    return actual_x, actual_y, actual_x, pred_y
