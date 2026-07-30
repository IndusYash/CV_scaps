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
        processed_data: Dict[str, pd.DataFrame],
        models: Dict[str, RandomForestRegressor],
        metrics: Dict[str, MetricPair],
        is_single_curve: bool = False,
    ):
        self.sheet_name = sheet_name
        self.processed_data = processed_data
        self.models = models
        self.metrics = metrics
        self.is_single_curve = is_single_curve

        # Parameter details
        self.param_col = None
        self.param_values = []
        for df in processed_data.values():
            for col in df.columns:
                if col not in ["Voltage", "Wavelength", "Current", "QE", "Responsivity", "Detectivity"]:
                    self.param_col = col
                    self.param_values = sorted(list(df[col].unique()))
                    break
            if self.param_col:
                break


# In-memory store keyed by (file_id, sheet_name)
_SESSION_STORE: Dict[Tuple[str, str], TrainedSheetSession] = {}


def train_sheet_models(
    file_id: str,
    sheet_name: str,
    df_raw: pd.DataFrame,
    test_size: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42,
) -> TrainedSheetSession:
    """Trains RandomForestRegressor models for all available curve types on a sheet."""
    # Check if single curve sheet
    col_str_lower = " ".join([str(c).lower() for c in df_raw.columns])
    is_single_curve = "jtot" in col_str_lower or "d*" in col_str_lower

    if is_single_curve:
        processed_data = preprocess_single_curve_sheet(df_raw)
    else:
        processed_data = preprocess_parametric_sheet(df_raw, sheet_name)

    models: Dict[str, RandomForestRegressor] = {}
    metrics: Dict[str, MetricPair] = {}

    for curve_type, df in processed_data.items():
        if df.empty:
            continue

        if is_single_curve:
            if curve_type in ["jv", "detectivity"]:
                feature_cols = ["Voltage"]
            else: # qe, responsivity
                feature_cols = ["Wavelength"]
            target_col = df.columns[1]
        else:
            param_col = [c for c in df.columns if c not in ["Voltage", "Wavelength", "Current", "QE", "Responsivity"]][0]
            x_col = "Voltage" if "Voltage" in df.columns else "Wavelength"
            feature_cols = [x_col, param_col]
            target_col = [c for c in df.columns if c not in feature_cols][0]

        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))

        models[curve_type] = model
        metrics[curve_type] = MetricPair(r2=r2, mae=mae)

    session = TrainedSheetSession(
        sheet_name=sheet_name,
        processed_data=processed_data,
        models=models,
        metrics=metrics,
        is_single_curve=is_single_curve,
    )

    _SESSION_STORE[(file_id, sheet_name)] = session
    return session


def get_trained_session(file_id: str, sheet_name: str) -> Optional[TrainedSheetSession]:
    return _SESSION_STORE.get((file_id, sheet_name))


def predict_point(
    file_id: str,
    sheet_name: str,
    curve_type: str,
    x_val: float,
    param_val: Optional[float] = None,
) -> float:
    session = get_trained_session(file_id, sheet_name)
    if not session or curve_type not in session.models:
        raise ValueError(f"No trained model found for sheet '{sheet_name}' and curve '{curve_type}'")

    model = session.models[curve_type]
    df_proc = session.processed_data[curve_type]

    if session.is_single_curve:
        x_col = df_proc.columns[0]
        input_df = pd.DataFrame({x_col: [x_val]})
    else:
        x_col = "Voltage" if "Voltage" in df_proc.columns else "Wavelength"
        param_col = session.param_col
        input_df = pd.DataFrame({x_col: [x_val], param_col: [param_val]})

    pred = model.predict(input_df)[0]
    return float(pred)


def predict_curve_comparison(
    file_id: str,
    sheet_name: str,
    curve_type: str,
    param_val: float,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    session = get_trained_session(file_id, sheet_name)
    if not session or curve_type not in session.models:
        raise ValueError(f"No trained model found for sheet '{sheet_name}' and curve '{curve_type}'")

    df_proc = session.processed_data[curve_type]
    param_col = session.param_col
    x_col = "Voltage" if "Voltage" in df_proc.columns else "Wavelength"
    target_col = [c for c in df_proc.columns if c not in [x_col, param_col]][0]

    actual_df = df_proc[df_proc[param_col] == param_val].sort_values(x_col)
    if actual_df.empty:
        # Fallback to closest parameter value if exact match not found
        closest_val = min(session.param_values, key=lambda v: abs(v - param_val))
        actual_df = df_proc[df_proc[param_col] == closest_val].sort_values(x_col)

    actual_x = actual_df[x_col].tolist()
    actual_y = actual_df[target_col].tolist()

    model = session.models[curve_type]
    input_df = pd.DataFrame({x_col: actual_x, param_col: [param_val] * len(actual_x)})
    pred_y = model.predict(input_df).tolist()

    return actual_x, actual_y, actual_x, pred_y
