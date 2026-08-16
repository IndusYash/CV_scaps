"""
app.services.rf_pv_regressor
----------------------------
Curation and training of production RandomForestRegressor models for PV targets (Jsc, Voc, FF, PCE).
Provides strictly isolated, leak-proof experiment-level splits and baselines (Mean, Linear).
Supports interpolation and unseen_workbook validation modes.
"""
from __future__ import annotations

import datetime
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.database import db_manager
from app.ml.parameter_curation import CurationStatus, curate_parameter

# Standard feature keys for PV parameters
# Note: illumination_wavelength = 0.0 is a sentinel representing the standard 
# broadband/default AM1.5G condition. It is NOT a physical wavelength of 0 nm.
FEATURE_KEYS = [
    "thickness",
    "defect_density",
    "temperature",
    "series_resistance",
    "shunt_resistance",
    "doping_density",
    "illumination_wavelength",
]

# Minimum test samples for statistical validation significance
MIN_MEANINGFUL_TEST_SAMPLES = 5


def build_ml_dataset_from_db() -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Scans the SQLite database for all valid illuminated JV curves, curates standard features,
    and returns a clean tabular dataset for RandomForestRegressor training.
    """
    datasets = db_manager.list_datasets(scientific_type="JV")
    records = []
    audit_log = []

    for ds in datasets:
        # Check validation status and ML readiness
        if not ds.is_ml_ready:
            audit_log.append({
                "dataset_id": ds.dataset_id,
                "status": "EXCLUDED",
                "reason": f"Dataset validation status is not ready ({ds.validation_status})"
            })
            continue

        # Extract material
        workbook_stem = Path(ds.workbook_name).stem.lower()
        material = "SolarDevice"
        for m in ["ca3asbr3", "ca3pbr3", "ca3pi3"]:
            if m in workbook_stem or m in ds.sheet_name.lower():
                material = m.capitalize()
                break

        # Curation-level group_id to prevent leakages
        parent_ref = ds.parent_region_id or ds.region_id
        group_id = f"grp_{material}_{ds.sheet_name}_{parent_ref}"

        # Fetch global features
        global_features = {}
        for p in ds.global_parameters:
            curated = curate_parameter(p, ds.workbook_name, ds.sheet_name, ds.region_id, "JV")
            if curated.curation_status != CurationStatus.INDEX_COLUMN:
                global_features[curated.curated_name] = curated.curated_value

        for trace in ds.traces:
            trace_id = trace.trace_id
            
            # Retrieve already persisted actual PV metrics (Jsc, Voc, FF, PCE)
            metrics = db_manager.get_derived_metrics(ds.dataset_id, trace_id)
            if not metrics or metrics["j_sc"] is None or metrics["v_oc"] is None or metrics["ff"] is None or metrics["pce"] is None:
                audit_log.append({
                    "dataset_id": ds.dataset_id,
                    "trace_id": trace_id,
                    "status": "EXCLUDED",
                    "reason": "Derived PV metrics (Jsc, Voc, FF, PCE) not found in database"
                })
                continue

            # Exclude dark-current traces:
            # Match "dark" or "diode" keywords in trace name, parameter, unit or dataset details
            p_raw = trace.parameter.raw_text.lower() if trace.parameter and trace.parameter.raw_text else ""
            p_val = str(trace.parameter.value).lower() if trace.parameter and trace.parameter.value is not None else ""
            p_name = trace.parameter.name.lower() if trace.parameter and trace.parameter.name else ""
            trace_text = f"{trace_id} {p_raw} {p_val} {p_name}".lower()
            ds_text = f"{ds.dataset_id} {ds.sheet_name}".lower()
            all_text = f"{trace_text} {ds_text}"

            is_dark = "dark" in all_text or "diode" in all_text
            is_low_current = metrics["j_sc"] < 0.1 # Low Jsc check for dark current validation

            if is_dark or is_low_current:
                audit_log.append({
                    "dataset_id": ds.dataset_id,
                    "trace_id": trace_id,
                    "status": "EXCLUDED",
                    "reason": "Trace identified as dark current or non-illuminated J-V sweep"
                })
                continue

            # Curate trace-level parameter if present
            trace_features = dict(global_features)
            if trace.parameter:
                curated = curate_parameter(trace.parameter, ds.workbook_name, ds.sheet_name, ds.region_id, "JV")
                if curated.curation_status != CurationStatus.INDEX_COLUMN:
                    trace_features[curated.curated_name] = curated.curated_value

            # Construct clean tabular record
            row = {
                "dataset_id": ds.dataset_id,
                "trace_id": trace_id,
                "workbook_name": ds.workbook_name,
                "material": material,
                "group_id": group_id,
                "j_sc": metrics["j_sc"],
                "v_oc": metrics["v_oc"],
                "ff": metrics["ff"],
                "pce": metrics["pce"],
            }
            
            # Map standard features (defaulting missing to 0.0)
            for key in FEATURE_KEYS:
                val = trace_features.get(key, 0.0)
                try:
                    row[key] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    row[key] = 0.0

            records.append(row)
            audit_log.append({
                "dataset_id": ds.dataset_id,
                "trace_id": trace_id,
                "status": "ELIGIBLE",
                "reason": "PV Metrics extracted successfully"
            })

    df = pd.DataFrame(records)
    return df, audit_log


def generate_ml_audit_report(audit_log: List[Dict[str, Any]], output_path: Path):
    """Writes a markdown audit report detailing curation statistics and exclusions."""
    eligible = [r for r in audit_log if r["status"] == "ELIGIBLE"]
    excluded = [r for r in audit_log if r["status"] == "EXCLUDED"]

    lines = [
        "# V1 ML Dataset Curation and Audit Report",
        f"\nGenerated At: {datetime.datetime.utcnow().isoformat()}Z\n",
        "## Curation Statistics",
        f"- **Total Scanned Curves (Traces)**: {len(audit_log)}",
        f"- **Eligible Curve Records**: {len(eligible)}",
        f"- **Excluded Curve Records**: {len(excluded)}",
        "\n## Exclusions Breakdown",
    ]

    exclusions_by_reason = defaultdict(int)
    for r in excluded:
        exclusions_by_reason[r["reason"]] += 1

    for reason, cnt in exclusions_by_reason.items():
        lines.append(f"- *{reason}*: {cnt} curves")

    lines.append("\n## Detailed Excluded Curves")
    for r in excluded[:30]:
        lines.append(f"- `{r['dataset_id']}` / `{r.get('trace_id', 'N/A')}`: {r['reason']}")
    if len(excluded) > 30:
        lines.append(f"- ... and {len(excluded)-30} more.")

    output_path.write_text("\n".join(lines), encoding="utf-8")


class MLRegressorManager:
    """Manages training and metrics evaluation of classical models on curated metrics."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models: Dict[str, RandomForestRegressor] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.predictions: Dict[str, Dict[str, List[Any]]] = {}

    def train_and_evaluate(
        self,
        target_name: str,
        material_filter: Optional[str] = None,
        test_size: float = 0.25,
        random_state: int = 42,
        validation_mode: str = "interpolation",
        test_workbook: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train RandomForestRegressor, LinearRegression, and Mean Baseline models.
        Supports:
          - "interpolation": splits data by group_id (experiment-level split) to prevent data leakage.
          - "unseen_workbook": completely isolates an Excel workbook to the test set.
        """
        df_target = self.df.copy()
        if material_filter:
            df_target = df_target[df_target["material"].str.lower() == material_filter.lower()]

        if len(df_target) < 2:
            return {"status": "INSUFFICIENT_DATA", "count": len(df_target), "validation_mode": validation_mode}

        # Determine train/test splits based on validation mode
        if validation_mode == "unseen_workbook":
            if not test_workbook:
                # Pick the first workbook in the list as default
                test_workbook = df_target["workbook_name"].iloc[0] if not df_target.empty else ""

            test_mask = df_target["workbook_name"] == test_workbook
            train_mask = ~test_mask

            df_train = df_target[train_mask]
            df_test = df_target[test_mask]
        else:
            # Interpolation Mode: split strictly by experiment group_id
            unique_groups = df_target["group_id"].unique()
            
            # Deterministic shuffle of groups to split
            rng = np.random.RandomState(random_state)
            shuffled_groups = rng.permutation(unique_groups)
            
            split_idx = int(len(shuffled_groups) * (1 - test_size))
            train_groups = set(shuffled_groups[:split_idx])
            test_groups = set(shuffled_groups[split_idx:])

            # Handle edge case where split would result in empty train/test sets
            if not train_groups:
                train_groups = {shuffled_groups[0]}
                test_groups = set(shuffled_groups[1:])
            if not test_groups:
                test_groups = {shuffled_groups[-1]}
                train_groups = set(shuffled_groups[:-1])

            train_mask = df_target["group_id"].isin(train_groups)
            test_mask = df_target["group_id"].isin(test_groups)

            df_train = df_target[train_mask]
            df_test = df_target[test_mask]

        if len(df_train) < 1 or len(df_test) < 1:
            return {
                "status": "INSUFFICIENT_DATA",
                "count": len(df_target),
                "train_count": len(df_train),
                "test_count": len(df_test),
                "validation_mode": validation_mode
            }

        features = FEATURE_KEYS.copy()
        
        # Fit one-hot material encoding from train set only (prevent information leakage)
        if not material_filter and len(df_target["material"].unique()) > 1:
            train_materials = sorted(df_train["material"].unique())
            mat_cols = [f"mat_{m.lower()}" for m in train_materials]
            
            for mc, mat in zip(mat_cols, train_materials):
                df_train = df_train.copy()
                df_test = df_test.copy()
                df_train[mc] = (df_train["material"] == mat).astype(float)
                df_test[mc] = (df_test["material"] == mat).astype(float)
                
            features.extend(mat_cols)

        X_train = df_train[features].to_numpy()
        y_train = df_train[target_name].to_numpy()
        X_test = df_test[features].to_numpy()
        y_test = df_test[target_name].to_numpy()

        # 1. Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=150, random_state=random_state, max_depth=8)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # 2. Baseline: Mean Predictor
        mean_val = np.mean(y_train)
        y_pred_mean = np.full_like(y_test, mean_val)

        # 3. Baseline: Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        # Calculate metrics safely
        def get_metrics(act, pred):
            if len(act) < 2:
                return {
                    "r2": float("nan"),
                    "mae": float(mean_absolute_error(act, pred)),
                    "rmse": float(np.sqrt(mean_squared_error(act, pred))),
                }
            return {
                "r2": float(r2_score(act, pred)),
                "mae": float(mean_absolute_error(act, pred)),
                "rmse": float(np.sqrt(mean_squared_error(act, pred))),
            }

        is_statistically_meaningful = len(y_test) >= MIN_MEANINGFUL_TEST_SAMPLES

        self.models[target_name] = rf
        metrics_dict = {
            "validation_mode": validation_mode,
            "is_statistically_meaningful": is_statistically_meaningful,
            "min_meaningful_test_samples": MIN_MEANINGFUL_TEST_SAMPLES,
            "train_count": len(df_train),
            "test_count": len(df_test),
            "train_workbooks": sorted(df_train["workbook_name"].unique().tolist()),
            "test_workbooks": sorted(df_test["workbook_name"].unique().tolist()),
            "random_forest": get_metrics(y_test, y_pred_rf),
            "baseline_mean": get_metrics(y_test, y_pred_mean),
            "baseline_linear": get_metrics(y_test, y_pred_lr),
            "features": features,
        }
        
        self.metrics[target_name] = metrics_dict

        # Collect predictions vs actuals for visualization
        self.predictions[target_name] = {
            "actuals": y_test.tolist(),
            "predicted_rf": y_pred_rf.tolist(),
            "predicted_mean": y_pred_mean.tolist(),
            "predicted_linear": y_pred_lr.tolist(),
            "group_ids": df_test["group_id"].tolist(),
            "dataset_ids": df_test["dataset_id"].tolist(),
            "workbook_names": df_test["workbook_name"].tolist(),
        }

        return metrics_dict
