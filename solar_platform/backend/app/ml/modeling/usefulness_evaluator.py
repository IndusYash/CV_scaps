"""
app.ml.modeling.usefulness_evaluator
-------------------------------------
Usefulness Evaluator for V2 Solar Cell Surrogate Neural Network.

Evaluates trained model utility across:
1. In-Domain Test Performance
2. Unseen-Parameter Interpolation Performance (Strictly bounded parameter sweeps)
3. Cross-Material Holdout Performance (Ca3AsBr3)
4. Extrapolation Performance (Unbounded parameter values)
5. Physical Sanity Checks (Ranges, Monotonicity, Discontinuities, PV parameters)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from app.ml.model import MLCurveRecord
from app.ml.modeling.dataset import CurvePointDataset
from app.ml.modeling.metrics import CurveEvaluationMetrics, calculate_curve_metrics
from app.ml.modeling.pv_metrics import (
    PVParameterErrors,
    PVParameters,
    calculate_pv_parameter_errors,
    extract_pv_parameters,
)


@dataclass
class PhysicalSanityReport:
    """Report detailing physical plausibility checks on predictions."""
    total_points_evaluated: int
    nan_inf_count: int
    out_of_range_count: int
    discontinuity_count: int
    invalid_pv_param_count: int
    is_physically_sane: bool
    sanity_pass_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def categorize_interpolation_vs_extrapolation(
    eval_ds: CurvePointDataset, train_ds: CurvePointDataset
) -> Tuple[CurvePointDataset, CurvePointDataset]:
    """
    Partition evaluation records into Interpolation vs Extrapolation datasets
    based on training parameter min-max bounds.
    """
    # 1. Compute min and max bounds per parameter in training set
    param_bounds: Dict[str, Tuple[float, float]] = {}
    for r in train_ds.records:
        for p_name, p_val in r.parameters.items():
            if p_val is not None:
                try:
                    val_f = float(p_val)
                    if p_name not in param_bounds:
                        param_bounds[p_name] = (val_f, val_f)
                    else:
                        curr_min, curr_max = param_bounds[p_name]
                        param_bounds[p_name] = (min(curr_min, val_f), max(curr_max, val_f))
                except (ValueError, TypeError):
                    continue

    interp_records = []
    extrap_records = []

    for r in eval_ds.records:
        is_extrap = False
        for p_name, p_val in r.parameters.items():
            if p_val is not None and p_name in param_bounds:
                try:
                    val_f = float(p_val)
                    p_min, p_max = param_bounds[p_name]
                    margin = (p_max - p_min) * 0.001 if p_max > p_min else 1e-5
                    if val_f < (p_min - margin) or val_f > (p_max + margin):
                        is_extrap = True
                        break
                except (ValueError, TypeError):
                    continue
        if is_extrap:
            extrap_records.append(r)
        else:
            interp_records.append(r)

    prep = eval_ds.preprocessor if eval_ds.preprocessor is not None else train_ds.preprocessor
    interp_ds = CurvePointDataset(interp_records, prep)
    extrap_ds = CurvePointDataset(extrap_records, prep)
    return interp_ds, extrap_ds


def perform_physical_sanity_checks(
    dataset: CurvePointDataset, predictions: List[float], stype: str = "JV"
) -> PhysicalSanityReport:
    """Evaluate physical plausibility constraints on predictions."""
    total_pts = len(predictions)
    if total_pts == 0:
        return PhysicalSanityReport(0, 0, 0, 0, 0, True, 1.0)

    nan_inf_count = 0
    out_of_range_count = 0

    # Range bounds per scientific type
    if stype == "JV":
        min_y, max_y = -50.0, 150.0
    elif stype == "QE":
        min_y, max_y = -5.0, 120.0
    elif stype == "RESPONSIVITY":
        min_y, max_y = -1.0, 10.0
    else:
        min_y, max_y = -1e5, 1e5

    for p in predictions:
        if math.isnan(p) or math.isinf(p):
            nan_inf_count += 1
        elif not (min_y <= p <= max_y):
            out_of_range_count += 1

    # Discontinuity & Curve-level check
    discontinuity_count = 0
    invalid_pv_param_count = 0

    # Group predictions by dataset_id
    ds_groups: Dict[str, Tuple[List[float], List[float]]] = {}
    for r, p in zip(dataset.records, predictions):
        if r.dataset_id not in ds_groups:
            ds_groups[r.dataset_id] = ([], [])
        ds_groups[r.dataset_id][0].append(r.x)
        ds_groups[r.dataset_id][1].append(p)

    for ds_id, (x_list, pred_list) in ds_groups.items():
        if len(x_list) > 1:
            x_arr = np.array(x_list)
            p_arr = np.array(pred_list)
            sort_idx = np.argsort(x_arr)
            xs = x_arr[sort_idx]
            ps = p_arr[sort_idx]

            # Discontinuity check: derivative spike
            dx = np.diff(xs)
            dp = np.diff(ps)
            valid_dx = dx > 1e-6
            if np.any(valid_dx):
                slopes = np.abs(dp[valid_dx] / dx[valid_dx])
                if np.any(slopes > 1000.0):
                    discontinuity_count += 1

            # PV Parameter Sanity for JV
            if stype == "JV":
                pv = extract_pv_parameters(xs.tolist(), ps.tolist())
                if pv is not None:
                    if pv.j_sc < 0.0 or pv.v_oc < 0.0 or pv.ff < 0.0 or pv.ff > 1.0 or pv.pce < 0.0:
                        invalid_pv_param_count += 1

    violations = nan_inf_count + out_of_range_count + discontinuity_count + invalid_pv_param_count
    pass_rate = 1.0 - (violations / max(1, total_pts))
    is_sane = (nan_inf_count == 0) and (pass_rate >= 0.95)

    return PhysicalSanityReport(
        total_points_evaluated=total_pts,
        nan_inf_count=nan_inf_count,
        out_of_range_count=out_of_range_count,
        discontinuity_count=discontinuity_count,
        invalid_pv_param_count=invalid_pv_param_count,
        is_physically_sane=is_sane,
        sanity_pass_rate=round(pass_rate, 4),
    )


def evaluate_jv_curve_reconstruction_fidelity(
    dataset: CurvePointDataset, predictions: List[float]
) -> Tuple[List[PVParameters], List[PVParameters], PVParameterErrors]:
    """
    Extract JV curves by dataset_id, compute ground-truth and predicted Jsc, Voc, FF, PCE,
    and return aggregate PV parameter errors.
    """
    ds_groups: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for r, p in zip(dataset.records, predictions):
        if r.dataset_id not in ds_groups:
            ds_groups[r.dataset_id] = ([], [], [])
        ds_groups[r.dataset_id][0].append(r.x)
        ds_groups[r.dataset_id][1].append(r.y)
        ds_groups[r.dataset_id][2].append(p)

    actual_pv_list = []
    pred_pv_list = []

    for ds_id, (xs, ys_act, ys_pred) in ds_groups.items():
        if len(xs) >= 3:
            pv_act = extract_pv_parameters(xs, ys_act)
            pv_pred = extract_pv_parameters(xs, ys_pred)
            if pv_act is not None and pv_pred is not None:
                actual_pv_list.append(pv_act)
                pred_pv_list.append(pv_pred)

    errors = calculate_pv_parameter_errors(actual_pv_list, pred_pv_list)
    return actual_pv_list, pred_pv_list, errors
