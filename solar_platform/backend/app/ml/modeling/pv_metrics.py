"""
app.ml.modeling.pv_metrics
--------------------------
Photovoltaic Parameter Extraction & Full-Curve Fidelity Evaluation Engine.

Provides exact calculations for:
- Jsc: Short-circuit current density J(V=0)
- Voc: Open-circuit voltage V where J(V)=0
- FF: Fill Factor Pmax / (Jsc * Voc)
- PCE: Power Conversion Efficiency Pmax / Pin
- Detailed Curve Metrics: MAE, RMSE, R2, NRMSE, Max Error
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PVParameters:
    """Core Photovoltaic Performance Parameters derived from JV curves."""
    j_sc: float  # mA/cm²
    v_oc: float  # V
    p_max: float  # mW/cm²
    ff: float     # Fill Factor (0 to 1)
    pce: float    # Efficiency %

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class PVParameterErrors:
    """Absolute and relative error metrics between actual and predicted PV parameters."""
    j_sc_mae: float
    j_sc_mape: float
    v_oc_mae: float
    v_oc_mape: float
    ff_mae: float
    ff_mape: float
    pce_mae: float
    pce_mape: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def extract_pv_parameters(
    v_vals: List[float], j_vals: List[float], p_in: float = 100.0
) -> Optional[PVParameters]:
    """
    Extract Jsc, Voc, FF, PCE from JV curve (Voltage in V, Current Density in mA/cm²).
    Assumes standard power-generating quadrant (J > 0 at V=0, J drops as V increases).
    """
    if len(v_vals) < 2 or len(j_vals) < 2 or len(v_vals) != len(j_vals):
        return None

    v_arr = np.array(v_vals, dtype=np.float64)
    j_arr = np.array(j_vals, dtype=np.float64)

    # Sort by voltage
    sort_idx = np.argsort(v_arr)
    v_sorted = v_arr[sort_idx]
    j_sorted = j_arr[sort_idx]

    # 1. Jsc = J(V=0) via linear interpolation
    if 0.0 >= v_sorted[0] and 0.0 <= v_sorted[-1]:
        j_sc = float(np.interp(0.0, v_sorted, j_sorted))
    else:
        j_sc = float(j_sorted[0])
    j_sc = max(0.0, j_sc)

    # 2. Voc = V where J(V)=0 via zero-crossing linear interpolation
    v_oc = 0.0
    zero_crossings = []
    for i in range(len(j_sorted) - 1):
        if (j_sorted[i] >= 0 and j_sorted[i + 1] <= 0) or (j_sorted[i] <= 0 and j_sorted[i + 1] >= 0):
            j1, j2 = j_sorted[i], j_sorted[i + 1]
            v1, v2 = v_sorted[i], v_sorted[i + 1]
            if abs(j2 - j1) > 1e-9:
                v_zero = v1 + (0.0 - j1) * (v2 - v1) / (j2 - j1)
                zero_crossings.append(v_zero)

    if zero_crossings:
        v_oc = max(0.0, float(zero_crossings[0]))
    else:
        v_oc = max(0.0, float(v_sorted[-1]))

    # 3. Pmax = max(V * J(V)) for V in [0, Voc]
    power_vals = v_sorted * j_sorted
    valid_power_mask = (v_sorted >= 0.0) & (v_sorted <= (v_oc + 0.1)) & (j_sorted >= 0.0)
    if np.any(valid_power_mask):
        p_max = float(np.max(power_vals[valid_power_mask]))
    else:
        p_max = float(np.max(np.maximum(0.0, power_vals)))

    # 4. Fill Factor FF = Pmax / (Jsc * Voc)
    denom = j_sc * v_oc
    if denom > 1e-6:
        ff = p_max / denom
        ff = max(0.0, min(1.0, ff))  # Bound physical FF between 0 and 1
    else:
        ff = 0.0

    # 5. PCE = Pmax / Pin * 100%
    pce = (p_max / p_in) * 100.0 if p_in > 0 else 0.0

    return PVParameters(
        j_sc=round(j_sc, 4),
        v_oc=round(v_oc, 4),
        p_max=round(p_max, 4),
        ff=round(ff, 4),
        pce=round(pce, 4),
    )


def calculate_pv_parameter_errors(
    actual_params: List[PVParameters], pred_params: List[PVParameters]
) -> PVParameterErrors:
    """Compute MAE and MAPE error metrics across lists of PV parameters."""
    n = len(actual_params)
    if n == 0 or n != len(pred_params):
        return PVParameterErrors(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    j_sc_errs = [abs(a.j_sc - p.j_sc) for a, p in zip(actual_params, pred_params)]
    j_sc_pes = [abs(a.j_sc - p.j_sc) / max(1e-3, abs(a.j_sc)) * 100.0 for a, p in zip(actual_params, pred_params)]

    v_oc_errs = [abs(a.v_oc - p.v_oc) for a, p in zip(actual_params, pred_params)]
    v_oc_pes = [abs(a.v_oc - p.v_oc) / max(1e-3, abs(a.v_oc)) * 100.0 for a, p in zip(actual_params, pred_params)]

    ff_errs = [abs(a.ff - p.ff) for a, p in zip(actual_params, pred_params)]
    ff_pes = [abs(a.ff - p.ff) / max(1e-3, abs(a.ff)) * 100.0 for a, p in zip(actual_params, pred_params)]

    pce_errs = [abs(a.pce - p.pce) for a, p in zip(actual_params, pred_params)]
    pce_pes = [abs(a.pce - p.pce) / max(1e-3, abs(a.pce)) * 100.0 for a, p in zip(actual_params, pred_params)]

    return PVParameterErrors(
        j_sc_mae=round(float(np.mean(j_sc_errs)), 4),
        j_sc_mape=round(float(np.mean(j_sc_pes)), 2),
        v_oc_mae=round(float(np.mean(v_oc_errs)), 4),
        v_oc_mape=round(float(np.mean(v_oc_pes)), 2),
        ff_mae=round(float(np.mean(ff_errs)), 4),
        ff_mape=round(float(np.mean(ff_pes)), 2),
        pce_mae=round(float(np.mean(pce_errs)), 4),
        pce_mape=round(float(np.mean(pce_pes)), 2),
    )
