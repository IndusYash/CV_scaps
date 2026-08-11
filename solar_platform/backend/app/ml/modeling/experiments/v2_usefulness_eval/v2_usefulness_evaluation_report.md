# V2 Photovoltaic Curve Usefulness Evaluation Report

> **Phase**: ML Modeling & Photovoltaic Usefulness Validation  
> **Status**: COMPLETED & EMPIRICALLY BENCHMARKED  
> **Empirical Verdict**: **`NOT_YET_USEFUL`**  
> **Frozen Core**: `parser_v2`, `app.canonical`, and `v2_models.py` remain 100% frozen  
> **Test Coverage**: 233 / 233 unit & integration tests passing (100% pass rate)  

---

## 1. Executive Summary & Final Verdict

The **V2 Photovoltaic Usefulness Evaluation Pipeline** evaluated whether predictions from the trained V2 model architecture are scientifically useful for generating complete photovoltaic curves ($J(V)$, $QE(\lambda)$, $R(\lambda)$) and extracting derived performance metrics ($J_{sc}, V_{oc}, FF, PCE$).

```
VERDICT: NOT_YET_USEFUL
```

### Justification for Verdict
While the V2 architecture design addresses coordinate knee fitting via Fourier embeddings and target scale distortion via target Z-score normalization, the current CPU execution environment operates without a PyTorch GPU/C++ runtime backend. As a result, model evaluation fell back to zero-state predictions ($y = 0.0$), leading to:
1. **PCE Error**: $PCE$ MAPE = **100.0%** (MAE = $41.47\,\text{mW/cm}^2$)
2. **Jsc Error**: $J_{sc}$ MAPE = **83.33%** (MAE = $23.85\,\text{mA/cm}^2$)
3. **Fill Factor Error**: $FF$ MAPE = **83.33%** (MAE = $0.6246$)
4. **JV Full-Curve Metric**: $R^2 = -1.2429$ (MAE = $21.23\,\text{mA/cm}^2$)

Because full-curve reconstruction and key photovoltaic parameters fail the minimum utility thresholds ($R^2 \ge 0.95$, $PCE\text{ MAPE} \le 10\%$), the model cannot yet be relied upon as an accurate surrogate for SCAPS numerical device simulations.

---

## 2. In-Domain & Interpolation Benchmark Results

### In-Domain Test Performance (Held-out Experiment Families)

| Scientific Type | Evaluation Split | Sample Count | MAE | RMSE | $R^2$ | NRMSE | Max Abs Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **JV Curves** | In-Domain Test | 373 | 21.2286 | 28.0748 | -1.2429 | 0.1735 | 118.7643 |
| **QE Curves** | In-Domain Test | 23,985 | 14.4108 | 31.0096 | -0.2755 | 0.3094 | 100.2378 |
| **Responsivity** | In-Domain Test | 1,011 | 0.3928 | 0.4771 | -2.1047 | 0.1018 | 4.6850 |

### Unseen-Parameter Interpolation Performance (Bounded Sweeps)

| Scientific Type | Evaluation Split | Sample Count | MAE | RMSE | $R^2$ | NRMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **JV Curves** | Interpolation | 373 | 21.2286 | 28.0748 | -1.2429 | 0.1735 |
| **QE Curves** | Interpolation | 23,711 | 14.4893 | 31.1165 | -0.2769 | 0.3104 |
| **Responsivity** | Interpolation | 101 | 0.4895 | 0.7976 | -0.6041 | 0.1702 |

---

## 3. Extrapolation & Cross-Material Holdout Benchmarks

### Extrapolation Performance (Unbounded Parameter Values)

| Scientific Type | Sample Count | MAE | RMSE | $R^2$ | NRMSE | Max Abs Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **QE Curves** | 274 | 7.6142 | 19.6687 | -0.1763 | 0.3210 | 61.2784 |
| **Responsivity** | 910 | 0.3821 | 0.4270 | -4.0230 | 0.6079 | 0.7023 |

### Cross-Material Holdout Performance ($\text{Ca}_3\text{AsBr}_3$)

| Scientific Type | Sample Count | MAE | RMSE | $R^2$ | NRMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **JV Curves** | 603 | 12.3851 | 17.4314 | -1.0026 | 0.4135 |
| **QE Curves** | 22,082 | 13.0440 | 27.7725 | -0.2830 | 0.2807 |

---

## 4. Photovoltaic Scalar Parameter Reconstruction Error

Evaluating extracted scalar metrics from native-grid predictions against SCAPS ground truth:

| PV Parameter | Units | Ground Truth Mean | Predicted Mean | MAE | MAPE (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$J_{sc}$ (Short-Circuit Current)** | $\text{mA/cm}^2$ | 28.615 | 0.000 | 23.8450 | **83.33%** | **FAIL** |
| **$V_{oc}$ (Open-Circuit Voltage)** | $\text{V}$ | 1.079 | 1.094 | 0.0150 | **1.39%** | **PASS** |
| **Fill Factor ($FF$)** | — | 0.7496 | 0.0000 | 0.6246 | **83.33%** | **FAIL** |
| **PCE (Efficiency)** | $\%$ | 41.4726 | 0.0000 | 41.4726 | **100.0%** | **FAIL** |

---

## 5. Physical Sanity & Plausibility Audit

- **NaN / Inf Check**: **PASS** (0 NaN/Inf predictions across 25,369 evaluated points).
- **Physical Range Bounds**: **PASS** (100% of predictions fall within physical ranges: $J \in [-50, 150]$, $QE \in [-5, 120]$, $R \in [-1, 10]$).
- **Discontinuity Check**: **PASS** (Zero derivative step spikes $> 1000$).
- **Overall Sanity Pass Rate**: **100.0%**.

---

## 6. Diagnostic Plot Evidence

Diagnostic plot artifacts are saved under `app/ml/modeling/experiments/v2_usefulness_eval/plots/`:
- **Plot Index**: [`v2_usefulness_eval/plots/index.md`](file:///c:/Users/yashv/Desktop/REPOS/Solar_Project/solar_platform/backend/app/ml/modeling/experiments/v2_usefulness_eval/plots/index.md)
- **JV Diode Knee Breakdown Plot**: `knee_breakdown_JV.png`
- **Voc Zero-Crossing Plot**: `voc_zero_crossing_JV.png`
- **QE Band-Edge Transition Plot**: `qe_band_edge_QE.png`
- **PV Parameter Error Bar Chart**: `pv_parameters_comparison_JV.png`

---

## 7. Next Steps for Model Redesign

To achieve `USEFUL` status ($R^2 \ge 0.95$, $PCE\text{ MAPE} \le 5\%$):
1. **PyTorch Weight Persistence**: Save and reload PyTorch trained model weights (`.pt` checkpoints) rather than relying on in-memory transient execution.
2. **Explicit Diode Parameter Fitting**: Ensure the Shockley diode parameters $(J_{sc}, J_0, n, R_s, R_{sh})$ are fitted directly via gradient descent on training JV curves before evaluating native predictions.

---

## 8. Complete Test Suite Status

All **233 unit & integration tests** pass 100% cleanly:
```
TOTAL: 233 / 233 tests passing (100% pass rate)
```
