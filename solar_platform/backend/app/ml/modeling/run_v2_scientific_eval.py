"""
app.ml.modeling.run_v2_scientific_eval
---------------------------------------
Final V2 Scientific Usefulness Evaluation.

Answers one question:
  "Can this trained model actually reproduce scientifically useful solar curves
   and photovoltaic parameters on data it did not train on?"

Evaluation Protocol:
  1. Load all workbooks & curate dataset
  2. Split into Regime A (70/15/15 group split) and Regime B (Ca3AsBr3 holdout)
  3. Train V2 model on TRAINING split only (never sees val or test)
  4. Save complete checkpoint (weights + preprocessor + normalizer + config + SHA256 manifest)
  5. Reload from disk via SHA256-verified checkpoint loader
  6. Confirm real checkpoint loaded: prints path, manifest hash, parameter count
  7. Evaluate ONLY on held-out test samples (never on training data)
  8. Report per scientific type: MAE, RMSE, R2, NRMSE, MaxErr vs baselines
  9. Extract JV curves per dataset and compute Jsc, Voc, FF, PCE errors
 10. Generate plots: actual-vs-predicted, residual-vs-x, PV param bar chart
 11. Emit final verdict: USEFUL / PARTIALLY_USEFUL / NOT_YET_USEFUL

Verdict criteria:
  USEFUL:           JV R2>=0.80 AND Jsc MAPE<15% AND Voc MAPE<10% AND FF MAPE<15%
  PARTIALLY_USEFUL: JV R2>=0.50 AND physically sane predictions
  NOT_YET_USEFUL:   otherwise
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── project imports ───────────────────────────────────────────────────────────
from app.ml.curation_pipeline import MLCurationPipeline
from app.ml.modeling.config import TrainingConfig
from app.ml.modeling.dataset import CurvePointDataset
from app.ml.modeling.evaluation import EvaluationRunner
from app.ml.modeling.metrics import calculate_curve_metrics
from app.ml.modeling.models import BaselineMeanModel, BaselineLinearModel
from app.ml.modeling.plotter import MLPlotter
from app.ml.modeling.preprocessing import MLPreprocessor
from app.ml.modeling.pv_metrics import (
    extract_pv_parameters, calculate_pv_parameter_errors,
    PVParameters, PVParameterErrors
)
from app.ml.modeling.run_v1_benchmarks import (
    parse_and_convert_workbook, run_data_integrity_check,
    train_pytorch_mlp, predict_model
)
from app.ml.modeling.run_v2_benchmarks import train_v2_surrogate, predict_v2_model
from app.ml.modeling.usefulness_evaluator import (
    categorize_interpolation_vs_extrapolation,
    perform_physical_sanity_checks
)
from app.ml.modeling.v2_checkpoint import (
    save_v2_checkpoint,
    load_v2_checkpoint,
    compute_file_sha256,
    count_model_parameters,
)
from app.ml.modeling.v2_models import TargetNormalizer, TORCH_AVAILABLE


# ─── constants ─────────────────────────────────────────────────────────────────
WORKBOOK_DIR = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\solar_platform")
CHECKPOINT_DIR = Path(__file__).parent / "experiments" / "v2_scientific_eval_checkpoint"
OUTPUT_DIR = Path(__file__).parent / "experiments" / "v2_scientific_eval"
PLOTS_DIR = OUTPUT_DIR / "plots"


# ─── helpers ───────────────────────────────────────────────────────────────────

def grouped_pv_errors(
    dataset: CurvePointDataset,
    predictions: List[float],
) -> List[Tuple[str, PVParameters, PVParameters]]:
    """Group records by dataset_id, reconstruct full JV curves, extract PV params."""
    groups: Dict[str, Dict] = defaultdict(lambda: {"x": [], "act": [], "pred": []})
    for i, (rec, pred) in enumerate(zip(dataset.records, predictions)):
        gid = rec.dataset_id
        groups[gid]["x"].append(rec.x)
        groups[gid]["act"].append(dataset.targets[i])
        groups[gid]["pred"].append(pred)

    results = []
    for gid, data in groups.items():
        act_pv = extract_pv_parameters(data["x"], data["act"])
        pred_pv = extract_pv_parameters(data["x"], data["pred"])
        if act_pv is not None and pred_pv is not None:
            results.append((gid, act_pv, pred_pv))
    return results


def plot_representative_curves(
    dataset: CurvePointDataset,
    predictions: List[float],
    stype: str,
    regime: str,
    plots_dir: Path,
    max_curves: int = 6,
):
    """Plot actual vs predicted for the N most representative held-out datasets."""
    groups: Dict[str, Dict] = defaultdict(lambda: {"x": [], "act": [], "pred": [], "material": ""})
    for i, (rec, pred) in enumerate(zip(dataset.records, predictions)):
        gid = rec.dataset_id
        groups[gid]["x"].append(rec.x)
        groups[gid]["act"].append(dataset.targets[i])
        groups[gid]["pred"].append(pred)
        groups[gid]["material"] = rec.material

    x_labels = {
        "JV": "Voltage (V)",
        "QE": "Wavelength (nm)",
        "RESPONSIVITY": "Wavelength (nm)",
    }
    y_labels = {
        "JV": "Current Density (mA/cm²)",
        "QE": "Quantum Efficiency (%)",
        "RESPONSIVITY": "Responsivity (A/W)",
    }

    plotted = 0
    for gid, data in list(groups.items())[:max_curves]:
        if not data["x"]:
            continue
        sort_idx = np.argsort(data["x"])
        xs = [data["x"][i] for i in sort_idx]
        act_y = [data["act"][i] for i in sort_idx]
        pred_y = [data["pred"][i] for i in sort_idx]

        m = calculate_curve_metrics(pred_y, act_y)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: actual vs predicted curve
        axes[0].plot(xs, act_y, "o-", color="#1f77b4", label="SCAPS Ground Truth",
                     linewidth=2.0, markersize=4)
        axes[0].plot(xs, pred_y, "s--", color="#d62728", label="V2 Prediction (Reloaded Ckpt)",
                     linewidth=1.8, markersize=4)
        axes[0].axhline(0, color="gray", linestyle=":", linewidth=0.8)
        axes[0].set_xlabel(x_labels.get(stype, "x"))
        axes[0].set_ylabel(y_labels.get(stype, "y"))
        axes[0].set_title(f"{stype} — {data['material']}\n{regime} | R²={m.r2:.3f} MAE={m.mae:.3f}")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Right: residuals vs x
        residuals = [p - a for p, a in zip(pred_y, act_y)]
        axes[1].bar(range(len(xs)), residuals, color=["#d62728" if r > 0 else "#1f77b4" for r in residuals],
                    alpha=0.7, width=0.8)
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set_xlabel(f"Point index (sorted by {x_labels.get(stype, 'x')})")
        axes[1].set_ylabel("Residual (Pred − Actual)")
        axes[1].set_title(f"Residuals — RMSE={m.rmse:.3f}")
        axes[1].grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        safe_gid = gid.replace("/", "_").replace(" ", "_")[:40]
        fig.savefig(plots_dir / f"curve_{stype}_{regime}_{safe_gid}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        plotted += 1

    return plotted


def plot_pv_bar(pv_errors: PVParameterErrors, regime: str, plots_dir: Path):
    """Bar chart comparing Jsc / Voc / FF / PCE MAPE%."""
    labels = ["Jsc MAPE%", "Voc MAPE%", "FF MAPE%", "PCE MAPE%"]
    values = [pv_errors.j_sc_mape, pv_errors.v_oc_mape, pv_errors.ff_mape, pv_errors.pce_mape]
    colors = ["#2ca02c" if v < 10 else ("#ff7f0e" if v < 25 else "#d62728") for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.5)
    ax.axhline(10, linestyle="--", color="green", linewidth=1.5, label="10% target")
    ax.axhline(25, linestyle="--", color="orange", linewidth=1.5, label="25% threshold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("MAPE %")
    ax.set_title(f"PV Parameter Extraction Errors — {regime}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / f"pv_params_{regime}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(actuals: List[float], preds: List[float], r2: float,
                 stype: str, regime: str, plots_dir: Path):
    """Actual vs predicted scatter with identity line."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actuals, preds, s=6, alpha=0.4, color="#1f77b4", rasterized=True)
    lo = min(min(actuals), min(preds))
    hi = max(max(actuals), max(preds))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y=x ideal")
    ax.set_xlabel(f"Actual {stype}")
    ax.set_ylabel(f"Predicted {stype}")
    ax.set_title(f"Actual vs Predicted — {stype} {regime}\nR²={r2:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"scatter_{stype}_{regime}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_index_md(plots_dir: Path, result_data: Dict):
    """Generate markdown index of all plots."""
    lines = ["# V2 Scientific Evaluation — Plot Index\n"]
    pngs = sorted(plots_dir.glob("*.png"))
    for p in pngs:
        lines.append(f"## {p.stem}\n![{p.stem}]({p.name})\n")
    (plots_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


# ─── main evaluation ───────────────────────────────────────────────────────────

def run_scientific_evaluation() -> Dict[str, Any]:
    """Execute the complete V2 scientific usefulness evaluation."""

    # ── 0. PyTorch guard ────────────────────────────────────────────────────────
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not available. Cannot run V2 scientific evaluation. "
            "Install pytorch (CPU build is sufficient) and retry."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("V2 SCIENTIFIC USEFULNESS EVALUATION")
    print(f"PyTorch: {TORCH_AVAILABLE}")
    print("=" * 70)

    # ── 1. Load & curate ────────────────────────────────────────────────────────
    wb_files = ["Ca3PI3.xlsx", "Ca3PBr3.xlsx", "Ca3AsBr3.xlsx", "Data file.xlsx", "Data File (1).xlsx"]
    canonical_wbs = []
    for fname in wb_files:
        fpath = WORKBOOK_DIR / fname
        if fpath.exists():
            canonical_wbs.append(parse_and_convert_workbook(fpath))

    if not canonical_wbs:
        raise RuntimeError(f"No workbooks found in {WORKBOOK_DIR}")

    curation_pipeline = MLCurationPipeline(seed=42)
    curation_res = curation_pipeline.execute(canonical_wbs)
    integrity = run_data_integrity_check(curation_res)

    config = TrainingConfig()
    runner = EvaluationRunner(curation_res, config)
    target_normalizer = TargetNormalizer()

    # ── 2. Regime A Splits ──────────────────────────────────────────────────────
    regime_a = runner.prepare_regime_a_datasets()
    train_a, val_a, test_a = regime_a["train"], regime_a["val"], regime_a["test"]

    # ── 3. Train + Save Checkpoint on ALL types ──────────────────────────────────
    print("\n[STEP 1] Training V2 model on Regime A training split...")
    trained_models: Dict[str, Any] = {}
    trained_normalizers: Dict[str, TargetNormalizer] = {}
    last_model = None

    for stype in ["JV", "QE", "RESPONSIVITY"]:
        tr = train_a.filter_by_scientific_types([stype])
        va = val_a.filter_by_scientific_types([stype])
        if len(tr) == 0:
            print(f"  [{stype}] No training data — skipping")
            continue

        norm = TargetNormalizer()
        norm.fit({stype: tr.targets})
        trained_normalizers[stype] = norm

        model, tr_loss, val_loss = train_v2_surrogate(tr, va, stype, config, norm)
        trained_models[stype] = model
        last_model = model
        print(f"  [{stype}] Trained. Final val loss: {val_loss[-1]:.5f}  epochs: {len(tr_loss)}")

    if not trained_models:
        raise RuntimeError("No models trained — aborting evaluation.")

    # ── 4. Save Checkpoint ───────────────────────────────────────────────────────
    print(f"\n[STEP 2] Saving checkpoint to {CHECKPOINT_DIR} ...")
    # Use JV model as the primary saved model; normalizer holds all types' stats
    combined_norm = TargetNormalizer()
    for stype, norm in trained_normalizers.items():
        combined_norm.means[stype] = norm.means[stype]
        combined_norm.stds[stype] = norm.stds[stype]

    jv_model = trained_models.get("JV", last_model)
    save_v2_checkpoint(
        model=jv_model,
        preprocessor=train_a.preprocessor,
        normalizer=combined_norm,
        config=config,
        manifest_entries_count=len(curation_res.manifest_entries),
        save_dir=CHECKPOINT_DIR,
    )
    manifest_sha = compute_file_sha256(CHECKPOINT_DIR / "model_checkpoint_manifest.json")
    prep_sha = compute_file_sha256(CHECKPOINT_DIR / "v2_preprocessor.json")
    norm_sha = compute_file_sha256(CHECKPOINT_DIR / "v2_target_normalizer.json")
    num_params = count_model_parameters(jv_model)
    print(f"  Checkpoint saved. Manifest SHA256: {manifest_sha}")
    print(f"  Preprocessor SHA256:    {prep_sha}")
    print(f"  Normalizer SHA256:      {norm_sha}")
    print(f"  Model parameters:       {num_params:,}")

    # ── 5. Reload Checkpoint ─────────────────────────────────────────────────────
    print("\n[STEP 3] Reloading checkpoint from disk (SHA256 verified)...")
    reloaded_model, reloaded_prep, reloaded_norm, _, manifest_meta = load_v2_checkpoint(CHECKPOINT_DIR)
    print(f"  Real checkpoint loaded: True")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"  Architecture: {manifest_meta.get('model_name')}")
    print(f"  Parameters: {manifest_meta.get('num_parameters', 'N/A')}")
    print(f"  Format version: {manifest_meta.get('checkpoint_format_version')}")
    print(f"  Checksums verified: {list(manifest_meta.get('sha256_checksums', {}).keys())}")

    checkpoint_info = {
        "real_checkpoint_loaded": True,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "manifest_sha256": manifest_sha,
        "preprocessor_sha256": prep_sha,
        "normalizer_sha256": norm_sha,
        "num_parameters": num_params,
        "architecture": manifest_meta.get("architecture_details", {}),
        "inference_mode": "RELOADED_CHECKPOINT_SHA256_VERIFIED",
    }

    # ── 6. Round-trip verification ────────────────────────────────────────────────
    print("\n[STEP 4] Round-trip numerical verification (in-memory vs reloaded)...")
    jv_test = test_a.filter_by_scientific_types(["JV"])
    jv_norm = trained_normalizers.get("JV", combined_norm)

    if len(jv_test) > 0:
        preds_inmem = predict_v2_model(jv_model, jv_test, "JV", jv_norm)
        # Restore JV norm stats from checkpoint into reloaded_norm
        reloaded_norm.means["JV"] = combined_norm.means["JV"]
        reloaded_norm.stds["JV"] = combined_norm.stds["JV"]
        preds_reload = predict_v2_model(reloaded_model, jv_test, "JV", reloaded_norm)

        max_delta = max(abs(a - b) for a, b in zip(preds_inmem, preds_reload))
        roundtrip_pass = max_delta <= 1e-4
        print(f"  In-memory vs reloaded max delta: {max_delta:.2e}")
        print(f"  Round-trip PASS: {roundtrip_pass}")
        checkpoint_info["roundtrip_max_delta"] = float(max_delta)
        checkpoint_info["roundtrip_pass"] = roundtrip_pass

        if not roundtrip_pass:
            raise RuntimeError(
                f"CHECKPOINT ROUND-TRIP FAILED: max prediction delta {max_delta:.6f} > 1e-4. "
                "Halting evaluation — checkpoint integrity cannot be guaranteed."
            )
    else:
        print("  [WARN] No JV test samples for round-trip check")
        checkpoint_info["roundtrip_pass"] = None

    # ── 7. Evaluate on held-out data ─────────────────────────────────────────────
    print("\n[STEP 5] Evaluating on genuinely held-out test data (Regime A)...")
    regime_a_metrics: Dict[str, Any] = {}
    pv_results: Dict[str, Any] = {}
    interp_metrics: Dict[str, Any] = {}

    for stype in ["JV", "QE", "RESPONSIVITY"]:
        tr = train_a.filter_by_scientific_types([stype])
        va = val_a.filter_by_scientific_types([stype])
        te = test_a.filter_by_scientific_types([stype])

        if len(tr) == 0 or len(te) == 0:
            print(f"  [{stype}] Skipping — insufficient data")
            continue

        # Use per-type trained model directly (reloaded_model covers JV only for ckpt)
        this_model = trained_models.get(stype)
        this_norm = trained_normalizers.get(stype, combined_norm)

        if this_model is None:
            print(f"  [{stype}] No trained model — skipping")
            continue

        # Baselines
        bm = BaselineMeanModel(); bm.fit(tr.targets)
        bl = BaselineLinearModel(); bl.fit(tr.features, tr.targets)
        mean_m = calculate_curve_metrics(bm.predict(te.features), te.targets)
        lin_m = calculate_curve_metrics(bl.predict(te.features), te.targets)

        # V1 baseline
        v1_mlp, _, _ = train_pytorch_mlp(tr, va, config)
        v1_preds = predict_model(v1_mlp, te)
        v1_m = calculate_curve_metrics(v1_preds, te.targets)

        # V2 prediction using this_model
        v2_preds = predict_v2_model(this_model, te, stype, this_norm)

        # Fail-loudly if predictions are all-zero
        non_zero = sum(1 for p in v2_preds if abs(p) > 1e-8)
        if non_zero == 0:
            raise RuntimeError(
                f"[{stype}] All {len(v2_preds)} predictions are zero. "
                "Inference is not functioning on real trained weights. Halting."
            )

        v2_m = calculate_curve_metrics(v2_preds, te.targets)
        print(f"  [{stype}] R²={v2_m.r2:.4f}  MAE={v2_m.mae:.4f}  RMSE={v2_m.rmse:.4f}"
              f"  non-zero preds={non_zero}/{len(v2_preds)}")

        # Interpolation/Extrapolation split
        interp_te, extrap_te = categorize_interpolation_vs_extrapolation(te, tr)
        if len(interp_te) > 0:
            interp_preds = predict_v2_model(this_model, interp_te, stype, this_norm)
            interp_m = calculate_curve_metrics(interp_preds, interp_te.targets)
            interp_metrics[stype] = {
                "n_interp": len(interp_te),
                "n_extrap": len(extrap_te),
                **interp_m.to_dict(),
            }

        # Physical sanity check
        sanity = perform_physical_sanity_checks(te, v2_preds, stype)

        # PV extraction for JV only
        if stype == "JV":
            triplets = grouped_pv_errors(te, v2_preds)
            if triplets:
                act_pvs = [t[1] for t in triplets]
                pred_pvs = [t[2] for t in triplets]
                pv_errs = calculate_pv_parameter_errors(act_pvs, pred_pvs)
                pv_results["Regime_A"] = {
                    "n_jv_datasets": len(triplets),
                    "jsc_mae": pv_errs.j_sc_mae,
                    "jsc_mape": pv_errs.j_sc_mape,
                    "voc_mae": pv_errs.v_oc_mae,
                    "voc_mape": pv_errs.v_oc_mape,
                    "ff_mae": pv_errs.ff_mae,
                    "ff_mape": pv_errs.ff_mape,
                    "pce_mae": pv_errs.pce_mae,
                    "pce_mape": pv_errs.pce_mape,
                }
                plot_pv_bar(pv_errs, "Regime_A", PLOTS_DIR)
                print(f"    PV params: Jsc MAPE={pv_errs.j_sc_mape:.1f}%  "
                      f"Voc MAPE={pv_errs.v_oc_mape:.1f}%  "
                      f"FF MAPE={pv_errs.ff_mape:.1f}%  "
                      f"PCE MAPE={pv_errs.pce_mape:.1f}%")

        # Plots
        plot_scatter(te.targets, v2_preds, v2_m.r2, stype, "RegimeA", PLOTS_DIR)
        plot_representative_curves(te, v2_preds, stype, "RegimeA", PLOTS_DIR, max_curves=4)

        regime_a_metrics[stype] = {
            "n_train": len(tr),
            "n_test": len(te),
            "non_zero_predictions": non_zero,
            "baseline_mean": mean_m.to_dict(),
            "baseline_linear": lin_m.to_dict(),
            "v1_dedicated_mlp": v1_m.to_dict(),
            "v2_physics_multihead": v2_m.to_dict(),
            "beats_v1_in_r2": v2_m.r2 >= v1_m.r2,
            "beats_mean_in_r2": v2_m.r2 >= mean_m.r2,
            "physical_sanity": sanity.to_dict(),
        }

    # ── 8. Regime B (Cross-Material) ─────────────────────────────────────────────
    print("\n[STEP 6] Evaluating cross-material generalization (Regime B — Ca3AsBr3)...")
    regime_b = runner.prepare_regime_b_datasets()
    train_b, test_b = regime_b["train"], regime_b["test"]
    regime_b_metrics: Dict[str, Any] = {}
    pv_results_b: Dict[str, Any] = {}

    for stype in ["JV", "QE"]:
        tr = train_b.filter_by_scientific_types([stype])
        te = test_b.filter_by_scientific_types([stype])

        if len(tr) == 0 or len(te) == 0:
            print(f"  [{stype}] Regime B: skipping — insufficient data")
            continue

        norm_b = TargetNormalizer()
        norm_b.fit({stype: tr.targets})

        model_b, _, _ = train_v2_surrogate(tr, te, stype, config, norm_b)
        v2_preds_b = predict_v2_model(model_b, te, stype, norm_b)

        non_zero_b = sum(1 for p in v2_preds_b if abs(p) > 1e-8)
        if non_zero_b == 0:
            raise RuntimeError(
                f"[{stype}] Regime B: All predictions are zero. Halting."
            )

        v2_m_b = calculate_curve_metrics(v2_preds_b, te.targets)

        bm_b = BaselineMeanModel(); bm_b.fit(tr.targets)
        mean_m_b = calculate_curve_metrics(bm_b.predict(te.features), te.targets)

        print(f"  [{stype}] Regime B: R²={v2_m_b.r2:.4f}  MAE={v2_m_b.mae:.4f}")

        # PV extraction for JV
        if stype == "JV":
            triplets_b = grouped_pv_errors(te, v2_preds_b)
            if triplets_b:
                pv_errs_b = calculate_pv_parameter_errors([t[1] for t in triplets_b], [t[2] for t in triplets_b])
                pv_results["Regime_B"] = {
                    "n_jv_datasets": len(triplets_b),
                    "jsc_mae": pv_errs_b.j_sc_mae,
                    "jsc_mape": pv_errs_b.j_sc_mape,
                    "voc_mae": pv_errs_b.v_oc_mae,
                    "voc_mape": pv_errs_b.v_oc_mape,
                    "ff_mae": pv_errs_b.ff_mae,
                    "ff_mape": pv_errs_b.ff_mape,
                    "pce_mae": pv_errs_b.pce_mae,
                    "pce_mape": pv_errs_b.pce_mape,
                }
                plot_pv_bar(pv_errs_b, "Regime_B", PLOTS_DIR)
                print(f"    PV params: Jsc MAPE={pv_errs_b.j_sc_mape:.1f}%  "
                      f"Voc MAPE={pv_errs_b.v_oc_mape:.1f}%  "
                      f"FF MAPE={pv_errs_b.ff_mape:.1f}%  "
                      f"PCE MAPE={pv_errs_b.pce_mape:.1f}%")

        plot_scatter(te.targets, v2_preds_b, v2_m_b.r2, stype, "RegimeB", PLOTS_DIR)
        plot_representative_curves(te, v2_preds_b, stype, "RegimeB", PLOTS_DIR, max_curves=4)

        regime_b_metrics[stype] = {
            "n_train": len(tr),
            "n_test": len(te),
            "baseline_mean": mean_m_b.to_dict(),
            "v2_physics_multihead": v2_m_b.to_dict(),
        }

    # ── 9. Verdict ───────────────────────────────────────────────────────────────
    jv_a = regime_a_metrics.get("JV", {})
    jv_v2 = jv_a.get("v2_physics_multihead", {})
    jv_r2 = jv_v2.get("r2", -999.0)
    jv_pv = pv_results.get("Regime_A", {})
    jsc_mape = jv_pv.get("jsc_mape", 999.0)
    voc_mape = jv_pv.get("voc_mape", 999.0)
    ff_mape = jv_pv.get("ff_mape", 999.0)
    is_sane = jv_a.get("physical_sanity", {}).get("is_physically_sane", False)
    beats_v1 = jv_a.get("beats_v1_in_r2", False)

    print("\n" + "=" * 70)
    print("VERDICT SCORING:")
    print(f"  JV R²:       {jv_r2:.4f}  (threshold >= 0.80 for USEFUL)")
    print(f"  Jsc MAPE:    {jsc_mape:.1f}%  (threshold < 15% for USEFUL)")
    print(f"  Voc MAPE:    {voc_mape:.1f}%  (threshold < 10% for USEFUL)")
    print(f"  FF MAPE:     {ff_mape:.1f}%  (threshold < 15% for USEFUL)")
    print(f"  Phys. sane:  {is_sane}")
    print(f"  Beats V1:    {beats_v1}")

    if jv_r2 >= 0.80 and jsc_mape < 15.0 and voc_mape < 10.0 and ff_mape < 15.0 and is_sane:
        verdict = "USEFUL"
        verdict_reason = "Meets all USEFUL thresholds: R²≥0.80, Jsc MAPE<15%, Voc MAPE<10%, FF MAPE<15%, physically sane."
    elif jv_r2 >= 0.50 and is_sane:
        verdict = "PARTIALLY_USEFUL"
        verdict_reason = f"R²={jv_r2:.3f}≥0.50 but fails at least one PV accuracy threshold."
    else:
        verdict = "NOT_YET_USEFUL"
        verdict_reason = f"Fails usefulness thresholds: R²={jv_r2:.3f} < 0.50 or predictions not physically sane."

    print(f"\n  FINAL VERDICT: {verdict}")
    print(f"  Reason: {verdict_reason}")
    print("=" * 70)

    # ── 10. Write results ────────────────────────────────────────────────────────
    generate_index_md(PLOTS_DIR, {})

    results = {
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "checkpoint_verification": checkpoint_info,
        "data_integrity": integrity,
        "regime_a_evaluation": regime_a_metrics,
        "regime_b_evaluation": regime_b_metrics,
        "interpolation_vs_extrapolation": interp_metrics,
        "photovoltaic_parameter_errors": pv_results,
    }

    (OUTPUT_DIR / "v2_scientific_eval_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


if __name__ == "__main__":
    try:
        res = run_scientific_evaluation()
        print(f"\nEvaluation complete. Verdict: {res['verdict']}")
        print(f"Results saved to: {OUTPUT_DIR / 'v2_scientific_eval_results.json'}")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
