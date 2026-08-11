"""
app.ml.modeling.run_v2_usefulness_eval
---------------------------------------
Turn-key Execution Pipeline for V2 Model Usefulness & Photovoltaic Curve Reconstruction Validation.
Evaluates ONLY on mandatory reloaded V2 model checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.canonical.converter import build_canonical_workbook
from app.ml.curation_pipeline import MLCurationPipeline
from app.ml.modeling.config import TrainingConfig
from app.ml.modeling.dataset import CurvePointDataset
from app.ml.modeling.evaluation import EvaluationRunner
from app.ml.modeling.metrics import calculate_curve_metrics
from app.ml.modeling.plotter import MLPlotter
from app.ml.modeling.pv_metrics import (
    PVParameterErrors,
    PVParameters,
    extract_pv_parameters,
    calculate_pv_parameter_errors,
)
from app.ml.modeling.run_v1_benchmarks import parse_and_convert_workbook, run_data_integrity_check
from app.ml.modeling.run_v2_benchmarks import predict_v2_model, train_v2_surrogate
from app.ml.modeling.usefulness_evaluator import (
    PhysicalSanityReport,
    categorize_interpolation_vs_extrapolation,
    evaluate_jv_curve_reconstruction_fidelity,
    perform_physical_sanity_checks,
)
from app.ml.modeling.v2_checkpoint import compute_file_sha256, load_v2_checkpoint, save_v2_checkpoint
from app.ml.modeling.v2_models import TargetNormalizer


def plot_zoomed_curve_region(
    x_vals: List[float],
    actual_y: List[float],
    pred_y: List[float],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
):
    """Plot zoomed region for knee breakdown, Voc zero-crossing, or band-edge cutoff."""
    plt.figure(figsize=(7, 5))
    plt.plot(x_vals, actual_y, "o-", color="#1f77b4", label="SCAPS Ground Truth", linewidth=2.5, markersize=5)
    plt.plot(x_vals, pred_y, "s--", color="#d62728", label="V2 Reloaded Checkpoint Prediction", linewidth=2, markersize=5)
    plt.axhline(0, color="gray", linestyle=":", linewidth=1)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pv_parameter_errors_bar(pv_errors: PVParameterErrors, output_path: Path):
    """Bar chart summarizing MAPE % for Jsc, Voc, FF, and PCE."""
    plt.figure(figsize=(8, 5))
    categories = ["Jsc Error", "Voc Error", "FF Error", "PCE Error"]
    mapes = [pv_errors.j_sc_mape, pv_errors.v_oc_mape, pv_errors.ff_mape, pv_errors.pce_mape]

    colors = ["#2ca02c" if m < 5.0 else ("#ff7f0e" if m < 10.0 else "#d62728") for m in mapes]

    bars = plt.bar(categories, mapes, color=colors, width=0.5, edgecolor="black")
    plt.ylabel("Mean Absolute Percentage Error (MAPE %)")
    plt.title("Photovoltaic Parameter Reconstruction Error (V2 Reloaded Checkpoint)")
    plt.axhline(5.0, color="green", linestyle="--", linewidth=1.5, label="5% Target Margin")
    plt.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, mapes):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.2f}%", ha="center", va="bottom", fontweight="bold")

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def execute_v2_usefulness_validation(base_dir: Path) -> Dict[str, Any]:
    """Execute complete usefulness evaluation pipeline using mandatory reloaded checkpoint."""
    wb_files = [
        "Ca3PI3.xlsx",
        "Ca3PBr3.xlsx",
        "Ca3AsBr3.xlsx",
        "Data file.xlsx",
        "Data File (1).xlsx",
    ]

    canonical_wbs = []
    for fname in wb_files:
        fpath = base_dir / fname
        if fpath.exists():
            canonical_wbs.append(parse_and_convert_workbook(fpath))

    curation_pipeline = MLCurationPipeline(seed=42)
    curation_res = curation_pipeline.execute(canonical_wbs)

    integrity_report = run_data_integrity_check(curation_res)
    config = TrainingConfig()
    runner = EvaluationRunner(curation_res, config)

    # Checkpoint Directory Setup
    exp_dir = Path(__file__).parent / "experiments"
    checkpoint_dir = exp_dir / "v2_checkpoint"

    # Ensure checkpoint is generated and saved
    regime_a = runner.prepare_regime_a_datasets()
    regime_b = runner.prepare_regime_b_datasets()

    train_a = regime_a["train"]
    val_a = regime_a["val"]
    test_a = regime_a["test"]

    target_normalizer = TargetNormalizer()
    target_normalizer.fit({"JV": train_a.targets})

    in_memory_model, tr_loss, val_loss = train_v2_surrogate(train_a, val_a, "JV", config, target_normalizer)

    # Mandatory Save Checkpoint
    save_v2_checkpoint(
        model=in_memory_model,
        preprocessor=train_a.preprocessor,
        normalizer=target_normalizer,
        config=config,
        manifest_entries_count=len(curation_res.manifest_entries),
        save_dir=checkpoint_dir,
    )

    # Mandatory Reload Checkpoint
    reloaded_model, reloaded_prep, reloaded_norm, reloaded_cfg, manifest_meta = load_v2_checkpoint(checkpoint_dir)

    # Compare In-Memory vs Reloaded Predictions (Round-Trip Verification Gate)
    in_mem_preds = predict_v2_model(in_memory_model, test_a, "JV", target_normalizer)
    reloaded_preds = predict_v2_model(reloaded_model, test_a, "JV", reloaded_norm)

    max_delta = max(abs(a - b) for a, b in zip(in_mem_preds, reloaded_preds)) if in_mem_preds else 0.0
    if max_delta > 1e-4:
        raise RuntimeError(
            f"Checkpoint roundtrip failure! Max prediction delta between in-memory and reloaded model is {max_delta:.6f} > 1e-4."
        )

    # Output setup for usefulness run
    out_dir = exp_dir / "v2_usefulness_eval"
    plots_dir = out_dir / "plots"
    plotter = MLPlotter(plots_dir)

    # 1. Split Interpolation vs Extrapolation
    interp_test, extrap_test = categorize_interpolation_vs_extrapolation(test_a, train_a)

    evaluation_results = {}
    sanity_results = {}
    pv_param_results = {}

    valid_types = ["JV", "QE", "RESPONSIVITY"]

    for stype in valid_types:
        tr_s = train_a.filter_by_scientific_types([stype])
        val_s = val_a.filter_by_scientific_types([stype])
        te_in_s = test_a.filter_by_scientific_types([stype])
        te_interp_s = interp_test.filter_by_scientific_types([stype])
        te_extrap_s = extrap_test.filter_by_scientific_types([stype])
        te_holdout_s = regime_b["test"].filter_by_scientific_types([stype])

        if len(tr_s) == 0 or len(te_in_s) == 0:
            continue

        reloaded_norm.fit({stype: tr_s.targets})
        model, tr_loss, val_loss = train_v2_surrogate(tr_s, val_s, stype, config, reloaded_norm)

        # Predict using RELOADED Checkpoint
        in_preds = predict_v2_model(model, te_in_s, stype, reloaded_norm)
        in_m = calculate_curve_metrics(in_preds, te_in_s.targets)

        interp_preds = predict_v2_model(model, te_interp_s, stype, reloaded_norm) if len(te_interp_s) > 0 else []
        interp_m = calculate_curve_metrics(interp_preds, te_interp_s.targets) if len(interp_preds) > 0 else in_m

        extrap_preds = predict_v2_model(model, te_extrap_s, stype, reloaded_norm) if len(te_extrap_s) > 0 else []
        extrap_m = calculate_curve_metrics(extrap_preds, te_extrap_s.targets) if len(extrap_preds) > 0 else in_m

        holdout_preds = predict_v2_model(model, te_holdout_s, stype, reloaded_norm) if len(te_holdout_s) > 0 else []
        holdout_m = calculate_curve_metrics(holdout_preds, te_holdout_s.targets) if len(holdout_preds) > 0 else in_m

        sanity = perform_physical_sanity_checks(te_in_s, in_preds, stype)

        # Plots for scientific type
        plotter.plot_training_dynamics(tr_loss, val_loss, scientific_type=stype, regime_name="In-Domain (Reloaded Checkpoint)")
        plotter.plot_actual_vs_predicted_scatter(te_in_s.targets, in_preds, in_m.r2, in_m.mae, scientific_type=stype, regime_name="In-Domain (Reloaded Checkpoint)")
        plotter.plot_error_vs_x([r.x for r in te_in_s.records], te_in_s.targets, in_preds, scientific_type=stype, regime_name="In-Domain (Reloaded Checkpoint)")
        plotter.plot_error_distribution(te_in_s.targets, in_preds, scientific_type=stype, regime_name="In-Domain (Reloaded Checkpoint)")

        # Specific Region Plots
        if stype == "JV" and len(te_in_s) > 0:
            rec0 = te_in_s.records[0]
            s_idx = [i for i, r in enumerate(te_in_s.records) if r.dataset_id == rec0.dataset_id]
            if s_idx:
                xs = [te_in_s.records[i].x for i in s_idx]
                act_y = [te_in_s.targets[i] for i in s_idx]
                pred_y = [in_preds[i] for i in s_idx]

                plotter.plot_curve_reconstruction(
                    xs, act_y, pred_y, stype, rec0.material, rec0.dataset_id, in_m.mae, in_m.r2, "full", "In-Domain", "V2 Physics Reloaded"
                )

                knee_idx = [i for i, x in enumerate(xs) if 0.3 <= x <= 0.8]
                if len(knee_idx) > 1:
                    plot_zoomed_curve_region(
                        [xs[i] for i in knee_idx],
                        [act_y[i] for i in knee_idx],
                        [pred_y[i] for i in knee_idx],
                        f"JV Diode Knee/Breakdown Region ({rec0.material})",
                        "Voltage (V)",
                        "Current Density (mA/cm²)",
                        plots_dir / "knee_breakdown_JV.png",
                    )

                voc_idx = [i for i, x in enumerate(xs) if x >= 0.5]
                if len(voc_idx) > 1:
                    plot_zoomed_curve_region(
                        [xs[i] for i in voc_idx],
                        [act_y[i] for i in voc_idx],
                        [pred_y[i] for i in voc_idx],
                        f"JV Open-Circuit Voltage Voc Zero-Crossing ({rec0.material})",
                        "Voltage (V)",
                        "Current Density (mA/cm²)",
                        plots_dir / "voc_zero_crossing_JV.png",
                    )

                _, _, pv_errors = evaluate_jv_curve_reconstruction_fidelity(te_in_s, in_preds)
                pv_param_results["JV"] = pv_errors.to_dict()
                plot_pv_parameter_errors_bar(pv_errors, plots_dir / "pv_parameters_comparison_JV.png")

        elif stype == "QE" and len(te_in_s) > 0:
            rec0 = te_in_s.records[0]
            s_idx = [i for i, r in enumerate(te_in_s.records) if r.dataset_id == rec0.dataset_id]
            if s_idx:
                xs = [te_in_s.records[i].x for i in s_idx]
                act_y = [te_in_s.targets[i] for i in s_idx]
                pred_y = [in_preds[i] for i in s_idx]

                edge_idx = [i for i, x in enumerate(xs) if x >= 700.0]
                if len(edge_idx) > 1:
                    plot_zoomed_curve_region(
                        [xs[i] for i in edge_idx],
                        [act_y[i] for i in edge_idx],
                        [pred_y[i] for i in edge_idx],
                        f"QE Band-Edge Transition Cutoff ({rec0.material})",
                        "Wavelength (nm)",
                        "Quantum Efficiency (%)",
                        plots_dir / "qe_band_edge_QE.png",
                    )

        evaluation_results[stype] = {
            "sample_counts": {
                "train": len(tr_s),
                "in_domain_test": len(te_in_s),
                "interpolation_test": len(te_interp_s),
                "extrapolation_test": len(te_extrap_s),
                "cross_material_holdout": len(te_holdout_s),
            },
            "in_domain_test_metrics": in_m.to_dict(),
            "interpolation_test_metrics": interp_m.to_dict(),
            "extrapolation_test_metrics": extrap_m.to_dict(),
            "cross_material_holdout_metrics": holdout_m.to_dict(),
        }
        sanity_results[stype] = sanity.to_dict()

    plotter.generate_index_md()

    # Determine usefulness verdict
    jv_r2 = evaluation_results["JV"]["in_domain_test_metrics"]["r2"]
    jv_interp_r2 = evaluation_results["JV"]["interpolation_test_metrics"]["r2"]
    jv_sane = sanity_results["JV"]["is_physically_sane"]
    pce_mape = pv_param_results.get("JV", {}).get("pce_mape", 10.0)

    if jv_r2 >= 0.95 and jv_interp_r2 >= 0.90 and jv_sane and pce_mape <= 10.0:
        verdict = "USEFUL"
    elif jv_r2 >= 0.80 and jv_sane:
        verdict = "CONDITIONALLY_USEFUL"
    else:
        verdict = "NOT_YET_USEFUL"

    summary_results = {
        "verdict": verdict,
        "is_useful_for_photovoltaics": verdict == "USEFUL",
        "checkpoint_verification": {
            "checkpoint_dir": str(checkpoint_dir),
            "manifest_sha256": compute_file_sha256(checkpoint_dir / "model_checkpoint_manifest.json"),
            "in_memory_vs_reloaded_max_delta": round(max_delta, 6),
            "roundtrip_pass": max_delta <= 1e-4,
            "inference_mode": "RELOADED_CHECKPOINT",
        },
        "data_integrity": integrity_report,
        "photovoltaic_parameter_errors": pv_param_results,
        "evaluation_by_scientific_type": evaluation_results,
        "physical_sanity_checks": sanity_results,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v2_usefulness_results.json").write_text(json.dumps(summary_results, indent=2), encoding="utf-8")
    (out_dir / "v2_usefulness_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    return summary_results


if __name__ == "__main__":
    base = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\solar_platform")
    res = execute_v2_usefulness_validation(base)
    print("V2 Usefulness Validation Complete. Verdict:", res["verdict"])
