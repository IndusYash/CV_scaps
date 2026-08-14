"""
app.ml.modeling.run_v2_convergence_study
-----------------------------------------
Controlled Training-Convergence Study for V2 Solar Cell Curve Surrogate Architecture.

Evaluates 3 Configurations:
  - Experiment 1: Baseline (max_epochs=100, patience=15)
  - Experiment 2: Increased Patience (max_epochs=150, patience=50)
  - Experiment 3: Fixed Epoch Budget (max_epochs=150, patience=999, checkpoint via best validation loss)

Tracks specifically for JV:
  1. predicted J at V=0
  2. predicted Jsc
  3. predicted Voc
  4. predicted FF
  5. predicted PCE
  6. current-density prediction range (min, max, mean, std)
  7. validation loss vs epoch
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

import torch
import torch.nn as nn
import torch.optim as optim

from app.canonical.converter import build_canonical_workbook
from app.ml.curation_pipeline import MLCurationPipeline
from app.ml.modeling.config import TrainingConfig
from app.ml.modeling.dataset import CurvePointDataset
from app.ml.modeling.evaluation import EvaluationRunner
from app.ml.modeling.metrics import calculate_curve_metrics
from app.ml.modeling.models import BaselineLinearModel, BaselineMeanModel
from app.ml.modeling.plotter import MLPlotter
from app.ml.modeling.pv_metrics import (
    PVParameterErrors,
    PVParameters,
    calculate_pv_parameter_errors,
    extract_pv_parameters,
)
from app.ml.modeling.run_v1_benchmarks import parse_and_convert_workbook, run_data_integrity_check
from app.ml.modeling.usefulness_evaluator import (
    categorize_interpolation_vs_extrapolation,
    perform_physical_sanity_checks,
)
from app.ml.modeling.v2_checkpoint import (
    compute_file_sha256,
    count_model_parameters,
    load_v2_checkpoint,
    save_v2_checkpoint,
)
from app.ml.modeling.v2_models import (
    TORCH_AVAILABLE,
    SolarV2MultiHeadSurrogate,
    TargetNormalizer,
)


def train_v2_surrogate_with_diagnostics(
    train_ds: CurvePointDataset,
    val_ds: CurvePointDataset,
    stype: str,
    config: TrainingConfig,
    target_normalizer: TargetNormalizer,
    patience: int,
    max_epochs: int,
) -> Tuple[Any, List[float], List[float], int, float]:
    """
    Train SolarV2MultiHeadSurrogate tracking detailed epoch-by-epoch loss
    and returning (model, train_losses, val_losses, best_epoch, best_val_loss).
    Model weights loaded are strictly from the epoch with lowest validation loss.
    """
    torch.manual_seed(config.seed)
    model = SolarV2MultiHeadSurrogate(
        param_dim=14,
        num_fourier_freqs=8,
        hidden_dims=[128, 128],
        dropout_rate=0.05,
    )

    X_train = torch.tensor(train_ds.features, dtype=torch.float32)
    y_norm_train = [target_normalizer.normalize(y, stype) for y in train_ds.targets]
    y_train = torch.tensor(y_norm_train, dtype=torch.float32).unsqueeze(1)

    X_val = torch.tensor(val_ds.features, dtype=torch.float32) if len(val_ds) > 0 else X_train
    y_norm_val = [target_normalizer.normalize(y, stype) for y in val_ds.targets] if len(val_ds) > 0 else y_norm_train
    y_val = torch.tensor(y_norm_val, dtype=torch.float32).unsqueeze(1)

    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 1
    best_weights = None
    patience_counter = 0

    batch_size = config.batch_size
    n_samples = len(train_ds)

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            out = model(batch_x, stype=stype)
            loss = criterion(out, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(1, n_batches)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_out = model(X_val, stype=stype)
            val_loss = criterion(val_out, y_val).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()

    return model, train_losses, val_losses, best_epoch, best_val_loss


def predict_v2(model: Any, dataset: CurvePointDataset, stype: str, target_normalizer: TargetNormalizer) -> List[float]:
    """Predict and denormalize outputs."""
    if model is None or len(dataset) == 0:
        raise RuntimeError("Model is None or dataset empty")
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(dataset.features, dtype=torch.float32)
        out_norm = model(X_t, stype=stype).squeeze(1).tolist()
        return [target_normalizer.denormalize(y_n, stype) for y_n in out_norm]


def analyze_jv_curves_detailed(
    dataset: CurvePointDataset, predictions: List[float]
) -> Dict[str, Any]:
    """
    Detailed diagnostic analysis of predicted JV curves:
    - J at V=0
    - Jsc, Voc, FF, PCE
    - Current density range
    """
    groups: Dict[str, Dict] = defaultdict(lambda: {"x": [], "act": [], "pred": []})
    for i, (rec, pred) in enumerate(zip(dataset.records, predictions)):
        gid = rec.dataset_id
        groups[gid]["x"].append(rec.x)
        groups[gid]["act"].append(dataset.targets[i])
        groups[gid]["pred"].append(pred)

    j_at_v0_acts = []
    j_at_v0_preds = []
    act_pvs = []
    pred_pvs = []

    for gid, data in groups.items():
        v_arr = np.array(data["x"])
        act_j = np.array(data["act"])
        pred_j = np.array(data["pred"])

        # J at V=0
        if 0.0 >= min(v_arr) and 0.0 <= max(v_arr):
            sort_idx = np.argsort(v_arr)
            v_s = v_arr[sort_idx]
            act_j0 = float(np.interp(0.0, v_s, act_j[sort_idx]))
            pred_j0 = float(np.interp(0.0, v_s, pred_j[sort_idx]))
            j_at_v0_acts.append(act_j0)
            j_at_v0_preds.append(pred_j0)

        act_pv = extract_pv_parameters(data["x"], data["act"])
        pred_pv = extract_pv_parameters(data["x"], data["pred"])
        if act_pv and pred_pv:
            act_pvs.append(act_pv)
            pred_pvs.append(pred_pv)

    pv_errors = calculate_pv_parameter_errors(act_pvs, pred_pvs) if act_pvs else None

    pred_arr = np.array(predictions)
    act_arr = np.array(dataset.targets)

    return {
        "j_at_v0": {
            "actual_mean": float(np.mean(j_at_v0_acts)) if j_at_v0_acts else None,
            "predicted_mean": float(np.mean(j_at_v0_preds)) if j_at_v0_preds else None,
            "actual_values": [round(v, 4) for v in j_at_v0_acts],
            "predicted_values": [round(v, 4) for v in j_at_v0_preds],
        },
        "current_density_range": {
            "pred_min": float(np.min(pred_arr)),
            "pred_max": float(np.max(pred_arr)),
            "pred_mean": float(np.mean(pred_arr)),
            "pred_std": float(np.std(pred_arr)),
            "act_min": float(np.min(act_arr)),
            "act_max": float(np.max(act_arr)),
            "act_mean": float(np.mean(act_arr)),
            "act_std": float(np.std(act_arr)),
        },
        "pv_parameters": {
            "num_curves": len(act_pvs),
            "actual_jsc_mean": float(np.mean([p.j_sc for p in act_pvs])) if act_pvs else None,
            "predicted_jsc_mean": float(np.mean([p.j_sc for p in pred_pvs])) if pred_pvs else None,
            "actual_voc_mean": float(np.mean([p.v_oc for p in act_pvs])) if act_pvs else None,
            "predicted_voc_mean": float(np.mean([p.v_oc for p in pred_pvs])) if pred_pvs else None,
            "actual_ff_mean": float(np.mean([p.ff for p in act_pvs])) if act_pvs else None,
            "predicted_ff_mean": float(np.mean([p.ff for p in pred_pvs])) if pred_pvs else None,
            "actual_pce_mean": float(np.mean([p.pce for p in act_pvs])) if act_pvs else None,
            "predicted_pce_mean": float(np.mean([p.pce for p in pred_pvs])) if pred_pvs else None,
            "errors": pv_errors.to_dict() if pv_errors else {},
        },
    }


def run_convergence_study(base_dir: Path) -> Dict[str, Any]:
    """Execute the full 3-experiment convergence study."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for convergence study.")

    wb_files = ["Ca3PI3.xlsx", "Ca3PBr3.xlsx", "Ca3AsBr3.xlsx", "Data file.xlsx", "Data File (1).xlsx"]
    canonical_wbs = []
    for fname in wb_files:
        fpath = base_dir / fname
        if fpath.exists():
            canonical_wbs.append(parse_and_convert_workbook(fpath))

    curation_pipeline = MLCurationPipeline(seed=42)
    curation_res = curation_pipeline.execute(canonical_wbs)
    integrity = run_data_integrity_check(curation_res)

    config = TrainingConfig()
    runner = EvaluationRunner(curation_res, config)

    # Datasets
    regime_a = runner.prepare_regime_a_datasets()
    train_a, val_a, test_a = regime_a["train"], regime_a["val"], regime_a["test"]

    regime_b = runner.prepare_regime_b_datasets()
    train_b, test_b = regime_b["train"], regime_b["test"]

    base_exp_dir = Path(__file__).parent / "experiments" / "convergence_study"
    plots_dir = base_exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    experiments_def = [
        {
            "id": "exp1_baseline",
            "name": "Experiment 1 — Current Baseline",
            "max_epochs": 100,
            "patience": 15,
            "desc": "Baseline config: max_epochs=100, early_stopping_patience=15",
        },
        {
            "id": "exp2_increased_patience",
            "name": "Experiment 2 — Increased Patience",
            "max_epochs": 150,
            "patience": 50,
            "desc": "Extended patience: max_epochs=150, early_stopping_patience=50",
        },
        {
            "id": "exp3_fixed_budget",
            "name": "Experiment 3 — Fixed Epoch Budget (150 Epochs)",
            "max_epochs": 150,
            "patience": 999,
            "desc": "Full fixed budget: max_epochs=150, no early stopping, best val checkpoint",
        },
    ]

    study_results: Dict[str, Any] = {
        "study_description": "Controlled V2 Training Convergence Study across 3 patience/epoch configurations",
        "data_integrity": integrity,
        "experiments": {},
    }

    all_exp_loss_histories: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for exp in experiments_def:
        exp_id = exp["id"]
        exp_name = exp["name"]
        max_ep = exp["max_epochs"]
        pat = exp["patience"]
        print(f"\n{'='*70}\nSTARTING: {exp_name}\nConfig: max_epochs={max_ep}, patience={pat}\n{'='*70}")

        exp_ckpt_dir = base_exp_dir / exp_id / "checkpoint"
        exp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        exp_models: Dict[str, Any] = {}
        exp_norms: Dict[str, TargetNormalizer] = {}
        exp_train_stats: Dict[str, Any] = {}
        loss_histories: Dict[str, Dict[str, List[float]]] = {}

        # ------------------------------------------------------------------
        # 1. Train models for Regime A (JV, QE, RESPONSIVITY)
        # ------------------------------------------------------------------
        for stype in ["JV", "QE", "RESPONSIVITY"]:
            tr = train_a.filter_by_scientific_types([stype])
            va = val_a.filter_by_scientific_types([stype])
            if len(tr) == 0:
                continue

            norm = TargetNormalizer()
            norm.fit({stype: tr.targets})
            exp_norms[stype] = norm

            model, tr_losses, val_losses, best_ep, best_val_l = train_v2_surrogate_with_diagnostics(
                tr, va, stype, config, norm, patience=pat, max_epochs=max_ep
            )
            exp_models[stype] = model
            loss_histories[stype] = {"train": tr_losses, "val": val_losses}
            exp_train_stats[stype] = {
                "best_val_epoch": best_ep,
                "final_epoch": len(tr_losses),
                "best_val_loss": best_val_l,
                "final_train_loss": tr_losses[-1],
                "final_val_loss": val_losses[-1],
            }
            print(f"  [{stype}] Best Val Epoch: {best_ep}/{len(tr_losses)} | Best Val Loss: {best_val_l:.5f} | Final Train Loss: {tr_losses[-1]:.5f}")

        all_exp_loss_histories[exp_id] = loss_histories

        # Save Primary Checkpoint (JV model + combined normalizers)
        combined_norm = TargetNormalizer()
        for stype, norm in exp_norms.items():
            combined_norm.means[stype] = norm.means[stype]
            combined_norm.stds[stype] = norm.stds[stype]

        jv_model = exp_models.get("JV")
        save_v2_checkpoint(
            model=jv_model,
            preprocessor=train_a.preprocessor,
            normalizer=combined_norm,
            config=config,
            manifest_entries_count=len(curation_res.manifest_entries),
            save_dir=exp_ckpt_dir,
        )
        manifest_sha = compute_file_sha256(exp_ckpt_dir / "model_checkpoint_manifest.json")

        # Reload Checkpoint and verify roundtrip
        reloaded_model, reloaded_prep, reloaded_norm, _, manifest_meta = load_v2_checkpoint(exp_ckpt_dir)
        jv_test = test_a.filter_by_scientific_types(["JV"])
        reloaded_norm.means["JV"] = combined_norm.means["JV"]
        reloaded_norm.stds["JV"] = combined_norm.stds["JV"]
        preds_inmem = predict_v2(jv_model, jv_test, "JV", combined_norm)
        preds_reload = predict_v2(reloaded_model, jv_test, "JV", reloaded_norm)
        max_delta = max(abs(a - b) for a, b in zip(preds_inmem, preds_reload))
        if max_delta > 1e-4:
            raise RuntimeError(f"Roundtrip failed for {exp_id}: delta {max_delta}")

        # ------------------------------------------------------------------
        # 2. Evaluate Regime A on Test Set (strictly untouched during training)
        # ------------------------------------------------------------------
        regime_a_eval: Dict[str, Any] = {}
        jv_diagnostics = None

        for stype in ["JV", "QE", "RESPONSIVITY"]:
            tr = train_a.filter_by_scientific_types([stype])
            va = val_a.filter_by_scientific_types([stype])
            te = test_a.filter_by_scientific_types([stype])
            if len(tr) == 0 or len(te) == 0:
                continue

            model = exp_models[stype]
            norm = exp_norms[stype]
            preds = predict_v2(model, te, stype, norm)
            m = calculate_curve_metrics(preds, te.targets)

            # Baselines
            bm = BaselineMeanModel(); bm.fit(tr.targets)
            bl = BaselineLinearModel(); bl.fit(tr.features, tr.targets)
            mean_m = calculate_curve_metrics(bm.predict(te.features), te.targets)
            lin_m = calculate_curve_metrics(bl.predict(te.features), te.targets)

            interp_te, extrap_te = categorize_interpolation_vs_extrapolation(te, tr)
            interp_preds = predict_v2(model, interp_te, stype, norm) if len(interp_te) > 0 else []
            interp_m = calculate_curve_metrics(interp_preds, interp_te.targets) if interp_preds else m

            sanity = perform_physical_sanity_checks(te, preds, stype)

            regime_a_eval[stype] = {
                "n_test": len(te),
                "metrics": m.to_dict(),
                "interpolation_metrics": interp_m.to_dict(),
                "baseline_mean_r2": mean_m.r2,
                "baseline_linear_r2": lin_m.r2,
                "physical_sanity": sanity.to_dict(),
            }

            if stype == "JV":
                jv_diagnostics = analyze_jv_curves_detailed(te, preds)

        # ------------------------------------------------------------------
        # 3. Evaluate Regime B (Cross-Material Ca3AsBr3)
        # ------------------------------------------------------------------
        regime_b_eval: Dict[str, Any] = {}
        for stype in ["JV", "QE"]:
            tr = train_b.filter_by_scientific_types([stype])
            te = test_b.filter_by_scientific_types([stype])
            if len(tr) == 0 or len(te) == 0:
                continue

            norm_b = TargetNormalizer()
            norm_b.fit({stype: tr.targets})
            model_b, _, _, _, _ = train_v2_surrogate_with_diagnostics(
                tr, te, stype, config, norm_b, patience=pat, max_epochs=max_ep
            )
            preds_b = predict_v2(model_b, te, stype, norm_b)
            m_b = calculate_curve_metrics(preds_b, te.targets)
            regime_b_eval[stype] = {
                "n_test": len(te),
                "metrics": m_b.to_dict(),
            }

        # Store experiment summary
        study_results["experiments"][exp_id] = {
            "name": exp_name,
            "description": exp["desc"],
            "checkpoint_manifest_sha256": manifest_sha,
            "training_stats": exp_train_stats,
            "regime_a_evaluation": regime_a_eval,
            "regime_b_evaluation": regime_b_eval,
            "jv_detailed_diagnostics": jv_diagnostics,
        }

    # ----------------------------------------------------------------------
    # 4. Generate Comparative Diagnostic Plots
    # ----------------------------------------------------------------------
    # Plot 1: Loss curves comparison for JV, QE, RESP
    for stype in ["JV", "QE", "RESPONSIVITY"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for exp in experiments_def:
            eid = exp["id"]
            ename = exp["name"]
            hist = all_exp_loss_histories.get(eid, {}).get(stype, {})
            if hist:
                val_losses = hist["val"]
                train_losses = hist["train"]
                ax.plot(range(1, len(val_losses) + 1), val_losses, label=f"{ename} (Val Loss)", linewidth=2.0)
                ax.plot(range(1, len(train_losses) + 1), train_losses, "--", label=f"{ename} (Train Loss)", alpha=0.5)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Smooth L1 Loss (Normalized Scale)")
        ax.set_title(f"Training & Validation Loss Convergence — {stype}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"convergence_loss_{stype}.png", dpi=200)
        plt.close(fig)

    # Plot 2: Actual vs Predicted Representative JV curves across 3 experiments
    te_jv = test_a.filter_by_scientific_types(["JV"])
    groups_jv: Dict[str, Dict] = defaultdict(lambda: {"x": [], "act": []})
    for r in te_jv.records:
        groups_jv[r.dataset_id]["x"].append(r.x)
        groups_jv[r.dataset_id]["act"].append(r.y)

    sample_gids = list(groups_jv.keys())[:3]
    for gid in sample_gids:
        x_raw = groups_jv[gid]["x"]
        act_y = groups_jv[gid]["act"]
        sort_i = np.argsort(x_raw)
        x_s = [x_raw[i] for i in sort_i]
        act_s = [act_y[i] for i in sort_i]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_s, act_s, "k-o", label="Ground Truth (SCAPS)", linewidth=2.5, markersize=5)

        for exp in experiments_def:
            eid = exp["id"]
            ename = exp["name"]
            # reload model and predict
            ckpt_p = base_exp_dir / eid / "checkpoint"
            m_reload, _, n_reload, _, _ = load_v2_checkpoint(ckpt_p)
            sub_ds = CurvePointDataset([r for r in te_jv.records if r.dataset_id == gid], train_a.preprocessor)
            preds_sub = predict_v2(m_reload, sub_ds, "JV", n_reload)
            preds_s = [preds_sub[i] for i in sort_i]
            r2_sub = calculate_curve_metrics(preds_s, act_s).r2
            ax.plot(x_s, preds_s, "--s", label=f"{ename} (R²={r2_sub:.2f})", linewidth=1.8, markersize=4)

        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current Density (mA/cm²)")
        safe_gid = gid.replace("/", "_").replace(" ", "_")[:30]
        ax.set_title(f"JV Curve Convergence Comparison: {safe_gid}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"jv_curve_convergence_{safe_gid}.png", dpi=200)
        plt.close(fig)

    # Save JSON results
    out_json = base_exp_dir / "convergence_study_results.json"
    out_json.write_text(json.dumps(study_results, indent=2), encoding="utf-8")
    print(f"\nConvergence study complete! Results written to: {out_json}")

    return study_results


if __name__ == "__main__":
    base = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\solar_platform")
    res = run_convergence_study(base)
