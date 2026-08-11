"""
app.ml.modeling.plotter
-----------------------
Publication-quality Matplotlib Plot Generator for ML Model Benchmarking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class MLPlotter:
    """Generates all required publication-quality PNG diagnostic plots for V1 benchmarking."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_plots: List[Dict[str, Any]] = []

    def plot_training_dynamics(
        self, train_losses: List[float], val_losses: List[float], regime_name: str = "Regime A"
    ) -> Path:
        """Plot train loss and validation loss vs epoch."""
        plt.figure(figsize=(8, 5))
        epochs = list(range(1, len(train_losses) + 1))
        plt.plot(epochs, train_losses, label="Train Loss (Huber)", color="#1f77b4", linewidth=2)
        plt.plot(epochs, val_losses, label="Validation Loss (Huber)", color="#ff7f0e", linewidth=2, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Huber Loss")
        plt.title(f"V1 Training & Validation Loss Dynamics ({regime_name})")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()

        file_path = self.output_dir / "training_dynamics.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        self.generated_plots.append({
            "name": "training_dynamics",
            "regime": regime_name,
            "scientific_type": "ALL",
            "file": str(file_path),
            "description": "Training and validation loss progression over epochs."
        })
        return file_path

    def plot_actual_vs_predicted_scatter(
        self, actuals: List[float], predictions: List[float], r2: float, mae: float, regime_name: str = "Regime A"
    ) -> Path:
        """Plot predicted vs actual scatter plot with y=x reference line."""
        plt.figure(figsize=(7, 6))
        plt.scatter(actuals, predictions, alpha=0.4, color="#2ca02c", s=15, label="Data Points")

        min_v = min(min(actuals), min(predictions))
        max_v = max(max(actuals), max(predictions))
        plt.plot([min_v, max_v], [min_v, max_v], color="red", linestyle="--", linewidth=2, label="Ideal (y = x)")

        plt.xlabel("Actual Value")
        plt.ylabel("Predicted Value")
        plt.title(f"Predicted vs Actual Scatter ({regime_name})\nR² = {r2:.4f} | MAE = {mae:.4f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        file_path = self.output_dir / "predicted_vs_actual.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        self.generated_plots.append({
            "name": "predicted_vs_actual",
            "regime": regime_name,
            "scientific_type": "ALL",
            "file": str(file_path),
            "description": f"Scatter plot of predicted vs actual values with y=x line (R²={r2:.4f})."
        })
        return file_path

    def plot_curve_reconstruction(
        self,
        x_vals: List[float],
        actual_y: List[float],
        pred_y: List[float],
        scientific_type: str,
        material: str,
        dataset_id: str,
        mae: float,
        r2: float,
        tag: str = "best",
        regime_name: str = "Regime A",
    ) -> Path:
        """Plot actual vs predicted curve on native x-grid."""
        plt.figure(figsize=(8, 5))

        x_unit = "V" if scientific_type == "JV" else "nm"
        y_unit = "mA/cm²" if scientific_type == "JV" else "%"

        plt.plot(x_vals, actual_y, "o-", color="#1f77b4", label="Actual Native Target", linewidth=2, markersize=4)
        plt.plot(x_vals, pred_y, "s--", color="#d62728", label="SolarSurrogateMLP Prediction", linewidth=2, markersize=4)

        plt.xlabel(f"Native Axis ({x_unit})")
        plt.ylabel(f"Scientific Target ({y_unit})")
        plt.title(f"Curve Reconstruction ({tag.upper()} Case - {scientific_type})\n{material} | {dataset_id[:25]} | MAE = {mae:.4f} | R² = {r2:.4f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        file_path = self.output_dir / f"{tag}_case_curve.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        self.generated_plots.append({
            "name": f"{tag}_case_curve",
            "regime": regime_name,
            "scientific_type": scientific_type,
            "file": str(file_path),
            "description": f"{tag.capitalize()} prediction curve reconstruction for {scientific_type} dataset."
        })
        return file_path

    def plot_baseline_comparison(
        self,
        x_vals: List[float],
        actual_y: List[float],
        mean_y: List[float],
        linear_y: List[float],
        mlp_y: List[float],
        scientific_type: str,
        regime_name: str = "Regime A",
    ) -> Path:
        """Plot baseline comparison curves against neural MLP surrogate."""
        plt.figure(figsize=(9, 5))
        x_unit = "V" if scientific_type == "JV" else "nm"
        y_unit = "mA/cm²" if scientific_type == "JV" else "%"

        plt.plot(x_vals, actual_y, "k-", label="Actual Target", linewidth=2.5)
        plt.plot(x_vals, mean_y, ":", color="#7f7f7f", label="Baseline Mean Model", linewidth=2)
        plt.plot(x_vals, linear_y, "-.", color="#ff7f0e", label="Baseline Linear Model", linewidth=2)
        plt.plot(x_vals, mlp_y, "--", color="#2ca02c", label="SolarSurrogateMLP (Neural)", linewidth=2)

        plt.xlabel(f"Native Axis ({x_unit})")
        plt.ylabel(f"Scientific Target ({y_unit})")
        plt.title(f"Baseline vs Neural Model Comparison ({scientific_type}) - {regime_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        file_path = self.output_dir / "baseline_comparison.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        self.generated_plots.append({
            "name": "baseline_comparison",
            "regime": regime_name,
            "scientific_type": scientific_type,
            "file": str(file_path),
            "description": "Comparison of Mean, Linear, and Neural MLP models against actual curve."
        })
        return file_path

    def plot_regime_comparison_bar(
        self,
        metrics_a: Dict[str, float],
        metrics_b: Dict[str, float],
    ) -> Path:
        """Bar plot comparing Regime A (In-Domain) vs Regime B (Material Holdout) metrics."""
        plt.figure(figsize=(8, 5))
        categories = ["MAE", "RMSE", "NRMSE"]
        vals_a = [metrics_a["mae"], metrics_a["rmse"], metrics_a["nrmse"]]
        vals_b = [metrics_b["mae"], metrics_b["rmse"], metrics_b["nrmse"]]

        x = np.arange(len(categories))
        width = 0.35

        plt.bar(x - width / 2, vals_a, width, label="Regime A (In-Domain)", color="#1f77b4")
        plt.bar(x + width / 2, vals_b, width, label="Regime B (Cross-Material Holdout)", color="#d62728")

        plt.ylabel("Error Metric Value")
        plt.title("Model Generalization: Regime A vs Regime B Error Comparison")
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        file_path = self.output_dir / "regime_comparison.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        self.generated_plots.append({
            "name": "regime_comparison",
            "regime": "Regime A vs B",
            "scientific_type": "ALL",
            "file": str(file_path),
            "description": "Bar plot comparing MAE, RMSE, and NRMSE across Regime A and Regime B."
        })
        return file_path

    def generate_index_md(self) -> str:
        """Generate markdown index listing all generated plot artifacts."""
        lines = [
            "# Diagnostic Plot Index",
            "",
            "| Plot Name | Regime | Scientific Type | Description | Image File |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for p in self.generated_plots:
            rel_file = Path(p["file"]).name
            lines.append(
                f"| `{p['name']}` | {p['regime']} | {p['scientific_type']} | {p['description']} | [`{rel_file}`]({rel_file}) |"
            )

        index_md = "\n".join(lines)
        index_path = self.output_dir / "index.md"
        index_path.write_text(index_md, encoding="utf-8")
        return index_md
