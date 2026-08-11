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
    """Generates publication-quality PNG diagnostic plots for V1 and V2 benchmarking."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_plots: List[Dict[str, Any]] = []

    def get_axis_units(self, scientific_type: str) -> Tuple[str, str]:
        if scientific_type == "JV":
            return "Voltage (V)", "Current Density (mA/cm²)"
        elif scientific_type == "QE":
            return "Wavelength (nm)", "Quantum Efficiency (%)"
        elif scientific_type == "RESPONSIVITY":
            return "Wavelength (nm)", "Responsivity (A/W)"
        return "Native Axis (x)", "Target Value (y)"

    def plot_training_dynamics(
        self, train_losses: List[float], val_losses: List[float], scientific_type: str = "ALL", regime_name: str = "Regime A"
    ) -> Path:
        """Plot train loss and validation loss vs epoch."""
        plt.figure(figsize=(8, 5))
        epochs = list(range(1, len(train_losses) + 1))
        plt.plot(epochs, train_losses, label="Train Loss (Huber)", color="#1f77b4", linewidth=2)
        plt.plot(epochs, val_losses, label="Validation Loss (Huber)", color="#ff7f0e", linewidth=2, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Huber Loss")
        plt.title(f"Training & Validation Loss Dynamics ({scientific_type}) - {regime_name}")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()

        file_path = self.output_dir / f"training_dynamics_{scientific_type}.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        self.generated_plots.append({
            "name": f"training_dynamics_{scientific_type}",
            "regime": regime_name,
            "scientific_type": scientific_type,
            "file": str(file_path),
            "description": f"Training and validation loss progression over epochs for {scientific_type}."
        })
        return file_path

    def plot_actual_vs_predicted_scatter(
        self, actuals: List[float], predictions: List[float], r2: float, mae: float, scientific_type: str = "ALL", regime_name: str = "Regime A"
    ) -> Path:
        """Plot predicted vs actual scatter plot with y=x reference line."""
        plt.figure(figsize=(7, 6))
        x_label, y_label = self.get_axis_units(scientific_type)

        plt.scatter(actuals, predictions, alpha=0.4, color="#2ca02c", s=15, label="Data Points")

        min_v = min(min(actuals), min(predictions))
        max_v = max(max(actuals), max(predictions))
        plt.plot([min_v, max_v], [min_v, max_v], color="red", linestyle="--", linewidth=2, label="Ideal (y = x)")

        plt.xlabel(f"Actual {y_label}")
        plt.ylabel(f"Predicted {y_label}")
        plt.title(f"Predicted vs Actual Scatter ({scientific_type}) - {regime_name}\nR² = {r2:.4f} | MAE = {mae:.4f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        file_path = self.output_dir / f"actual_vs_predicted_{scientific_type}.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        self.generated_plots.append({
            "name": f"actual_vs_predicted_{scientific_type}",
            "regime": regime_name,
            "scientific_type": scientific_type,
            "file": str(file_path),
            "description": f"Scatter plot of predicted vs actual values for {scientific_type} (R²={r2:.4f})."
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
        model_name: str = "SolarSurrogate",
    ) -> Path:
        """Plot actual vs predicted curve on native x-grid."""
        plt.figure(figsize=(8, 5))
        x_label, y_label = self.get_axis_units(scientific_type)

        plt.plot(x_vals, actual_y, "o-", color="#1f77b4", label="Actual Native Target", linewidth=2, markersize=4)
        plt.plot(x_vals, pred_y, "s--", color="#d62728", label=f"{model_name} Prediction", linewidth=2, markersize=4)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"Curve Reconstruction ({tag.upper()} Case - {scientific_type})\n{material} | {dataset_id[:25]} | MAE = {mae:.4f} | R² = {r2:.4f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        file_path = self.output_dir / f"{tag}_case_curve_{scientific_type}.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        self.generated_plots.append({
            "name": f"{tag}_case_curve_{scientific_type}",
            "regime": regime_name,
            "scientific_type": scientific_type,
            "file": str(file_path),
            "description": f"{tag.capitalize()} prediction curve reconstruction for {scientific_type} dataset ({material})."
        })
        return file_path

    def plot_error_vs_x(
        self,
        x_vals: List[float],
        actuals: List[float],
        predictions: List[float],
        scientific_type: str,
        regime_name: str = "Regime A",
    ) -> Path:
        """Plot prediction error (y - y_hat) vs native x coordinate."""
        plt.figure(figsize=(8, 5))
        x_label, y_label = self.get_axis_units(scientific_type)
        errors = [a - p for a, p in zip(actuals, predictions)]

        plt.scatter(x_vals, errors, alpha=0.4, color="#9467bd", s=15, label="Residuals")
        plt.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")

        plt.xlabel(x_label)
        plt.ylabel(f"Residual Error ({y_label})")
        plt.title(f"Prediction Error vs Physical Axis ({scientific_type}) - {regime_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        file_path = self.output_dir / f"error_vs_x_{scientific_type}.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        self.generated_plots.append({
            "name": f"error_vs_x_{scientific_type}",
            "regime": regime_name,
            "scientific_type": scientific_type,
            "file": str(file_path),
            "description": f"Residual error vs physical x-axis coordinate for {scientific_type}."
        })
        return file_path

    def plot_error_distribution(
        self,
        actuals: List[float],
        predictions: List[float],
        scientific_type: str,
        regime_name: str = "Regime A",
    ) -> Path:
        """Plot histogram of prediction errors (y - y_hat)."""
        plt.figure(figsize=(8, 5))
        _, y_label = self.get_axis_units(scientific_type)
        errors = [a - p for a, p in zip(actuals, predictions)]

        plt.hist(errors, bins=40, color="#8c564b", alpha=0.7, edgecolor="black")
        plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Bias")

        plt.xlabel(f"Error ({y_label})")
        plt.ylabel("Frequency (Count)")
        plt.title(f"Error Distribution Histogram ({scientific_type}) - {regime_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        file_path = self.output_dir / f"error_distribution_{scientific_type}.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        self.generated_plots.append({
            "name": f"error_distribution_{scientific_type}",
            "regime": regime_name,
            "scientific_type": scientific_type,
            "file": str(file_path),
            "description": f"Histogram distribution of prediction errors for {scientific_type}."
        })
        return file_path

    def plot_baseline_comparison(
        self,
        metrics_mean: Dict[str, float],
        metrics_linear: Dict[str, float],
        metrics_v1: Dict[str, float],
        metrics_v2: Optional[Dict[str, float]] = None,
        scientific_type: str = "JV",
        regime_name: str = "Regime A",
    ) -> Path:
        """Bar plot comparing metrics across Mean, Linear, V1, and V2 models."""
        plt.figure(figsize=(9, 5))
        categories = ["MAE", "RMSE", "NRMSE"]
        vals_mean = [metrics_mean["mae"], metrics_mean["rmse"], metrics_mean["nrmse"]]
        vals_linear = [metrics_linear["mae"], metrics_linear["rmse"], metrics_linear["nrmse"]]
        vals_v1 = [metrics_v1["mae"], metrics_v1["rmse"], metrics_v1["nrmse"]]

        x = np.arange(len(categories))
        width = 0.2 if metrics_v2 else 0.25

        plt.bar(x - width * 1.5 if metrics_v2 else x - width, vals_mean, width, label="Baseline Mean", color="#7f7f7f")
        plt.bar(x - width * 0.5 if metrics_v2 else x, vals_linear, width, label="Baseline Linear", color="#ff7f0e")
        plt.bar(x + width * 0.5 if metrics_v2 else x + width, vals_v1, width, label="V1 Dedicated MLP", color="#1f77b4")

        if metrics_v2:
            vals_v2 = [metrics_v2["mae"], metrics_v2["rmse"], metrics_v2["nrmse"]]
            plt.bar(x + width * 1.5, vals_v2, width, label="V2 Physics Multi-Head", color="#2ca02c")

        plt.ylabel("Metric Value")
        plt.title(f"Model Architecture Performance Comparison ({scientific_type}) - {regime_name}")
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        file_path = self.output_dir / f"baseline_comparison_{scientific_type}.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        self.generated_plots.append({
            "name": f"baseline_comparison_{scientific_type}",
            "regime": regime_name,
            "scientific_type": scientific_type,
            "file": str(file_path),
            "description": f"Performance metric comparison of models on {scientific_type}."
        })
        return file_path

    def plot_regime_comparison_bar(
        self,
        metrics_a: Dict[str, float],
        metrics_b: Dict[str, float],
        scientific_type: str = "JV",
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

        plt.ylabel("Metric Value")
        plt.title(f"Model Generalization: Regime A vs Regime B ({scientific_type})")
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        file_path = self.output_dir / f"regime_comparison_{scientific_type}.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        self.generated_plots.append({
            "name": f"regime_comparison_{scientific_type}",
            "regime": "Regime A vs B",
            "scientific_type": scientific_type,
            "file": str(file_path),
            "description": f"Bar plot comparing MAE, RMSE, and NRMSE across Regime A and Regime B for {scientific_type}."
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
