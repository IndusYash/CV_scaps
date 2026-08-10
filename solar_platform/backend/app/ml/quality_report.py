"""
app.ml.quality_report
---------------------
ML Dataset Quality Report and Leakage-Proof Experiment Grouping Diagnostics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.canonical.model import CanonicalWorkbook, ScientificDataset
from app.ml.builder import MLDatasetBuilder
from app.ml.model import MLCurveRecord, MLDatasetCollection, MLScalarRecord


@dataclass
class MLQualityReport:
    """Comprehensive ML Dataset Quality Report."""
    total_canonical_datasets: int = 0
    ml_ready_datasets_count: int = 0
    excluded_datasets_count: int = 0
    curve_datasets_count: int = 0
    scalar_datasets_count: int = 0
    total_traces_count: int = 0
    total_numeric_points_count: int = 0
    total_scalar_records_count: int = 0

    scientific_type_distribution: Dict[str, int] = field(default_factory=dict)
    material_distribution: Dict[str, int] = field(default_factory=dict)
    validation_status_distribution: Dict[str, int] = field(default_factory=dict)
    parameter_distributions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    x_grid_stats: Dict[str, Any] = field(default_factory=dict)
    missing_value_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_dataset_count: int = 0
    experiment_group_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_canonical_datasets": self.total_canonical_datasets,
            "ml_ready_datasets_count": self.ml_ready_datasets_count,
            "excluded_datasets_count": self.excluded_datasets_count,
            "curve_datasets_count": self.curve_datasets_count,
            "scalar_datasets_count": self.scalar_datasets_count,
            "total_traces_count": self.total_traces_count,
            "total_numeric_points_count": self.total_numeric_points_count,
            "total_scalar_records_count": self.total_scalar_records_count,
            "scientific_type_distribution": self.scientific_type_distribution,
            "material_distribution": self.material_distribution,
            "validation_status_distribution": self.validation_status_distribution,
            "parameter_distributions": self.parameter_distributions,
            "x_grid_stats": self.x_grid_stats,
            "missing_value_counts": self.missing_value_counts,
            "duplicate_dataset_count": self.duplicate_dataset_count,
            "experiment_group_count": self.experiment_group_count,
        }


def generate_ml_quality_report(
    canonical_workbooks: List[CanonicalWorkbook],
    builder: Optional[MLDatasetBuilder] = None,
) -> MLQualityReport:
    """Generate a comprehensive ML quality report across all canonical workbooks."""
    if builder is None:
        builder = MLDatasetBuilder(include_warnings=True, include_uncertain=False, include_unresolved=False)

    report = MLQualityReport()

    all_datasets: List[ScientificDataset] = []
    for wb in canonical_workbooks:
        all_datasets.extend(wb.datasets)

    report.total_canonical_datasets = len(all_datasets)

    sci_counter = Counter()
    mat_counter = Counter()
    status_counter = Counter()
    param_values = defaultdict(list)
    x_lengths = []
    x_mins = []
    x_maxs = []
    missing_counts = Counter()
    dataset_hashes = set()
    dup_count = 0
    group_set = set()

    for ds in all_datasets:
        sci_counter[ds.scientific_type] += 1
        mat_name = ds.workbook_name.replace(".xlsx", "")
        mat_counter[mat_name] += 1
        status_counter[ds.validation_status] += 1

        is_inc = builder.is_dataset_included(ds)
        if is_inc:
            report.ml_ready_datasets_count += 1
            grp = f"grp_{mat_name}_{ds.sheet_name}_{ds.parent_region_id or ds.region_id}"
            group_set.add(grp)
        else:
            report.excluded_datasets_count += 1

        ds_hash = f"{ds.workbook_name}_{ds.sheet_name}_{ds.region_id}_{len(ds.x_values)}_{len(ds.traces)}"
        if ds_hash in dataset_hashes:
            dup_count += 1
        else:
            dataset_hashes.add(ds_hash)

        if ds.traces and ds.x_values:
            report.curve_datasets_count += 1
            report.total_traces_count += len(ds.traces)
            report.total_numeric_points_count += len(ds.x_values) * len(ds.traces)

            x_lengths.append(len(ds.x_values))
            if ds.x_values:
                x_mins.append(min(ds.x_values))
                x_maxs.append(max(ds.x_values))

            for trace in ds.traces:
                if trace.parameter and trace.parameter.name and trace.parameter.value is not None:
                    param_values[trace.parameter.name].append(trace.parameter.value)
                elif not trace.parameter:
                    missing_counts["missing_trace_parameter"] += 1

        if ds.scalar_records:
            report.scalar_datasets_count += 1
            report.total_scalar_records_count += len(ds.scalar_records)
            for rec in ds.scalar_records:
                if rec.parameter and rec.parameter.name and rec.parameter.value is not None:
                    param_values[rec.parameter.name].append(rec.parameter.value)

    report.scientific_type_distribution = dict(sci_counter)
    report.material_distribution = dict(mat_counter)
    report.validation_status_distribution = dict(status_counter)
    report.duplicate_dataset_count = dup_count
    report.experiment_group_count = len(group_set)

    param_stats = {}
    for p_name, vals in param_values.items():
        num_vals = [v for v in vals if isinstance(v, (int, float))]
        if num_vals:
            param_stats[p_name] = {
                "count": len(num_vals),
                "min": min(num_vals),
                "max": max(num_vals),
                "distinct_count": len(set(num_vals)),
            }
        else:
            param_stats[p_name] = {"count": len(vals), "distinct_count": len(set(vals))}
    report.parameter_distributions = param_stats

    if x_lengths:
        report.x_grid_stats = {
            "count": len(x_lengths),
            "min_length": min(x_lengths),
            "max_length": max(x_lengths),
            "avg_length": sum(x_lengths) / len(x_lengths),
            "overall_min_x": min(x_mins) if x_mins else None,
            "overall_max_x": max(x_maxs) if x_maxs else None,
        }

    report.missing_value_counts = dict(missing_counts)
    return report


def get_experiment_grouping_diagnostics(collection: MLDatasetCollection) -> Dict[str, Any]:
    """
    Produce leakage-proof experiment grouping diagnostics for train/test split design.
    Shows how traces and scalar records are grouped into isolated experiment sweeps.
    """
    curve_groups = defaultdict(list)
    for r in collection.curve_records:
        curve_groups[r.group_id].append(r.trace_id)

    scalar_groups = defaultdict(list)
    for s in collection.scalar_records:
        scalar_groups[s.group_id].append(s.record_id)

    group_summary = {}
    for grp, traces in curve_groups.items():
        unique_traces = len(set(traces))
        group_summary[grp] = {
            "type": "curve_sweep",
            "point_count": len(traces),
            "unique_traces": unique_traces,
        }

    for grp, recs in scalar_groups.items():
        group_summary[grp] = {
            "type": "scalar_sweep",
            "record_count": len(recs),
        }

    return {
        "total_experiment_groups": len(group_summary),
        "total_curve_points": len(collection.curve_records),
        "total_scalar_records": len(collection.scalar_records),
        "groups": group_summary,
    }
