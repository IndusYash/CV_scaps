"""
app.ml.test_audit
-----------------
Unit & Integration Test Suite for ML Dataset Audit & Split Validation Layer (app.ml.audit).
"""

import unittest
from pathlib import Path

from app.canonical.model import (
    CanonicalParameter,
    CanonicalScalarRecord,
    CanonicalTrace,
    CanonicalWorkbook,
    CellRefDTO,
    ScientificDataset,
)
from app.ml.audit import (
    audit_duplicates,
    audit_native_x_grids,
    audit_parameter_distributions,
    audit_point_level_invariants,
    perform_leakage_free_group_split,
    run_comprehensive_ml_audit,
)
from app.ml.builder import MLDatasetBuilder
from app.ml.model import MLCurveRecord, MLDatasetCollection, MLScalarRecord


class TestMLDatasetAudit(unittest.TestCase):
    """Test suite covering ML audit, point-level checks, duplicate detection, and split isolation."""

    def test_point_level_provenance_and_length_invariants(self):
        """Verify len(x values) == len(x provenance) and len(y values) == len(y provenance)."""
        x_prov = [CellRefDTO(sheet_name="S1", row=1, col=0, raw_value=0.0), CellRefDTO(sheet_name="S1", row=2, col=0, raw_value=1.0)]
        y_prov = [CellRefDTO(sheet_name="S1", row=1, col=1, raw_value=10.0), CellRefDTO(sheet_name="S1", row=2, col=1, raw_value=5.0)]

        ds = ScientificDataset(
            dataset_id="ds_audit_1",
            workbook_name="Wb.xlsx",
            sheet_name="S1",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            x_values=[0.0, 1.0],
            x_provenance=x_prov,
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[10.0, 5.0], y_provenance=y_prov)],
        )

        wb = CanonicalWorkbook(workbook_name="Wb.xlsx", datasets=[ds])
        builder = MLDatasetBuilder()
        col = builder.build_from_canonical_workbook(wb)

        res = audit_point_level_invariants([wb], col)
        self.assertEqual(res.total_datasets_checked, 1)
        self.assertEqual(res.passed_x_length_check, 1)
        self.assertEqual(res.passed_y_length_check, 1)
        self.assertEqual(len(res.discrepancies), 0)

    def test_duplicate_dataset_detection(self):
        """Detect exact and same-experiment duplicate datasets based on content signatures."""
        ds1 = ScientificDataset(
            dataset_id="ds_dup_1",
            workbook_name="Wb1.xlsx",
            sheet_name="JV",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            axis_name="Voltage",
            target_name="CurrentDensity",
            x_values=[0.0, 0.5],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[20.0, 10.0])],
        )
        ds2 = ScientificDataset(
            dataset_id="ds_dup_2",
            workbook_name="Wb2.xlsx",
            sheet_name="JV",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            axis_name="Voltage",
            target_name="CurrentDensity",
            x_values=[0.0, 0.5],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[20.0, 10.0])],
        )

        wb1 = CanonicalWorkbook(workbook_name="Wb1.xlsx", datasets=[ds1])
        wb2 = CanonicalWorkbook(workbook_name="Wb2.xlsx", datasets=[ds2])

        dup_res = audit_duplicates([wb1, wb2])
        self.assertEqual(dup_res.total_datasets, 2)
        self.assertEqual(dup_res.same_experiment_diff_id_count, 1)
        self.assertEqual(len(dup_res.duplicate_groups), 1)

    def test_suspicious_parameter_detection(self):
        """Detect suspicious values for thickness < 1.0 nm and temperature < 100 K."""
        ds1 = ScientificDataset(
            dataset_id="ds_p1",
            workbook_name="Data file.xlsx",
            sheet_name="IDD",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            x_values=[0.0],
            traces=[
                CanonicalTrace(
                    trace_index=0,
                    trace_id="t1",
                    parameter=CanonicalParameter(name="thickness", value=0.0078, raw_text="0.0078", provenance=CellRefDTO("IDD", 0, 0, 0.0078)),
                    y_values=[10.0],
                )
            ],
        )
        wb = CanonicalWorkbook(workbook_name="Data file.xlsx", datasets=[ds1])
        dists, anomalies = audit_parameter_distributions([wb])

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["parameter"], "thickness")
        self.assertEqual(anomalies[0]["value"], 0.0078)

    def test_group_level_split_isolation_no_leakage(self):
        """Verify 100% group isolation across train, val, and test splits."""
        records = []
        for grp_idx in range(10):
            grp_id = f"grp_exp_{grp_idx}"
            for r_idx in range(5):
                records.append(
                    MLCurveRecord(
                        record_id=f"rec_{grp_idx}_{r_idx}",
                        dataset_id=f"ds_{grp_idx}",
                        group_id=grp_id,
                        material="Ca3PI3",
                        scientific_type="JV",
                        axis_name="Voltage",
                        axis_unit="V",
                        target_name="CurrentDensity",
                        target_unit="mA/cm2",
                        trace_id=f"tr_{r_idx}",
                        parameters={"temp": 300},
                        x=0.1 * r_idx,
                        y=25.0 - r_idx,
                    )
                )

        col = MLDatasetCollection(curve_records=records)
        split_res = perform_leakage_free_group_split(col, train_ratio=0.70, val_ratio=0.15, seed=42)

        self.assertEqual(split_res.total_groups, 10)
        self.assertTrue(split_res.is_leakage_free)
        self.assertEqual(split_res.train_val_overlap, 0)
        self.assertEqual(split_res.train_test_overlap, 0)
        self.assertEqual(split_res.val_test_overlap, 0)

        # Confirm no dataset/group appears in multiple splits
        train_set = set(split_res.train_groups)
        val_set = set(split_res.val_groups)
        test_set = set(split_res.test_groups)

        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)

    def test_native_x_grid_preservation(self):
        """Verify native x-grids are preserved per scientific type without forced interpolation."""
        ds_jv = ScientificDataset(
            dataset_id="ds_jv",
            workbook_name="Wb.xlsx",
            sheet_name="S1",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            x_values=[0.0, 0.1, 0.2, 0.3],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[25.0, 24.0, 22.0, 18.0])],
        )

        ds_qe = ScientificDataset(
            dataset_id="ds_qe",
            workbook_name="Wb.xlsx",
            sheet_name="S2",
            region_id="r2",
            parent_region_id=None,
            scientific_type="QE",
            structural_source="NumericCurveGrid",
            x_values=[400.0, 500.0, 600.0],
            traces=[CanonicalTrace(trace_index=0, trace_id="t2", y_values=[80.0, 85.0, 90.0])],
        )

        wb = CanonicalWorkbook(workbook_name="Wb.xlsx", datasets=[ds_jv, ds_qe])
        grids = audit_native_x_grids([wb])

        self.assertEqual(grids["JV"]["min_length"], 4)
        self.assertEqual(grids["QE"]["min_length"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
