"""
app.ml.test_ml
--------------
Comprehensive Unit Test Suite for ML Dataset Preparation Layer (app.ml).
"""

import sys
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
from app.ml.builder import MLDatasetBuilder
from app.ml.model import MLCurveRecord, MLDatasetCollection, MLScalarRecord
from app.ml.quality_report import (
    generate_ml_quality_report,
    get_experiment_grouping_diagnostics,
)


class TestMLDatasetPreparation(unittest.TestCase):
    """Test suite covering ML dataset builder, models, quality reports, and filtering."""

    def test_no_excel_parser_v2_dependencies(self):
        """Verify app.ml modules do not import openpyxl or parser_v2 internal modules."""
        ml_modules = ["app.ml.model", "app.ml.builder", "app.ml.quality_report"]
        for mod_name in ml_modules:
            mod = sys.modules.get(mod_name)
            if mod:
                for attr in dir(mod):
                    val = getattr(mod, attr)
                    if hasattr(val, "__module__") and val.__module__:
                        self.assertNotIn("openpyxl", val.__module__)
                        self.assertNotIn("parser_v2", val.__module__)

    def test_jv_canonical_dataset_to_ml_records(self):
        """JV canonical curve dataset -> ML curve records."""
        param = CanonicalParameter(name="thickness", value=100.0, unit="nm")
        trace = CanonicalTrace(
            trace_index=0,
            trace_id="jv_trace_1",
            parameter=param,
            y_values=[25.5, 25.0],
            y_provenance=[CellRefDTO(sheet_name="JV", row=1, col=1, raw_value=25.5), CellRefDTO(sheet_name="JV", row=2, col=1, raw_value=25.0)],
        )
        ds = ScientificDataset(
            dataset_id="ds_jv_1",
            workbook_name="Ca3PI3.xlsx",
            sheet_name="JV",
            region_id="reg_0",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            axis_name="Voltage",
            axis_unit="V",
            target_name="CurrentDensity",
            target_unit="mA/cm2",
            x_values=[0.0, 0.1],
            x_provenance=[CellRefDTO(sheet_name="JV", row=1, col=0, raw_value=0.0), CellRefDTO(sheet_name="JV", row=2, col=0, raw_value=0.1)],
            traces=[trace],
            validation_status="ACCEPTED_VALID",
            is_ml_ready=True,
        )

        builder = MLDatasetBuilder()
        col = builder.build_from_canonical_dataset(ds)

        self.assertEqual(len(col.curve_records), 2)
        r0 = col.curve_records[0]
        self.assertEqual(r0.dataset_id, "ds_jv_1")
        self.assertEqual(r0.material, "Ca3PI3")
        self.assertEqual(r0.scientific_type, "JV")
        self.assertEqual(r0.x, 0.0)
        self.assertEqual(r0.y, 25.5)
        self.assertEqual(r0.parameters["thickness"], 100.0)
        self.assertEqual(r0.x_provenance.col, 0)
        self.assertEqual(r0.y_provenance.col, 1)

    def test_qe_canonical_dataset_to_ml_records(self):
        """QE canonical dataset -> ML curve records."""
        trace = CanonicalTrace(
            trace_index=0,
            trace_id="qe_trace_1",
            y_values=[85.0, 90.0],
        )
        ds = ScientificDataset(
            dataset_id="ds_qe_1",
            workbook_name="Ca3PBr3.xlsx",
            sheet_name="QE",
            region_id="reg_0",
            parent_region_id=None,
            scientific_type="QE",
            structural_source="NumericCurveGrid",
            axis_name="Wavelength",
            axis_unit="nm",
            x_values=[400.0, 500.0],
            traces=[trace],
        )

        builder = MLDatasetBuilder()
        col = builder.build_from_canonical_dataset(ds)

        self.assertEqual(len(col.curve_records), 2)
        self.assertEqual(col.curve_records[0].scientific_type, "QE")
        self.assertEqual(col.curve_records[0].material, "Ca3PBr3")

    def test_responsivity_and_detectivity_records(self):
        """Responsivity and Detectivity canonical datasets -> ML curve records."""
        ds_resp = ScientificDataset(
            dataset_id="ds_resp",
            workbook_name="Ca3AsBr3.xlsx",
            sheet_name="Optimized",
            region_id="reg_1",
            parent_region_id=None,
            scientific_type="RESPONSIVITY",
            structural_source="NumericCurveGrid",
            axis_name="Wavelength",
            axis_unit="nm",
            x_values=[400.0],
            traces=[CanonicalTrace(trace_index=0, trace_id="tr_1", y_values=[0.45])],
        )
        ds_det = ScientificDataset(
            dataset_id="ds_det",
            workbook_name="Ca3AsBr3.xlsx",
            sheet_name="Optimized",
            region_id="reg_2",
            parent_region_id=None,
            scientific_type="DETECTIVITY",
            structural_source="NumericCurveGrid",
            axis_name="Wavelength",
            axis_unit="nm",
            x_values=[400.0],
            traces=[CanonicalTrace(trace_index=0, trace_id="tr_1", y_values=[1.2e12])],
        )

        builder = MLDatasetBuilder()
        col_resp = builder.build_from_canonical_dataset(ds_resp)
        col_det = builder.build_from_canonical_dataset(ds_det)

        self.assertEqual(col_resp.curve_records[0].scientific_type, "RESPONSIVITY")
        self.assertEqual(col_det.curve_records[0].scientific_type, "DETECTIVITY")

    def test_scalar_sweep_to_ml_records(self):
        """Scalar sweep dataset -> ML scalar records."""
        rec1 = CanonicalScalarRecord(
            record_id="rec_1",
            parameter=CanonicalParameter(name="thickness", value=100.0, unit="nm"),
            target_name="Jsc",
            target_value=25.5,
            target_unit="mA/cm2",
            target_provenance=CellRefDTO(sheet_name="Thick+DD", row=65, col=2, raw_value=25.5),
        )
        ds = ScientificDataset(
            dataset_id="ds_scalar",
            workbook_name="Data File (1).xlsx",
            sheet_name="Thick+DD",
            region_id="reg_1",
            parent_region_id=None,
            scientific_type="SCALAR_SWEEP",
            structural_source="ScalarSweepTable",
            scalar_records=[rec1],
        )

        builder = MLDatasetBuilder()
        col = builder.build_from_canonical_dataset(ds)

        self.assertEqual(len(col.scalar_records), 1)
        s0 = col.scalar_records[0]
        self.assertEqual(s0.target_name, "Jsc")
        self.assertEqual(s0.target_value, 25.5)
        self.assertEqual(s0.parameters["thickness"], 100.0)

    def test_multi_parameter_and_multi_trace_datasets(self):
        """Multi-parameter datasets (temperature + thickness + defect_density)."""
        glob_param = CanonicalParameter(name="temperature", value=300.0, unit="K")
        trace_param = CanonicalParameter(name="thickness", value=200.0, unit="nm")
        trace = CanonicalTrace(
            trace_index=0,
            trace_id="tr_multi",
            parameter=trace_param,
            y_values=[50.0],
        )
        ds = ScientificDataset(
            dataset_id="ds_multi",
            workbook_name="Ca3PI3.xlsx",
            sheet_name="QE",
            region_id="reg_0",
            parent_region_id=None,
            scientific_type="QE",
            structural_source="StackedCurveBlocks",
            global_parameters=[glob_param],
            x_values=[400.0],
            traces=[trace],
        )

        builder = MLDatasetBuilder()
        col = builder.build_from_canonical_dataset(ds)

        r0 = col.curve_records[0]
        self.assertEqual(r0.parameters["temperature"], 300.0)
        self.assertEqual(r0.parameters["thickness"], 200.0)

    def test_different_x_grid_lengths_preserved_natively(self):
        """Curves with 3 points vs 5 points retain native x/y lengths without forced grid resampling."""
        ds1 = ScientificDataset(
            dataset_id="ds_len3",
            workbook_name="Wb",
            sheet_name="S1",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            x_values=[0.0, 0.5, 1.0],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[20.0, 15.0, 0.0])],
        )
        ds2 = ScientificDataset(
            dataset_id="ds_len5",
            workbook_name="Wb",
            sheet_name="S2",
            region_id="r2",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            x_values=[0.0, 0.2, 0.4, 0.6, 0.8],
            traces=[CanonicalTrace(trace_index=0, trace_id="t2", y_values=[25.0, 24.0, 22.0, 18.0, 10.0])],
        )

        builder = MLDatasetBuilder()
        col1 = builder.build_from_canonical_dataset(ds1)
        col2 = builder.build_from_canonical_dataset(ds2)

        self.assertEqual(len(col1.curve_records), 3)
        self.assertEqual(len(col2.curve_records), 5)

    def test_accepted_with_warnings_inclusion_and_uncertain_exclusion(self):
        """ACCEPTED_WITH_WARNINGS is included by default; UNCERTAIN/UNRESOLVED are excluded."""
        ds_warn = ScientificDataset(
            dataset_id="ds_warn",
            workbook_name="Wb",
            sheet_name="S",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            validation_status="ACCEPTED_WITH_WARNINGS",
            x_values=[0.0],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[20.0])],
        )
        ds_unc = ScientificDataset(
            dataset_id="ds_unc",
            workbook_name="Wb",
            sheet_name="S",
            region_id="r2",
            parent_region_id=None,
            scientific_type="UNKNOWN",
            structural_source="CategoryTable",
            validation_status="UNCERTAIN",
        )
        ds_unres = ScientificDataset(
            dataset_id="ds_unres",
            workbook_name="Wb",
            sheet_name="S",
            region_id="r3",
            parent_region_id=None,
            scientific_type="UNKNOWN",
            structural_source="NumericCurveGrid",
            validation_status="UNRESOLVED",
        )

        builder_default = MLDatasetBuilder()
        self.assertTrue(builder_default.is_dataset_included(ds_warn))
        self.assertFalse(builder_default.is_dataset_included(ds_unc))
        self.assertFalse(builder_default.is_dataset_included(ds_unres))

        col_warn = builder_default.build_from_canonical_dataset(ds_warn)
        col_unc = builder_default.build_from_canonical_dataset(ds_unc)
        col_unres = builder_default.build_from_canonical_dataset(ds_unres)

        self.assertEqual(len(col_warn.curve_records), 1)
        self.assertEqual(len(col_unc.curve_records), 0)
        self.assertEqual(len(col_unres.curve_records), 0)

    def test_ml_quality_report_and_grouping_diagnostics(self):
        """Test ML quality report generation and leakage-proof grouping diagnostics."""
        ds1 = ScientificDataset(
            dataset_id="ds_1",
            workbook_name="Ca3PI3.xlsx",
            sheet_name="JV",
            region_id="r1",
            parent_region_id="parent_sweep_1",
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            validation_status="ACCEPTED_VALID",
            x_values=[0.0, 0.5],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", parameter=CanonicalParameter("temp", 300.0), y_values=[25.0, 20.0])],
        )
        wb = CanonicalWorkbook(workbook_name="Ca3PI3.xlsx", datasets=[ds1])

        report = generate_ml_quality_report([wb])
        self.assertEqual(report.total_canonical_datasets, 1)
        self.assertEqual(report.ml_ready_datasets_count, 1)
        self.assertEqual(report.scientific_type_distribution["JV"], 1)

        builder = MLDatasetBuilder()
        col = builder.build_from_canonical_workbook(wb)
        diag = get_experiment_grouping_diagnostics(col)
        self.assertEqual(diag["total_experiment_groups"], 1)
        self.assertEqual(diag["total_curve_points"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
