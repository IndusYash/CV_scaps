"""
app.services.test_pv_extractor
------------------------------
Unit tests for deterministic PV metrics extraction from JV curves.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.canonical.model import (
    CanonicalParameter,
    CanonicalTrace,
    CellRefDTO,
    ScientificDataset,
)
from app.database import db_manager
from app.services import pv_extractor


class TestPVExtractor(unittest.TestCase):
    """Test suite covering deterministic Jsc/Voc/FF/PCE parameter calculations from JV traces."""

    def setUp(self):
        # Override database path to use a temporary file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db_path = Path(self.temp_dir.name) / "test_solar_extractor.db"
        self.original_db_path = db_manager.DB_PATH
        db_manager.DB_PATH = self.temp_db_path

        # Initialize the test DB schema
        db_manager.initialize_database()

        self.prov = CellRefDTO(sheet_name="JV", row=10, col=2, raw_value="Test")
        self.param = CanonicalParameter(
            name="thickness", value=150.0, unit="nm", raw_text="150nm", provenance=self.prov
        )

        # Standard JV curve values (voltage, current density)
        # Jsc = 24.0 mA/cm2 (at V=0)
        # Voc = 1.0 V (at J=0)
        # Pmax is approx at V=0.8 V, J=20.0 mA/cm2 -> Pmax = 16.0 mW/cm2
        # FF = 16.0 / (24.0 * 1.0) = 0.6667
        # PCE = 16.0 %
        self.voltages = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        self.currents = [24.0, 23.5, 22.0, 20.0, 18.0, 0.0, -10.0]

        self.trace = CanonicalTrace(
            trace_index=0,
            trace_id="test_jv_trace_1",
            parameter=self.param,
            y_values=self.currents,
            y_provenance=[self.prov] * len(self.currents),
        )
        self.dataset = ScientificDataset(
            dataset_id="test_jv_ds",
            workbook_name="TestWorkbook.xlsx",
            sheet_name="JV",
            region_id="r0",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            axis_name="Voltage",
            axis_unit="V",
            target_name="Current",
            target_unit="mA/cm2",
            global_parameters=[],
            x_values=self.voltages,
            x_provenance=[self.prov] * len(self.voltages),
            traces=[self.trace],
            scalar_records=[],
            validation_status="ACCEPTED_VALID",
            acceptance_state="ACCEPTED",
            is_ml_ready=True,
            confidence_score=1.0,
            ambiguity_gap=1.0,
            validation_issues=[],
            raw_dataset_reference=None,
        )

    def tearDown(self):
        # Restore original database path and clean up temp dir
        db_manager.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_pv_metrics_extraction_and_persistence(self):
        """Verify Jsc, Voc, FF, PCE are deterministically extracted and stored."""
        # Insert parent dataset first (due to foreign key constraint)
        db_manager.insert_dataset(self.dataset)

        count = pv_extractor.extract_and_persist_pv_metrics(self.dataset)
        self.assertEqual(count, 1)

        # Retrieve and verify derived metrics
        metrics = db_manager.get_derived_metrics("test_jv_ds", "test_jv_trace_1")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["j_sc"], 24.0)
        self.assertEqual(metrics["v_oc"], 1.0)
        self.assertGreater(metrics["p_max"], 0.0)
        self.assertEqual(metrics["ff"], round(metrics["p_max"] / (24.0 * 1.0), 4))
        self.assertEqual(metrics["pce"], metrics["p_max"])

    def test_pv_metrics_interpolation_for_jsc_and_voc(self):
        """Verify correct linear interpolation when exact V=0 or J=0 are not present."""
        # V=0.0 is not in x, but bracketed by -0.1 and 0.1
        # J=0.0 is not in y, but bracketed by 1.0 and -1.0
        x_vals = [-0.1, 0.1, 0.5, 0.9, 1.1]
        y_vals = [24.2, 23.8, 20.0, 2.0, -2.0]
        
        trace = CanonicalTrace(
            trace_index=0,
            trace_id="test_jv_trace_interp",
            parameter=self.param,
            y_values=y_vals,
            y_provenance=[self.prov] * len(y_vals),
        )
        ds = ScientificDataset(
            dataset_id="test_jv_ds_interp",
            workbook_name="TestWorkbook.xlsx",
            sheet_name="JV",
            region_id="r0",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            axis_name="Voltage",
            axis_unit="V",
            target_name="Current",
            target_unit="mA/cm2",
            global_parameters=[],
            x_values=x_vals,
            x_provenance=[self.prov] * len(x_vals),
            traces=[trace],
            scalar_records=[],
            validation_status="ACCEPTED_VALID",
            acceptance_state="ACCEPTED",
            is_ml_ready=True,
        )
        
        db_manager.insert_dataset(ds)
        count = pv_extractor.extract_and_persist_pv_metrics(ds)
        self.assertEqual(count, 1)

        metrics = db_manager.get_derived_metrics("test_jv_ds_interp", "test_jv_trace_interp")
        self.assertIsNotNone(metrics)
        # Jsc interpolated between -0.1 (24.2) and 0.1 (23.8) -> 24.0
        self.assertEqual(metrics["j_sc"], 24.0)
        # Voc interpolated between 0.9 (2.0) and 1.1 (-2.0) -> 1.0
        self.assertEqual(metrics["v_oc"], 1.0)


if __name__ == "__main__":
    unittest.main()
