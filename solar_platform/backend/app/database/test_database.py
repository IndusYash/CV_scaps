"""
app.database.test_database
--------------------------
Unit tests covering SQLite persistence layer operations.
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


class TestSQLiteDatabase(unittest.TestCase):
    """Test suite covering database CRUD operations and derived metrics storage."""

    def setUp(self):
        # Override database path to use a temporary file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db_path = Path(self.temp_dir.name) / "test_solar.db"
        self.original_db_path = db_manager.DB_PATH
        db_manager.DB_PATH = self.temp_db_path

        # Initialize the test DB schema
        db_manager.initialize_database()

        # Build a dummy dataset for testing
        self.prov = CellRefDTO(sheet_name="JV", row=10, col=2, raw_value="Test")
        self.param = CanonicalParameter(
            name="thickness", value=150.0, unit="nm", raw_text="150nm", provenance=self.prov
        )
        self.trace = CanonicalTrace(
            trace_index=0,
            trace_id="test_ds_trace_1",
            parameter=self.param,
            y_values=[20.0, 15.0, 0.0],
            y_provenance=[self.prov, self.prov, self.prov],
        )
        self.dataset = ScientificDataset(
            dataset_id="test_ds_1",
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
            x_values=[0.0, 0.5, 1.0],
            x_provenance=[self.prov, self.prov, self.prov],
            traces=[self.trace],
            scalar_records=[],
            validation_status="ACCEPTED_VALID",
            acceptance_state="ACCEPTED",
            is_ml_ready=True,
            confidence_score=1.0,
            ambiguity_gap=1.0,
            validation_issues=[],
            raw_dataset_reference={"score": 100},
        )

    def tearDown(self):
        # Restore original database path and clean up temp dir
        db_manager.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_insert_and_retrieve_dataset(self):
        """Verify insertion and deterministic retrieval of Canonical ScientificDataset."""
        db_manager.insert_dataset(self.dataset)
        self.assertEqual(db_manager.count_datasets(), 1)

        retrieved = db_manager.get_dataset("test_ds_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.dataset_id, "test_ds_1")
        self.assertEqual(retrieved.workbook_name, "TestWorkbook.xlsx")
        self.assertEqual(len(retrieved.traces), 1)
        self.assertEqual(retrieved.traces[0].trace_id, "test_ds_trace_1")
        self.assertEqual(retrieved.traces[0].parameter.value, 150.0)
        self.assertEqual(retrieved.is_ml_ready, True)

    def test_list_and_filter_datasets(self):
        """Verify listing and filtering datasets by scientific type."""
        db_manager.insert_dataset(self.dataset)

        # Add another dataset with different scientific type
        other_dataset = ScientificDataset(
            dataset_id="test_ds_2",
            workbook_name="TestWorkbook.xlsx",
            sheet_name="QE",
            region_id="r1",
            parent_region_id=None,
            scientific_type="QE",
            structural_source="NumericCurveGrid",
            validation_status="ACCEPTED_VALID",
            is_ml_ready=True,
        )
        db_manager.insert_dataset(other_dataset)
        self.assertEqual(db_manager.count_datasets(), 2)

        # List all
        all_ds = db_manager.list_datasets()
        self.assertEqual(len(all_ds), 2)

        # Filter by JV
        jv_ds = db_manager.list_datasets("JV")
        self.assertEqual(len(jv_ds), 1)
        self.assertEqual(jv_ds[0].dataset_id, "test_ds_1")

        # Filter by QE
        qe_ds = db_manager.list_datasets("QE")
        self.assertEqual(len(qe_ds), 1)
        self.assertEqual(qe_ds[0].dataset_id, "test_ds_2")

    def test_clear_database(self):
        """Verify database rows are successfully cleared on reset."""
        db_manager.insert_dataset(self.dataset)
        self.assertEqual(db_manager.count_datasets(), 1)

        db_manager.clear_database()
        self.assertEqual(db_manager.count_datasets(), 0)

    def test_derived_metrics_persistence(self):
        """Verify insertion, retrieval, and cascading deletion of derived PV metrics."""
        db_manager.insert_dataset(self.dataset)

        db_manager.insert_derived_metrics(
            dataset_id="test_ds_1",
            trace_id="test_ds_trace_1",
            j_sc=20.0,
            v_oc=1.0,
            p_max=15.0,
            ff=0.75,
            pce=15.0,
            extracted_at="2026-08-15T00:00:00Z",
            provenance_notes="Unit test curve",
        )

        metrics = db_manager.get_derived_metrics("test_ds_1", "test_ds_trace_1")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["j_sc"], 20.0)
        self.assertEqual(metrics["v_oc"], 1.0)
        self.assertEqual(metrics["ff"], 0.75)
        self.assertEqual(metrics["pce"], 15.0)

        # Test cascading deletion of metrics when parent dataset is deleted
        db_manager.clear_database()
        metrics_after = db_manager.get_derived_metrics("test_ds_1", "test_ds_trace_1")
        self.assertNull = self.assertIsNone(metrics_after)


if __name__ == "__main__":
    unittest.main()
