"""
app.services.test_rf_pv_regressor
---------------------------------
Unit & Integration Tests for Tabular Curation and RandomForest PV regressor.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.canonical.model import (
    CanonicalParameter,
    CanonicalTrace,
    CellRefDTO,
    ScientificDataset,
)
from app.database import db_manager
from app.services import pv_extractor
from app.services.rf_pv_regressor import MLRegressorManager, build_ml_dataset_from_db


class TestRFPVRegressor(unittest.TestCase):
    """Test suite covering Random Forest training, evaluation metrics, and leakage checks."""

    def setUp(self):
        # Override database path to use a temporary file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db_path = Path(self.temp_dir.name) / "test_solar_rf.db"
        self.original_db_path = db_manager.DB_PATH
        db_manager.DB_PATH = self.temp_db_path

        # Initialize the test DB schema
        db_manager.initialize_database()

        self.prov = CellRefDTO(sheet_name="JV", row=10, col=2, raw_value="Test")
        self.voltages = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        self.currents_1 = [24.0, 23.5, 22.0, 20.0, 18.0, 0.0, -10.0]
        self.currents_2 = [12.0, 11.5, 11.0, 10.0, 9.0, 0.0, -5.0]

        # Insert 6 different datasets to ensure we have enough group splits for LeaveOneGroupOut/patience checks
        # Alternate between workbooks: 4 datasets for "Ca3PI3.xlsx", 2 datasets for "Ca3AsBr3.xlsx"
        for idx in range(6):
            wb_name = "Ca3PI3.xlsx" if idx < 4 else "Ca3AsBr3.xlsx"
            param = CanonicalParameter(
                name="thickness", value=100.0 + idx * 50.0, unit="nm", raw_text=f"{100+idx*50}nm", provenance=self.prov
            )
            # Alternate curves
            curr_vals = self.currents_1 if idx % 2 == 0 else self.currents_2
            trace = CanonicalTrace(
                trace_index=0,
                trace_id=f"trace_{idx}_1",
                parameter=param,
                y_values=curr_vals,
                y_provenance=[self.prov] * len(curr_vals),
            )
            
            ds = ScientificDataset(
                dataset_id=f"ds_jv_{idx}",
                workbook_name=wb_name,
                sheet_name="JV",
                region_id=f"r{idx}",
                parent_region_id=f"parent_r{idx}",
                scientific_type="JV",
                structural_source="NumericCurveGrid",
                axis_name="Voltage",
                axis_unit="V",
                target_name="Current",
                target_unit="mA/cm2",
                global_parameters=[CanonicalParameter("temperature", 300.0, "K")],
                x_values=self.voltages,
                x_provenance=[self.prov] * len(self.voltages),
                traces=[trace],
                scalar_records=[],
                validation_status="ACCEPTED_VALID",
                acceptance_state="ACCEPTED",
                is_ml_ready=True,
            )
            db_manager.insert_dataset(ds)
            # Run metric extraction to store Jsc, Voc, FF, PCE in DB
            pv_extractor.extract_and_persist_pv_metrics(ds)

    def tearDown(self):
        # Restore original database path and clean up temp dir
        db_manager.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_build_ml_dataset_and_curation(self):
        """Verify clean tabular ML dataset curation with curated parameters mapping."""
        df, audit_log = build_ml_dataset_from_db()
        self.assertEqual(len(df), 6)
        
        # Check standard features present
        for col in ["thickness", "temperature", "material", "group_id", "j_sc", "v_oc", "ff", "pce"]:
            self.assertIn(col, df.columns)

        # Check values
        self.assertEqual(df["temperature"].iloc[0], 300.0)

    def test_random_forest_training_and_metrics(self):
        """Verify classical models train cleanly with baseline comparison metrics."""
        df, _ = build_ml_dataset_from_db()
        manager = MLRegressorManager(df)

        metrics = manager.train_and_evaluate("pce", test_size=0.33, random_state=42)
        self.assertIn("random_forest", metrics)
        self.assertIn("baseline_mean", metrics)
        self.assertIn("baseline_linear", metrics)

        rf_met = metrics["random_forest"]
        self.assertIn("r2", rf_met)
        self.assertIn("mae", rf_met)
        self.assertIn("rmse", rf_met)

        # Check train/test counts
        self.assertEqual(metrics["train_count"] + metrics["test_count"], 6)

        # Ensure prediction payloads are populated
        preds = manager.predictions["pce"]
        self.assertEqual(len(preds["actuals"]), metrics["test_count"])
        self.assertEqual(len(preds["predicted_rf"]), metrics["test_count"])

    def test_split_leakage_prevention(self):
        """Verify strictly experiment-level split: no trace from a group leaks between train and test."""
        df, _ = build_ml_dataset_from_db()
        manager = MLRegressorManager(df)
        
        # Train and check trace splits
        manager.train_and_evaluate("j_sc", test_size=0.33, random_state=42)
        preds = manager.predictions["j_sc"]
        
        # Reload models and verify group_id split
        train_groups = set(df[~df["dataset_id"].isin(preds["dataset_ids"])]["group_id"])
        test_groups = set(preds["group_ids"])

        # No intersection between train and test group_ids
        self.assertEqual(train_groups.intersection(test_groups), set())

    def test_unseen_workbook_split(self):
        """Verify unseen workbook validation mode performs correct workbook-level isolation."""
        df, _ = build_ml_dataset_from_db()
        manager = MLRegressorManager(df)

        # Let's validate holding out Ca3AsBr3.xlsx
        metrics = manager.train_and_evaluate(
            "pce",
            validation_mode="unseen_workbook",
            test_workbook="Ca3AsBr3.xlsx"
        )
        
        self.assertEqual(metrics["validation_mode"], "unseen_workbook")
        self.assertIn("train_workbooks", metrics)
        self.assertIn("test_workbooks", metrics)
        
        # Verify strict isolation
        self.assertEqual(metrics["test_workbooks"], ["Ca3AsBr3.xlsx"])
        self.assertNotIn("Ca3AsBr3.xlsx", metrics["train_workbooks"])

        # Check statistical warning since test count = 2 (which is < 5)
        self.assertFalse(metrics["is_statistically_meaningful"])
        self.assertEqual(metrics["test_count"], 2)
        self.assertEqual(metrics["train_count"], 4)

    def test_train_only_material_encoding(self):
        """Verify that material categorical features fit exclusively on train set to prevent leakages."""
        df, _ = build_ml_dataset_from_db()
        
        # Ensure we have different materials in train vs test
        # Let's change the materials in df manually to test the encoder fitting logic
        df.loc[df["workbook_name"] == "Ca3PI3.xlsx", "material"] = "Ca3pi3"
        df.loc[df["workbook_name"] == "Ca3AsBr3.xlsx", "material"] = "Ca3asbr3" # Unseen in train!

        manager = MLRegressorManager(df)
        metrics = manager.train_and_evaluate(
            "pce",
            validation_mode="unseen_workbook",
            test_workbook="Ca3AsBr3.xlsx"
        )

        # Features should include 'mat_ca3pi3' but NOT 'mat_ca3asbr3' because the latter is only in test
        features = metrics["features"]
        self.assertIn("mat_ca3pi3", features)
        self.assertNotIn("mat_ca3asbr3", features)


if __name__ == "__main__":
    unittest.main()
