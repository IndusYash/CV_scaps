"""
app.tests.test_v1_e2e
---------------------
End-to-End integration test covering SCAPS workbook upload, SQLite database persistence,
PV metric extraction, and global RandomForest training on PV targets.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from io import BytesIO
from fastapi import UploadFile

backend_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(backend_dir))

from app.api.endpoints import (
    upload_file,
    get_db_status,
    list_datasets_endpoint,
    get_dataset_endpoint,
    train_global_random_forest,
    get_global_ml_results,
)
from app.database import db_manager


class TestV1EndToEnd(unittest.TestCase):
    """End-to-End test suite covering V1 database persistence and RandomForest regression flow."""

    @classmethod
    def setUpClass(cls):
        # Setup temporary test database
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_db_path = Path(cls.temp_dir.name) / "test_v1_e2e.db"
        cls.original_db_path = db_manager.DB_PATH
        db_manager.DB_PATH = cls.temp_db_path

        # Initialize the test DB schema
        db_manager.initialize_database()

        # Locate workbook Ca3PI3.xlsx
        wb_path = backend_dir.parent / "Ca3PI3.xlsx"
        if not wb_path.exists():
            wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"
            
        with open(wb_path, "rb") as f:
            contents = f.read()

        # Upload and trigger parsing/persistence
        upload_file_obj = UploadFile(filename=wb_path.name, file=BytesIO(contents))
        cls.upload_response = asyncio.run(upload_file(upload_file_obj))

    @classmethod
    def tearDownClass(cls):
        # Restore original database path and clean up temp dir
        db_manager.DB_PATH = cls.original_db_path
        cls.temp_dir.cleanup()

    def test_e2e_flow_and_persistence(self):
        """Verify upload -> database persist -> PV metric extraction -> RF model training pipeline."""
        # 1. Check status shows datasets have been persisted
        status = get_db_status()
        self.assertGreater(status["datasets_count"], 0)
        self.assertGreater(status["derived_metrics_count"], 0)

        # 2. Get list of datasets and ensure scientific metadata is correct
        datasets = list_datasets_endpoint()
        self.assertEqual(len(datasets), status["datasets_count"])
        
        # Check single JV dataset content
        jv_dataset = next((d for d in datasets if d["scientific_type"] == "JV"), None)
        self.assertIsNotNone(jv_dataset)

        # Retrieve specific dataset by ID
        ds_id = jv_dataset["dataset_id"]
        detail = get_dataset_endpoint(ds_id)
        self.assertEqual(detail["dataset_id"], ds_id)
        
        # Verify derived metrics were persisted Cascade-style
        self.assertTrue(len(detail["derived_metrics"]) >= 1)
        first_metric = detail["derived_metrics"][0]
        self.assertIsNotNone(first_metric["j_sc"])
        self.assertIsNotNone(first_metric["v_oc"])
        self.assertIsNotNone(first_metric["ff"])
        self.assertIsNotNone(first_metric["pce"])

        # 3. Train Global RandomForest models on PV metrics
        train_res = asyncio.run(train_global_random_forest(test_size=0.33, random_state=42))
        self.assertEqual(train_res["status"], "SUCCESS")
        
        # 4. Verify predictions results are served correctly
        results = get_global_ml_results()
        self.assertIn("j_sc", results["metrics"])
        self.assertIn("v_oc", results["metrics"])
        self.assertIn("ff", results["metrics"])
        self.assertIn("pce", results["metrics"])

        # Verify predictions actual vs pred parity payloads
        self.assertTrue(len(results["predictions"]["j_sc"]["actuals"]) > 0)


if __name__ == "__main__":
    unittest.main()
