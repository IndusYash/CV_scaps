import sys
import unittest
import asyncio
from pathlib import Path
from io import BytesIO
from fastapi import UploadFile

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.api.endpoints import upload_file, get_curves, train_model, predict_point_endpoint, predict_curve_endpoint
from app.models.schemas import TrainRequest, PointPredictRequest, CurvePredictRequest


class TestAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wb_path = backend_dir.parent / "Ca3PI3.xlsx"
        if not wb_path.exists():
            wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"
            
        with open(wb_path, "rb") as f:
            contents = f.read()

        upload_file_obj = UploadFile(filename=wb_path.name, file=BytesIO(contents))
        cls.upload_response = asyncio.run(upload_file(upload_file_obj))
        cls.file_id = cls.upload_response.file_id

    def test_upload_endpoint_returns_experiments(self):
        """Verify /api/upload returns Phase 2 experiment summaries."""
        self.assertIsNotNone(self.file_id)
        sheets = self.upload_response.sheets
        self.assertTrue(len(sheets) >= 1)
        
        sheet_illum = next((s for s in sheets if s.sheet_name == "Illumination"), None)
        self.assertIsNotNone(sheet_illum)
        self.assertEqual(len(sheet_illum.experiments), 1)
        exp_summary = sheet_illum.experiments[0]
        self.assertEqual(exp_summary.experiment_id, "exp_Illumination_Current")
        self.assertEqual(exp_summary.trace_count, 4)

    def test_curves_endpoint_returns_trace_payloads(self):
        """Verify /api/curves returns trace-oriented payloads for Plotly."""
        req = TrainRequest(file_id=self.file_id, sheet_name="QE (T+DD)")
        res = asyncio.run(get_curves(req))
        
        self.assertTrue(len(res.experiments) >= 1)
        exp_qe = next(e for e in res.experiments if e.target["name"] == "QE")
        self.assertEqual(exp_qe.experiment_id, "exp_QE_T_DD_QE")
        self.assertEqual(len(exp_qe.traces), 88)
        self.assertEqual(len(exp_qe.parameters), 3)
        
        # Verify trace structure
        first_trace = exp_qe.traces[0]
        print(f"DEBUG first_trace.x len = {len(first_trace.x)}, label = '{first_trace.label}'")
        self.assertEqual(len(first_trace.x), 61)
        self.assertEqual(len(first_trace.y), 61)
        self.assertTrue(first_trace.trace_id.startswith("trace_"))

    def test_train_and_prediction_endpoints(self):
        """Verify /api/train, /api/predict/point, and /api/predict/curve operate at experiment level."""
        # Train model for QE (T+DD)
        train_req = TrainRequest(file_id=self.file_id, sheet_name="QE (T+DD)")
        train_res = asyncio.run(train_model(train_req))
        self.assertIn("exp_QE_T_DD_QE", train_res.metrics)
        
        # Predict point
        pt_req = PointPredictRequest(
            file_id=self.file_id,
            sheet_name="QE (T+DD)",
            experiment_id="exp_QE_T_DD_QE",
            x_val=500.0,
            param_vals={"Thickness": 0.5, "Defect_Density": 1e14}
        )
        pt_res = asyncio.run(predict_point_endpoint(pt_req))
        self.assertIsNotNone(pt_res.predicted_value)
        
        # Predict curve
        cv_req = CurvePredictRequest(
            file_id=self.file_id,
            sheet_name="QE (T+DD)",
            experiment_id="exp_QE_T_DD_QE",
            param_vals={"Thickness": 0.5, "Defect_Density": 1e14}
        )
        cv_res = asyncio.run(predict_curve_endpoint(cv_req))
        self.assertEqual(len(cv_res.actual_x), len(cv_res.pred_y))
        self.assertTrue(len(cv_res.actual_x) > 10)


if __name__ == "__main__":
    unittest.main()
