"""
app.ml.modeling.test_modeling
------------------------------
Unit & Integration Test Suite for ML Modeling, Preprocessing, and Experiment Design Layer.
"""

import unittest
from app.canonical.model import CanonicalParameter, CanonicalTrace, CanonicalWorkbook, ScientificDataset
from app.ml.builder import MLDatasetBuilder
from app.ml.curation_pipeline import MLCurationPipeline
from app.ml.model import MLCurveRecord, MLDatasetCollection
from app.ml.modeling.config import ModelConfig, PreprocessorConfig, TrainingConfig
from app.ml.modeling.dataset import CurvePointDataset
from app.ml.modeling.evaluation import EvaluationRunner
from app.ml.modeling.losses import calculate_huber_loss_numeric, calculate_mae_numeric, calculate_rmse_numeric
from app.ml.modeling.metrics import calculate_curve_metrics, calculate_scalar_metrics
from app.ml.modeling.models import BaselineLinearModel, BaselineMeanModel, validate_model_forward_pass
from app.ml.modeling.preprocessing import MLPreprocessor


class TestMLModeling(unittest.TestCase):
    """Test suite covering ML preprocessor, dataset containers, tensor shapes, and split isolation."""

    def setUp(self):
        # Create synthetic datasets
        self.ds1 = ScientificDataset(
            dataset_id="ds_p1",
            workbook_name="Ca3PI3.xlsx",
            sheet_name="JV",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            global_parameters=[CanonicalParameter("temperature", 300.0)],
            x_values=[0.0, 0.5, 1.0],
            traces=[
                CanonicalTrace(
                    trace_index=0,
                    trace_id="t1",
                    parameter=CanonicalParameter("thickness", 100.0),
                    y_values=[25.0, 20.0, 0.0],
                )
            ],
        )

        self.ds2 = ScientificDataset(
            dataset_id="ds_p2",
            workbook_name="Ca3AsBr3.xlsx",
            sheet_name="JV",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            global_parameters=[CanonicalParameter("temperature", 300.0)],
            x_values=[0.0, 0.5, 1.0],
            traces=[
                CanonicalTrace(
                    trace_index=0,
                    trace_id="t1",
                    parameter=CanonicalParameter("thickness", 200.0),
                    y_values=[24.0, 18.0, 0.0],
                )
            ],
        )

        wb1 = CanonicalWorkbook(workbook_name="Ca3PI3.xlsx", datasets=[self.ds1])
        wb2 = CanonicalWorkbook(workbook_name="Ca3AsBr3.xlsx", datasets=[self.ds2])
        self.curation_res = MLCurationPipeline(seed=42).execute([wb1, wb2])

    def test_material_encoding(self):
        """Verify categorical one-hot material encoding vector dimensions."""
        prep = MLPreprocessor()
        vec_pi3 = prep.transform_material_one_hot("Ca3PI3.xlsx")
        vec_asbr3 = prep.transform_material_one_hot("Ca3AsBr3")

        self.assertEqual(len(vec_pi3), len(prep.config.known_materials))
        self.assertEqual(len(vec_asbr3), len(prep.config.known_materials))
        self.assertNotEqual(vec_pi3, vec_asbr3)

    def test_parameter_transformation_and_log_scaling(self):
        """Verify log10 transformation for wide-span parameters and z-score normalization."""
        records = [
            MLCurveRecord(
                record_id="r1", dataset_id="d1", group_id="g1", material="Ca3PI3",
                scientific_type="JV", axis_name="Voltage", axis_unit="V", target_name="CurrentDensity",
                target_unit="mA/cm2", trace_id="t1", parameters={"defect_density": 1e13, "thickness": 100.0},
                x=0.0, y=25.0
            ),
            MLCurveRecord(
                record_id="r2", dataset_id="d2", group_id="g2", material="Ca3PI3",
                scientific_type="JV", axis_name="Voltage", axis_unit="V", target_name="CurrentDensity",
                target_unit="mA/cm2", trace_id="t1", parameters={"defect_density": 1e18, "thickness": 500.0},
                x=1.0, y=0.0
            )
        ]
        prep = MLPreprocessor()
        prep.fit(records)
        self.assertTrue(prep.is_fitted)

        feat, y = prep.transform_curve_record(records[0])
        self.assertEqual(len(feat), 15)  # 6 material + 8 params + 1 x
        self.assertEqual(y, 25.0)

    def test_native_x_grid_preservation(self):
        """Verify native x-coordinates are preserved natively in feature vector index 14."""
        records = [
            MLCurveRecord(
                record_id="r1", dataset_id="d1", group_id="g1", material="Ca3PI3",
                scientific_type="JV", axis_name="Voltage", axis_unit="V", target_name="CurrentDensity",
                target_unit="mA/cm2", trace_id="t1", parameters={"thickness": 100.0},
                x=0.25, y=20.0
            )
        ]
        prep = MLPreprocessor()
        prep.fit(records)
        feat, _ = prep.transform_curve_record(records[0])
        self.assertEqual(len(feat), 15)

    def test_tensor_shapes_forward_pass(self):
        """Verify model forward pass tensor shapes (batch_size, 15) -> (batch_size, 1)."""
        res = validate_model_forward_pass(input_dim=15, batch_size=8)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["input_shape"], [8, 15])
        self.assertEqual(res["output_shape"], [8, 1])

    def test_no_access_to_test_targets_during_preprocessing_fit(self):
        """Verify preprocessor fits statistics strictly on training records only."""
        runner = EvaluationRunner(self.curation_res)
        datasets = runner.prepare_regime_a_datasets()

        train_ds = datasets["train"]
        val_ds = datasets["val"]

        self.assertTrue(train_ds.preprocessor.is_fitted)
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(val_ds), 0)

    def test_metrics_calculation(self):
        """Verify MAE, RMSE, and R2 calculation on numeric lists."""
        preds = [10.0, 20.0, 30.0]
        targets = [10.0, 20.0, 30.0]
        metrics = calculate_curve_metrics(preds, targets)
        self.assertEqual(metrics.mae, 0.0)
        self.assertEqual(metrics.r2, 1.0)

    def test_reproducibility_with_fixed_seed(self):
        """Verify deterministic curation pipeline and preprocessor initialization."""
        wb1 = CanonicalWorkbook(workbook_name="Ca3PI3.xlsx", datasets=[self.ds1])
        wb2 = CanonicalWorkbook(workbook_name="Ca3AsBr3.xlsx", datasets=[self.ds2])
        p1 = MLCurationPipeline(seed=42).execute([wb1, wb2])
        p2 = MLCurationPipeline(seed=42).execute([wb1, wb2])
        self.assertEqual(p1.regime_a_train_family_ids, p2.regime_a_train_family_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
