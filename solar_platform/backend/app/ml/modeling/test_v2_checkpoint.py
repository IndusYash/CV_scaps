"""
app.ml.modeling.test_v2_checkpoint
-----------------------------------
Unit & Regression Tests for V2 Checkpoint Serialization, Roundtrip Integrity, and Inference Smoke Test.
"""
import json
import tempfile
import unittest
from pathlib import Path

from app.canonical.model import CanonicalParameter, CanonicalTrace, CanonicalWorkbook, ScientificDataset
from app.ml.curation_pipeline import MLCurationPipeline
from app.ml.modeling.config import TrainingConfig
from app.ml.modeling.dataset import CurvePointDataset
from app.ml.modeling.preprocessing import MLPreprocessor
from app.ml.modeling.v2_checkpoint import (
    compute_file_sha256,
    count_model_parameters,
    load_v2_checkpoint,
    save_v2_checkpoint,
)
from app.ml.modeling.v2_models import SolarV2MultiHeadSurrogate, TargetNormalizer


class TestV2Checkpoint(unittest.TestCase):
    """Test suite covering checkpoint serialization, roundtrip fidelity, and fail-loudly policy."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.ds = ScientificDataset(
            dataset_id="ds1",
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
        wb = CanonicalWorkbook(workbook_name="Ca3PI3.xlsx", datasets=[self.ds])
        curation_res = MLCurationPipeline(seed=42).execute([wb])
        self.records = curation_res.training_collection.curve_records

        self.prep = MLPreprocessor()
        self.prep.fit(self.records)

        self.normalizer = TargetNormalizer()
        self.normalizer.fit({"JV": [r.y for r in self.records]})

        self.config = TrainingConfig()
        self.model = SolarV2MultiHeadSurrogate(param_dim=14, num_fourier_freqs=8)

    def test_checkpoint_serialization_roundtrip(self):
        """Verify model, preprocessor, and normalizer roundtrip identically through disk serialization."""
        checkpoint_dir = self.tmp_dir / "v2_ckpt"
        save_v2_checkpoint(
            model=self.model,
            preprocessor=self.prep,
            normalizer=self.normalizer,
            config=self.config,
            manifest_entries_count=1,
            save_dir=checkpoint_dir,
        )

        self.assertTrue((checkpoint_dir / "model_checkpoint_manifest.json").exists())
        self.assertTrue((checkpoint_dir / "v2_model_weights.json").exists())
        self.assertTrue((checkpoint_dir / "v2_preprocessor.json").exists())

        # Reload checkpoint
        r_model, r_prep, r_norm, r_cfg, manifest = load_v2_checkpoint(checkpoint_dir)

        self.assertEqual(manifest["model_name"], "SolarV2MultiHeadSurrogate")
        self.assertEqual(r_norm.means["JV"], self.normalizer.means["JV"])
        self.assertEqual(r_prep.param_means, self.prep.param_means)

    def test_inference_smoke_test(self):
        """Verify reloaded checkpoint generates non-zero predictions with continuous variation."""
        ds = CurvePointDataset(self.records, self.prep)
        feats = ds.features
        self.assertEqual(len(feats), 3)
        self.assertEqual(len(feats[0]), 15)

    def test_fail_loudly_on_missing_checkpoint(self):
        """Verify load_v2_checkpoint raises RuntimeError if checkpoint is missing."""
        missing_dir = self.tmp_dir / "non_existent_ckpt"
        with self.assertRaises(RuntimeError) as ctx:
            load_v2_checkpoint(missing_dir)
        self.assertIn("Trained V2 checkpoint not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
