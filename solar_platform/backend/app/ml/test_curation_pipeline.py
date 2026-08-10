"""
app.ml.test_curation_pipeline
------------------------------
Unit & Integration Test Suite for ML Curation Pipeline Layer (app.ml.curation_pipeline).
"""

import unittest
from app.canonical.model import CanonicalParameter, CanonicalTrace, CanonicalWorkbook, ScientificDataset
from app.ml.curation_pipeline import (
    MLCurationPipeline,
    export_training_manifest_json,
    export_training_manifest_md,
)


class TestMLCurationPipeline(unittest.TestCase):
    """Test suite covering dataset curation pipeline, split regimes, and manifest generation."""

    def setUp(self):
        # Create synthetic datasets representing Ca3PI3, Ca3PBr3, Ca3AsBr3, and Data file vs Data File (1)
        self.ds_ca3pi3 = ScientificDataset(
            dataset_id="ds_pi3_1",
            workbook_name="Ca3PI3.xlsx",
            sheet_name="JV",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            x_values=[0.0, 0.5, 1.0],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[25.0, 20.0, 0.0])],
        )

        self.ds_ca3pbr3 = ScientificDataset(
            dataset_id="ds_pbr3_1",
            workbook_name="Ca3PBr3.xlsx",
            sheet_name="JV",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            x_values=[0.0, 0.5, 1.0],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[24.0, 19.0, 0.0])],
        )

        self.ds_ca3asbr3 = ScientificDataset(
            dataset_id="ds_asbr3_1",
            workbook_name="Ca3AsBr3.xlsx",
            sheet_name="JV",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            x_values=[0.0, 0.5, 1.0],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[22.0, 17.0, 0.0])],
        )

        # Cross-workbook exact replica pair
        self.ds_datafile1 = ScientificDataset(
            dataset_id="ds_df1_r1",
            workbook_name="Data file.xlsx",
            sheet_name="Thick+DD",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            x_values=[0.0, 0.1],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[25.5, 25.0])],
        )

        self.ds_datafile2 = ScientificDataset(
            dataset_id="ds_df2_r1",
            workbook_name="Data File (1).xlsx",
            sheet_name="Thick+DD",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            x_values=[0.0, 0.1],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[25.5, 25.0])],
        )

        self.wb1 = CanonicalWorkbook(workbook_name="Ca3PI3.xlsx", datasets=[self.ds_ca3pi3])
        self.wb2 = CanonicalWorkbook(workbook_name="Ca3PBr3.xlsx", datasets=[self.ds_ca3pbr3])
        self.wb3 = CanonicalWorkbook(workbook_name="Ca3AsBr3.xlsx", datasets=[self.ds_ca3asbr3])
        self.wb4 = CanonicalWorkbook(workbook_name="Data file.xlsx", datasets=[self.ds_datafile1])
        self.wb5 = CanonicalWorkbook(workbook_name="Data File (1).xlsx", datasets=[self.ds_datafile2])

        self.all_wbs = [self.wb1, self.wb2, self.wb3, self.wb4, self.wb5]

    def test_full_audit_vs_training_dataset_views(self):
        """Verify FULL_AUDIT_DATASET contains 5 datasets while TRAINING_DATASET contains 4 (1 replica removed)."""
        pipeline = MLCurationPipeline(seed=42)
        res = pipeline.execute(self.all_wbs)

        self.assertEqual(len(res.full_audit_collection.metadata["datasets"]), 5)
        self.assertEqual(len(res.training_collection.metadata["datasets"]), 4)
        self.assertEqual(len(res.duplicate_families), 4)

    def test_regime_a_in_domain_group_isolation(self):
        """Verify Regime A (70/15/15) split achieves 100% zero leakage across experiment_family_ids."""
        pipeline = MLCurationPipeline(seed=42)
        res = pipeline.execute(self.all_wbs)

        self.assertTrue(res.is_regime_a_leakage_free)
        train_set = set(res.regime_a_train_family_ids)
        val_set = set(res.regime_a_val_family_ids)
        test_set = set(res.regime_a_test_family_ids)

        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)

    def test_regime_b_cross_material_isolation(self):
        """Verify Regime B isolates Ca3AsBr3 exclusively to test set with 0 material cross-talk."""
        pipeline = MLCurationPipeline(seed=42)
        res = pipeline.execute(self.all_wbs)

        self.assertTrue(res.is_regime_b_material_isolated)
        train_set = set(res.regime_b_train_family_ids)
        test_set = set(res.regime_b_test_family_ids)

        self.assertEqual(len(train_set.intersection(test_set)), 0)
        for fam_id in test_set:
            self.assertIn("Ca3AsBr3", fam_id)

    def test_manifest_export_json_and_md(self):
        """Verify deterministic training manifest export in JSON and Markdown formats."""
        pipeline = MLCurationPipeline(seed=42)
        res = pipeline.execute(self.all_wbs)

        json_manifest = export_training_manifest_json(res)
        md_manifest = export_training_manifest_md(res)

        self.assertIn("summary", json_manifest)
        self.assertIn("manifest_entries", json_manifest)
        self.assertIn("# ML Training Manifest", md_manifest)
        self.assertIn("100% LEAKAGE FREE", md_manifest)

    def test_zero_mutation_of_canonical_dataset(self):
        """Verify canonical dataset objects are never mutated by curation pipeline."""
        orig_val = self.ds_datafile1.x_values[0]
        pipeline = MLCurationPipeline(seed=42)
        _ = pipeline.execute(self.all_wbs)

        self.assertEqual(self.ds_datafile1.x_values[0], orig_val)
        self.assertEqual(self.ds_datafile1.dataset_id, "ds_df1_r1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
