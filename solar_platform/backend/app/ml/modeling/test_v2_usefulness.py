"""
app.ml.modeling.test_v2_usefulness
-----------------------------------
Unit tests for PV Metric Extraction, Zero-Crossing Root Finder, and Physical Sanity Checker.
"""
import unittest

from app.ml.modeling.pv_metrics import PVParameters, extract_pv_parameters, calculate_pv_parameter_errors
from app.ml.modeling.usefulness_evaluator import PhysicalSanityReport, perform_physical_sanity_checks


class TestV2Usefulness(unittest.TestCase):
    """Test suite covering PV metric extraction and usefulness evaluation."""

    def test_pv_parameter_extraction(self):
        """Verify extraction of Jsc, Voc, FF, PCE from synthetic JV curve."""
        v_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # Synthetic diode J(V) curve
        j_vals = [25.0, 24.5, 23.5, 20.0, 10.0, 0.0]

        pv = extract_pv_parameters(v_vals, j_vals)
        self.assertIsNotNone(pv)
        self.assertAlmostEqual(pv.j_sc, 25.0, places=1)
        self.assertAlmostEqual(pv.v_oc, 1.0, places=1)
        self.assertGreater(pv.pce, 0.0)

    def test_pv_parameter_errors_calculation(self):
        """Verify calculation of MAE and MAPE error metrics across PV parameter lists."""
        act = [PVParameters(j_sc=25.0, v_oc=1.0, p_max=18.0, ff=0.72, pce=18.0)]
        pred = [PVParameters(j_sc=24.0, v_oc=1.0, p_max=17.28, ff=0.72, pce=17.28)]

        errs = calculate_pv_parameter_errors(act, pred)
        self.assertEqual(errs.j_sc_mae, 1.0)
        self.assertEqual(errs.j_sc_mape, 4.0)

    def test_physical_sanity_checker_detects_out_of_range(self):
        """Verify PhysicalSanityReport flags unphysical predictions."""
        # Fake dataset dummy class
        class DummyRecord:
            x = 0.0
            y = 25.0
            dataset_id = "ds1"

        class DummyDataset:
            records = [DummyRecord()]

        preds = [9999.0]  # Unphysical value
        sanity = perform_physical_sanity_checks(DummyDataset(), preds, stype="JV")
        self.assertFalse(sanity.is_physically_sane)
        self.assertEqual(sanity.out_of_range_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
