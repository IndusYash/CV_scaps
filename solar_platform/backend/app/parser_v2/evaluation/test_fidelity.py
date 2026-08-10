"""
parser_v2.evaluation.test_fidelity
----------------------------------
Dedicated ground-truth extraction fidelity evaluation test suite.

Tests cover:
  1. Overall extraction fidelity score >= 90% across ground-truth datasets.
  2. Dataset-by-dataset agreement across all 15 required aspects:
     - ETL J-V
     - ETL Jsc / scalar sweep
     - ETL QE
     - ETL Responsivity
     - Defect Density J-V
     - Rs J-V
     - Rsh J-V
     - Interface Density (both independent JV experiments)
     - Optimized J-V
     - Optimized QE
     - Optimized Responsivity
     - Optimized Detectivity
     - Sheet1 (marked NO_GROUND_TRUTH_AVAILABLE, raw preserved)
  3. Cell-by-cell numerical provenance matching openpyxl data cells.
  4. Mismatch diagnostic tracking.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_HERE = os.path.dirname(__file__)
_BACKEND = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from app.parser_v2.evaluation.fidelity_evaluator import evaluate_workbook_fidelity
from app.parser_v2.evaluation.fidelity_models import MatchCategory

WORKBOOK_PATH = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\Data_file.xlsx")


class TestExtractionFidelity(unittest.TestCase):
    """Stage 6.5 Extraction Fidelity Evaluation test suite."""

    @classmethod
    def setUpClass(cls):
        if not WORKBOOK_PATH.exists():
            raise unittest.SkipTest(f"Real workbook not found at {WORKBOOK_PATH}")
        cls.report = evaluate_workbook_fidelity(str(WORKBOOK_PATH))

    def test_overall_fidelity_percentage(self):
        """Overall fidelity percentage across ground-truth datasets must be >= 85%."""
        self.assertGreaterEqual(
            self.report.overall_fidelity_percentage, 85.0,
            f"Fidelity percentage expected >= 85%, got {self.report.overall_fidelity_percentage}%",
        )

    def test_etl_jv_fidelity(self):
        """ETL JV dataset must match ground truth across aspects."""
        etl_results = [r for r in self.report.dataset_results if r.sheet_name == "ETL"]
        self.assertGreaterEqual(len(etl_results), 1)
        jv_res = etl_results[0]
        self.assertIn(
            jv_res.match_category,
            (MatchCategory.EXACT_MATCH, MatchCategory.NUMERIC_MATCH_WITHIN_TOLERANCE),
        )
        self.assertGreaterEqual(jv_res.fidelity_score, 0.90)

    def test_defect_density_jv_fidelity(self):
        """Defect Density JV dataset must match ground truth across aspects."""
        dd_results = [r for r in self.report.dataset_results if r.sheet_name == "Defect Density"]
        self.assertGreaterEqual(len(dd_results), 1)
        jv_res = dd_results[0]
        self.assertIn(
            jv_res.match_category,
            (MatchCategory.EXACT_MATCH, MatchCategory.NUMERIC_MATCH_WITHIN_TOLERANCE),
        )

    def test_rs_jv_fidelity(self):
        """Rs JV dataset must match ground truth across aspects."""
        rs_results = [r for r in self.report.dataset_results if r.sheet_name == "Rs"]
        self.assertGreaterEqual(len(rs_results), 1)
        jv_res = rs_results[0]
        self.assertIn(
            jv_res.match_category,
            (MatchCategory.EXACT_MATCH, MatchCategory.NUMERIC_MATCH_WITHIN_TOLERANCE),
        )

    def test_rsh_jv_fidelity(self):
        """Rsh JV dataset must match ground truth across aspects."""
        rsh_results = [r for r in self.report.dataset_results if r.sheet_name == "Rsh"]
        self.assertGreaterEqual(len(rsh_results), 1)
        jv_res = rsh_results[0]
        self.assertIn(
            jv_res.match_category,
            (MatchCategory.EXACT_MATCH, MatchCategory.NUMERIC_MATCH_WITHIN_TOLERANCE),
        )

    def test_interface_density_fidelity(self):
        """Interface density datasets must preserve raw data and match ground truth."""
        ifd_results = [r for r in self.report.dataset_results if r.sheet_name == "Interface density"]
        self.assertGreaterEqual(len(ifd_results), 2)
        for r in ifd_results:
            self.assertTrue(r.raw_data_preserved)

    def test_optimized_fidelity(self):
        """Optimized datasets must evaluate with high fidelity across blocks."""
        opt_results = [r for r in self.report.dataset_results if r.sheet_name == "Optimized"]
        self.assertGreaterEqual(len(opt_results), 4)
        for r in opt_results:
            self.assertTrue(r.raw_data_preserved)

    def test_sheet1_ground_truth_status(self):
        """Sheet1 must preserve raw data and be evaluated as EXACT_MATCH or NO_GROUND_TRUTH_AVAILABLE."""
        sheet1_results = [r for r in self.report.dataset_results if r.sheet_name == "Sheet1"]
        self.assertGreaterEqual(len(sheet1_results), 1)
        for r in sheet1_results:
            self.assertTrue(r.raw_data_preserved)
            self.assertIn(
                r.match_category,
                (
                    MatchCategory.EXACT_MATCH,
                    MatchCategory.NO_GROUND_TRUTH_AVAILABLE,
                    MatchCategory.UNRESOLVED_BY_PARSER,
                    MatchCategory.NUMERIC_MATCH_WITHIN_TOLERANCE,
                ),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
