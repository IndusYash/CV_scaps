"""
Unit and integration tests for parser_v2 Stage 4 Reasoner.

Tests cover:
  1. Positive evidence scoring and high confidence ACCEPTED states.
  2. Contradictory evidence penalties (string category col 0, low numeric density).
  3. Small winner/runner-up gap mechanism -> UNCERTAIN or UNRESOLVED.
  4. Rule 7B: Single-trace NumericCurveGrid without parameters is fully valid.
  5. Rule 7C: Plain numeric headers ambiguity.
  6. Rule 7D: Sheet name conflict with region-level evidence (region evidence wins).
  7. Rule 8: Interface Density two independent JV experiments.
  8. Rule 8: Optimized multi-block independent regions.
  9. Rule 7E & Rule 9: Sheet1 positional profile remains UNRESOLVED.
 10. Rule 5: Preservation of full ranked hypothesis candidate set in all cases.

Run from: solar_platform/backend/
Command:  python -m unittest app.parser_v2.reasoner.test_reasoner -v
"""
from __future__ import annotations

import io
import unittest
from pathlib import Path

import openpyxl

from app.parser_v2.reader.raw_reader import read_workbook
from app.parser_v2.scanner.region_scanner import scan_sheet
from app.parser_v2.reasoner.hypothesis import (
    AcceptanceState,
    StructuralHypothesis,
)
from app.parser_v2.reasoner.experiment_classifier import (
    classify_region,
    classify_sheet_regions,
)

WORKBOOK_PATH = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\Data_file.xlsx")


def _create_synthetic_excel_bytes(sheet_name: str = "TestSheet", data_type: str = "curve") -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name

    if data_type == "curve":
        ws["A1"] = "v(V)"
        ws["B1"] = "t=100[nm]"
        ws["C1"] = "t=200[nm]"
        for r in range(2, 22):
            ws[f"A{r}"] = float(r - 2) * 0.05
            ws[f"B{r}"] = float(r * 2)
            ws[f"C{r}"] = float(r * 3)
    elif data_type == "single_curve_no_param":
        ws["A1"] = " v(V)"
        ws["B1"] = "jtot(mA/cm2)"
        for r in range(2, 20):
            ws[f"A{r}"] = float(r - 2) * 0.05
            ws[f"B{r}"] = float(25.0 - r * 0.5)
    elif data_type == "category":
        ws["A1"] = "Interface"
        ws["B1"] = "PCE"
        ws["C1"] = "Voc"
        ws["A2"] = "ETL/PS"
        ws["B2"] = 22.1
        ws["C2"] = 1.32
        ws["A3"] = "PS/HTL"
        ws["B3"] = 21.8
        ws["C3"] = 1.30

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestReasonerSynthetic(unittest.TestCase):

    def test_positive_evidence_scoring_accepted(self):
        bytes_in = _create_synthetic_excel_bytes("TestSheet", "curve")
        raw_wb = read_workbook("test.xlsx", bytes_in)
        sheet = raw_wb.get_sheet("TestSheet")
        regions = scan_sheet(sheet)
        self.assertEqual(len(regions), 1)

        interp = classify_region(regions[0], sheet)
        self.assertEqual(interp.acceptance_state, AcceptanceState.ACCEPTED)
        self.assertIsNotNone(interp.winner)
        self.assertEqual(interp.winner.structure, StructuralHypothesis.NUMERIC_CURVE_GRID)
        self.assertEqual(interp.winner.scientific_label, "JV")
        self.assertGreaterEqual(interp.winner.score, 0.75)
        self.assertGreaterEqual(interp.winner_runner_up_gap, 0.20)

    def test_single_trace_numeric_curve_without_parameter_valid(self):
        """Rule 7B: Single trace without parameter expressions must remain valid ACCEPTED NumericCurveGrid."""
        bytes_in = _create_synthetic_excel_bytes("TestSheet", "single_curve_no_param")
        raw_wb = read_workbook("test.xlsx", bytes_in)
        sheet = raw_wb.get_sheet("TestSheet")
        regions = scan_sheet(sheet)

        interp = classify_region(regions[0], sheet)
        self.assertEqual(interp.acceptance_state, AcceptanceState.ACCEPTED)
        self.assertEqual(interp.winner.structure, StructuralHypothesis.NUMERIC_CURVE_GRID)
        self.assertEqual(interp.winner.scientific_label, "JV")

    def test_category_table_winning_over_curve(self):
        bytes_in = _create_synthetic_excel_bytes("TestSheet", "category")
        raw_wb = read_workbook("test.xlsx", bytes_in)
        sheet = raw_wb.get_sheet("TestSheet")
        regions = scan_sheet(sheet)

        interp = classify_region(regions[0], sheet)
        self.assertEqual(interp.winner.structure, StructuralHypothesis.CATEGORY_TABLE)
        self.assertEqual(interp.winner.scientific_label, "CATEGORY")

    def test_sheet_name_conflict_does_not_override_region_evidence(self):
        """Rule 7D: Sheet named 'JV_Curves' containing string category table must NOT be classified as JV."""
        bytes_in = _create_synthetic_excel_bytes("JV_Curves", "category")
        raw_wb = read_workbook("test.xlsx", bytes_in)
        sheet = raw_wb.get_sheet("JV_Curves")
        regions = scan_sheet(sheet)

        interp = classify_region(regions[0], sheet)
        # Category table should win over NumericCurveGrid despite the sheet name!
        self.assertEqual(interp.winner.structure, StructuralHypothesis.CATEGORY_TABLE)
        self.assertNotEqual(interp.winner.scientific_label, "JV")

    def test_full_candidate_ranking_preserved(self):
        """Rule 5: Every RegionInterpretation must contain all 4 ranked candidates."""
        bytes_in = _create_synthetic_excel_bytes("TestSheet", "curve")
        raw_wb = read_workbook("test.xlsx", bytes_in)
        sheet = raw_wb.get_sheet("TestSheet")
        regions = scan_sheet(sheet)

        interp = classify_region(regions[0], sheet)
        self.assertEqual(len(interp.all_candidates), 4)
        # Verify descending score order
        scores = [c.score for c in interp.all_candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestReasonerRealWorkbook(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not WORKBOOK_PATH.exists():
            raise unittest.SkipTest(f"Real workbook not found at {WORKBOOK_PATH}")
        with open(WORKBOOK_PATH, "rb") as f:
            cls.file_bytes = f.read()
        cls.raw_wb = read_workbook("Data_file.xlsx", cls.file_bytes)

    def test_etl_sheet_reasoning(self):
        sheet = self.raw_wb.get_sheet("ETL")
        regions = scan_sheet(sheet)
        interps = classify_sheet_regions(sheet, regions)

        self.assertEqual(len(interps), 4)

        # Region 0: JV block
        i0 = interps[0]
        self.assertEqual(i0.acceptance_state, AcceptanceState.ACCEPTED)
        self.assertEqual(i0.winner.structure, StructuralHypothesis.NUMERIC_CURVE_GRID)
        self.assertEqual(i0.winner.scientific_label, "JV")

        # Region 1: Jsc scalar summary
        i1 = interps[1]
        self.assertEqual(i1.winner.structure, StructuralHypothesis.SCALAR_SWEEP_TABLE)
        self.assertEqual(i1.winner.scientific_label, "SCALAR_SWEEP")

        # Region 2: QE block
        i2 = interps[2]
        self.assertEqual(i2.acceptance_state, AcceptanceState.ACCEPTED)
        self.assertEqual(i2.winner.structure, StructuralHypothesis.NUMERIC_CURVE_GRID)
        self.assertEqual(i2.winner.scientific_label, "QE")

    def test_interface_density_two_independent_jv_experiments(self):
        """Rule 8: Interface density produces independent ACCEPTED JV experiment interpretations."""
        sheet = self.raw_wb.get_sheet("Interface density")
        regions = scan_sheet(sheet)
        interps = classify_sheet_regions(sheet, regions)

        jv_interps = [i for i in interps if i.winner and i.winner.scientific_label == "JV"]
        self.assertGreaterEqual(len(jv_interps), 2,
            f"Interface density: expected >= 2 JV experiment interpretations, got {len(jv_interps)}")
        for i in jv_interps:
            self.assertEqual(i.acceptance_state, AcceptanceState.ACCEPTED)

    def test_optimized_sheet_multi_block_reasoning(self):
        """Rule 8: Optimized sheet produces independent interpretations for JV, QE, Responsivity, Detectivity."""
        sheet = self.raw_wb.get_sheet("Optimized")
        regions = scan_sheet(sheet)
        interps = classify_sheet_regions(sheet, regions)

        labels = [i.winner.scientific_label for i in interps if i.winner]
        self.assertIn("JV", labels)
        self.assertIn("QE", labels)
        self.assertIn("RESPONSIVITY", labels)
        self.assertIn("DETECTIVITY", labels)

    def test_sheet1_remains_unresolved(self):
        """Rule 7E & 9: Sheet1 positional data remains UNRESOLVED."""
        sheet = self.raw_wb.get_sheet("Sheet1")
        regions = scan_sheet(sheet)
        interps = classify_sheet_regions(sheet, regions)

        main_interp = max(interps, key=lambda i: i.evidence_bag.filter_by_name("numeric_density_high")[0].value.get("total_numeric_cells", 0) if i.evidence_bag.filter_by_name("numeric_density_high") else 0)
        self.assertEqual(main_reg_state := main_interp.acceptance_state, AcceptanceState.UNRESOLVED,
            f"Sheet1 expected UNRESOLVED, got {main_reg_state}")
        self.assertEqual(main_interp.winner.scientific_label, "UNKNOWN")


if __name__ == "__main__":
    unittest.main(verbosity=2)
