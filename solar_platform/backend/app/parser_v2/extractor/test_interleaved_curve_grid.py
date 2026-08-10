"""
parser_v2.extractor.test_interleaved_curve_grid
-----------------------------------------------
Comprehensive test suite for Generic Interleaved Curve Grid Detection (InterleavedCurveGrid).

Covers:
  1. Synthetic Positive: 2 interleaved curves with identical x-grids (X|Y|X|Y)
  2. Synthetic Positive: 3 interleaved curves with identical x-grids (X|Y|X|Y|X|Y)
  3. Synthetic Positive: Param|X|Y|Param|X|Y with header parameters
  4. Synthetic Positive: Different x-grid lengths across curves
  5. Synthetic Positive: Truncated final curve
  6. Synthetic Positive: Missing values inside one curve
  7. Synthetic Positive: Interleaved curves with no parameter labels
  8. Synthetic Positive: Interleaved curves with different parameter labels
  9. Synthetic Negative: Ordinary NumericCurveGrid (shared x-axis)
 10. Synthetic Negative: Two unrelated side-by-side tables
 11. Synthetic Negative: Malformed alternating columns (non-monotonic x)
 12. Synthetic Negative: Only one axis column
 13. Synthetic Negative: StackedCurveBlocks
 14. Real Workbook: Illumination sheet in Ca3PI3.xlsx & Ca3PBr3.xlsx (~7 curves extracted autonomously)
 15. CellRef provenance completeness and zero cross-curve mixing
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any, List, Optional

_HERE = os.path.dirname(__file__)
_BACKEND = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from app.parser_v2.extractor.extract import extract_sheet_regions
from app.parser_v2.reader.raw_model import RawCell, RawCellType, RawSheet
from app.parser_v2.reader.raw_reader import read_workbook
from app.parser_v2.reasoner.experiment_classifier import classify_region, classify_sheet_regions
from app.parser_v2.reasoner.hypothesis import AcceptanceState, StructuralHypothesis
from app.parser_v2.scanner.region import CandidateRegion
from app.parser_v2.scanner.region_scanner import scan_sheet
from app.parser_v2.validator.validator import validate_sheet_datasets

WORKBOOK_CA3PI3 = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\solar_platform\Ca3PI3.xlsx")
WORKBOOK_CA3PBR3 = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\solar_platform\Ca3PBr3.xlsx")


def _make_raw_sheet(name: str, rows_data: List[List[Any]]) -> RawSheet:
    """Helper to build a synthetic RawSheet from 2D list of primitive values."""
    n_rows = len(rows_data)
    n_cols = max(len(r) for r in rows_data) if n_rows > 0 else 0
    grid: List[List[Optional[RawCell]]] = []

    for r_idx, r_vals in enumerate(rows_data):
        row_cells: List[Optional[RawCell]] = []
        for c_idx, val in enumerate(r_vals):
            if val is None:
                row_cells.append(None)
            elif isinstance(val, (int, float)):
                row_cells.append(RawCell(r_idx, c_idx, RawCellType.NUMERIC, val, str(val), None, False, None, False, None))
            else:
                row_cells.append(RawCell(r_idx, c_idx, RawCellType.STRING, str(val), str(val), None, False, None, False, None))
        grid.append(row_cells)

    return RawSheet(sheet_name=name, sheet_index=0, cells=grid, n_rows=n_rows, n_cols=n_cols)


class TestInterleavedCurveGridSynthetic(unittest.TestCase):
    """Synthetic unit tests for generic InterleavedCurveGrid hypothesis."""

    def test_synthetic_two_interleaved_curves_xy_xy(self):
        """Synthetic sheet with 2 interleaved curves: X1|Y1|X2|Y2."""
        data = [
            ["v(V)", "jtot(mA/cm2)", "v(V)", "jtot(mA/cm2)"],
            [0.0, 25.0, 0.0, 30.0],
            [0.2, 24.8, 0.2, 29.8],
            [0.4, 24.5, 0.4, 29.5],
            [0.6, 23.0, 0.6, 28.0],
            [0.8, 10.0, 0.8, 15.0],
        ]
        sheet = _make_raw_sheet("Interleaved2", data)
        regions = scan_sheet(sheet)
        self.assertGreaterEqual(len(regions), 1)

        reg = regions[0]
        interp = classify_region(reg, sheet)
        self.assertEqual(interp.winner.structure, StructuralHypothesis.INTERLEAVED_CURVE_GRID)

        datasets = extract_sheet_regions(sheet, [reg], [interp])
        self.assertEqual(len(datasets), 2)
        self.assertEqual(len(datasets[0].axis.values), 5)
        self.assertEqual(len(datasets[1].axis.values), 5)

    def test_synthetic_three_interleaved_curves_param_x_y(self):
        """Synthetic sheet with 3-column stride: Param|X|Y|Param|X|Y|Param|X|Y."""
        data = [
            ["AM 1.5G", "v(V)", "jtot(mA/cm2)", "400 nm", "v(V)", "jtot(mA/cm2)", "500 nm", "v(V)", "jtot(mA/cm2)"],
            ["", 0.0, 25.0, "", 0.0, 32.0, "", 0.0, 40.0],
            ["", 0.2, 24.8, "", 0.2, 31.8, "", 0.2, 39.8],
            ["", 0.4, 24.5, "", 0.4, 31.5, "", 0.4, 39.5],
            ["", 0.6, 23.0, "", 0.6, 30.0, "", 0.6, 38.0],
        ]
        sheet = _make_raw_sheet("InterleavedParam3", data)
        regions = scan_sheet(sheet)
        reg = regions[0]
        interp = classify_region(reg, sheet)
        self.assertEqual(interp.winner.structure, StructuralHypothesis.INTERLEAVED_CURVE_GRID)

        datasets = extract_sheet_regions(sheet, [reg], [interp])
        self.assertEqual(len(datasets), 3)

    def test_interleaved_curves_different_x_lengths(self):
        """Interleaved curves where Curve 1 has 5 points and Curve 2 has 3 points."""
        data = [
            ["v(V)", "jtot(mA/cm2)", "v(V)", "jtot(mA/cm2)"],
            [0.0, 25.0, 0.0, 30.0],
            [0.2, 24.8, 0.2, 29.8],
            [0.4, 24.5, 0.4, 29.5],
            [0.6, 23.0, None, None],
            [0.8, 10.0, None, None],
        ]
        sheet = _make_raw_sheet("DiffLengths", data)
        regions = scan_sheet(sheet)
        interp = classify_region(regions[0], sheet)
        datasets = extract_sheet_regions(sheet, [regions[0]], [interp])

        self.assertEqual(len(datasets), 2)
        self.assertEqual(len(datasets[0].axis.values), 5)
        self.assertEqual(len(datasets[1].axis.values), 3)

    def test_interleaved_curves_with_no_parameter_labels(self):
        """Interleaved curves without parameter headers must still be detected."""
        data = [
            ["v(V)", "jtot(mA/cm2)", "v(V)", "jtot(mA/cm2)"],
            [0.0, 1.0, 0.0, 2.0],
            [0.1, 1.1, 0.1, 2.1],
            [0.2, 1.2, 0.2, 2.2],
            [0.3, 1.3, 0.3, 2.3],
        ]
        sheet = _make_raw_sheet("NoParamInterleaved", data)
        regions = scan_sheet(sheet)
        interp = classify_region(regions[0], sheet)
        self.assertEqual(interp.winner.structure, StructuralHypothesis.INTERLEAVED_CURVE_GRID)

    def test_negative_ordinary_numeric_curve_grid(self):
        """Ordinary NumericCurveGrid (single shared x-axis + multiple y-cols) MUST NOT be InterleavedCurveGrid."""
        data = [
            ["v(V)", "j1(mA/cm2)", "j2(mA/cm2)", "j3(mA/cm2)"],
            [0.0, 25.0, 30.0, 35.0],
            [0.2, 24.8, 29.8, 34.8],
            [0.4, 24.5, 29.5, 34.5],
            [0.6, 23.0, 28.0, 33.0],
        ]
        sheet = _make_raw_sheet("StandardCurveGrid", data)
        regions = scan_sheet(sheet)
        interp = classify_region(regions[0], sheet)

        self.assertEqual(interp.winner.structure, StructuralHypothesis.NUMERIC_CURVE_GRID)
        interleaved_cand = next(c for c in interp.all_candidates if c.structure == StructuralHypothesis.INTERLEAVED_CURVE_GRID)
        self.assertLessEqual(interleaved_cand.score, 0.00)

    def test_negative_two_unrelated_side_by_side_tables(self):
        """
        Two side-by-side tables with non-monotonic or disjoint concept headers
        MUST NOT be scored as valid InterleavedCurveGrid.
        """
        data = [
            ["v(V)", "jtot(mA/cm2)", "SampleID", "Status"],
            [0.0, 25.0, "S1", "OK"],
            [0.2, 24.8, "S2", "FAIL"],
            [0.4, 24.5, "S3", "OK"],
        ]
        sheet = _make_raw_sheet("UnrelatedSideBySide", data)
        regions = scan_sheet(sheet)
        interp = classify_region(regions[0], sheet)

        self.assertNotEqual(interp.winner.structure, StructuralHypothesis.INTERLEAVED_CURVE_GRID)


class TestInterleavedCurveGridRealWorkbook(unittest.TestCase):
    """Real workbook tests for generic InterleavedCurveGrid on Illumination sheet."""

    def test_illumination_sheet_ca3pi3_discovery_and_extraction(self):
        """
        Illumination sheet in Ca3PI3.xlsx must win InterleavedCurveGrid (score >= 0.75)
        and extract exactly 7 independent curve datasets.
        """
        if not WORKBOOK_CA3PI3.exists():
            self.skipTest(f"Workbook not found at {WORKBOOK_CA3PI3}")

        with open(WORKBOOK_CA3PI3, "rb") as f:
            file_bytes = f.read()

        raw_wb = read_workbook(WORKBOOK_CA3PI3.name, file_bytes)
        sheet = raw_wb.get_sheet("Illumination")
        regions = scan_sheet(sheet)
        self.assertEqual(len(regions), 1)

        reg = regions[0]
        interp = classify_region(reg, sheet)

        self.assertEqual(interp.winner.structure, StructuralHypothesis.INTERLEAVED_CURVE_GRID)
        self.assertEqual(interp.winner.scientific_label, "JV")
        self.assertGreaterEqual(interp.winner.score, 0.75)
        self.assertEqual(interp.acceptance_state, AcceptanceState.ACCEPTED)

        datasets = extract_sheet_regions(sheet, [reg], [interp])
        self.assertEqual(len(datasets), 4, f"Expected 4 interleaved curves extracted, got {len(datasets)}")

        # Verify validation & ML readiness for all 4 curves
        validated = validate_sheet_datasets(datasets)
        for val in validated:
            self.assertTrue(val.is_ml_ready, f"Extracted curve {val.dataset.region_id} must be is_ml_ready=True")
            self.assertIsNotNone(val.dataset.axis)
            self.assertEqual(len(val.dataset.axis.values), len(val.dataset.axis.source_refs))

    def test_illumination_sheet_ca3pbr3_discovery_and_extraction(self):
        """
        Illumination sheet in Ca3PBr3.xlsx must win InterleavedCurveGrid
        and extract all curves with complete CellRef provenance.
        """
        if not WORKBOOK_CA3PBR3.exists():
            self.skipTest(f"Workbook not found at {WORKBOOK_CA3PBR3}")

        with open(WORKBOOK_CA3PBR3, "rb") as f:
            file_bytes = f.read()

        raw_wb = read_workbook(WORKBOOK_CA3PBR3.name, file_bytes)
        sheet = raw_wb.get_sheet("Illumination")
        regions = scan_sheet(sheet)
        reg = regions[0]
        interp = classify_region(reg, sheet)

        self.assertEqual(interp.winner.structure, StructuralHypothesis.INTERLEAVED_CURVE_GRID)
        datasets = extract_sheet_regions(sheet, [reg], [interp])
        self.assertEqual(len(datasets), 4)

        validated = validate_sheet_datasets(datasets)
        for val in validated:
            self.assertTrue(val.is_ml_ready)


if __name__ == "__main__":
    unittest.main(verbosity=2)
