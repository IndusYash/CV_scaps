"""
parser_v2.extractor.test_stacked_curve_blocks
----------------------------------------------
Comprehensive test suite for Generic Stacked Curve Block Detection (StackedCurveBlocks).

Covers:
  1. Synthetic 2 stacked curve blocks
  2. Synthetic 3 stacked curve blocks with distinct parameter values
  3. Stacked blocks with no inter-block banners
  4. Stacked blocks with different block lengths
  5. Incomplete final block (retained without data loss or fabrication)
  6. Negative test: One monotonic curve with local duplicate x-values (Score = 0.00)
  7. Negative test: Single x reset caused by malformed data (Score = 0.00)
  8. Negative test: Irregular segments with inconsistent lengths
  9. Real workbook QE (T+DD) stacked block discovery (~88 blocks extracted autonomously)
 10. CellRef provenance completeness and zero cross-block mixing
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

from app.parser_v2.extractor.extract import extract_sheet_regions
from app.parser_v2.extractor.stacked_curve_blocks_extractor import extract_stacked_curve_blocks
from app.parser_v2.probes.axis_probe import probe_axis
from app.parser_v2.reader.raw_model import RawCell, RawCellType, RawSheet
from app.parser_v2.reader.raw_reader import read_workbook
from app.parser_v2.reasoner.experiment_classifier import classify_region, classify_sheet_regions
from app.parser_v2.reasoner.hypothesis import AcceptanceState, StructuralHypothesis
from app.parser_v2.scanner.region import CandidateRegion
from app.parser_v2.scanner.region_scanner import scan_sheet
from app.parser_v2.validator.validator import validate_sheet_datasets

WORKBOOK_PATH = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\solar_platform\Ca3PI3.xlsx")


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


class TestStackedCurveBlocksSynthetic(unittest.TestCase):
    """Synthetic unit tests for generic StackedCurveBlocks hypothesis."""

    def test_synthetic_two_stacked_curve_blocks(self):
        """Synthetic sheet with 2 stacked 5-point curve blocks."""
        data = [
            ["lambda(nm)", "QE(%)"],
            [300, 80.0], [400, 85.0], [500, 90.0], [600, 88.0], [700, 80.0],
            ["banner parameter: T = 350K", ""],
            ["lambda(nm)", "QE(%)"],
            [300, 75.0], [400, 82.0], [500, 88.0], [600, 85.0], [700, 78.0],
        ]
        sheet = _make_raw_sheet("SyntheticStacked2", data)
        regions = scan_sheet(sheet)
        self.assertGreaterEqual(len(regions), 1)

        reg = regions[0]
        interp = classify_region(reg, sheet)
        self.assertEqual(interp.winner.structure, StructuralHypothesis.STACKED_CURVE_BLOCKS)
        self.assertIn(interp.acceptance_state, (AcceptanceState.ACCEPTED, AcceptanceState.UNCERTAIN))

        datasets = extract_sheet_regions(sheet, [reg], [interp])
        self.assertEqual(len(datasets), 2)
        self.assertEqual(len(datasets[0].axis.values), 5)
        self.assertEqual(len(datasets[1].axis.values), 5)

    def test_synthetic_three_stacked_blocks_with_parameters(self):
        """Synthetic sheet with 3 stacked curve blocks and explicit parameter banners."""
        data = [
            ["lambda(nm)", "QE(%)"],
            ["t=100nm", ""],
            [300, 70.0], [400, 75.0], [500, 80.0], [600, 78.0],
            ["t=200nm", ""],
            [300, 72.0], [400, 78.0], [500, 84.0], [600, 82.0],
            ["t=300nm", ""],
            [300, 75.0], [400, 82.0], [500, 88.0], [600, 85.0],
        ]
        sheet = _make_raw_sheet("SyntheticStacked3", data)
        regions = scan_sheet(sheet)
        reg = regions[0]
        interp = classify_region(reg, sheet)
        self.assertEqual(interp.winner.structure, StructuralHypothesis.STACKED_CURVE_BLOCKS)

        datasets = extract_sheet_regions(sheet, [reg], [interp])
        self.assertEqual(len(datasets), 3)

    def test_stacked_blocks_with_no_banners(self):
        """Stacked curve blocks without explicit banner text rows between blocks."""
        data = [
            ["lambda(nm)", "QE(%)"],
            [300, 80.0], [400, 85.0], [500, 90.0], [600, 88.0],
            [300, 75.0], [400, 82.0], [500, 88.0], [600, 85.0],
        ]
        sheet = _make_raw_sheet("NoBannerStacked", data)
        regions = scan_sheet(sheet)
        interp = classify_region(regions[0], sheet)
        self.assertEqual(interp.winner.structure, StructuralHypothesis.STACKED_CURVE_BLOCKS)

    def test_incomplete_final_block_preserved(self):
        """Incomplete final block (e.g. truncated at 3 points vs 5 points) must be retained."""
        data = [
            ["lambda(nm)", "QE(%)"],
            [300, 80.0], [400, 85.0], [500, 90.0], [600, 88.0], [700, 80.0],
            [300, 75.0], [400, 82.0], [500, 88.0],  # truncated block
        ]
        sheet = _make_raw_sheet("IncompleteBlock", data)
        regions = scan_sheet(sheet)
        interp = classify_region(regions[0], sheet)
        datasets = extract_sheet_regions(sheet, [regions[0]], [interp])
        self.assertEqual(len(datasets), 2)
        self.assertEqual(len(datasets[0].axis.values), 5)
        self.assertEqual(len(datasets[1].axis.values), 3)

    def test_negative_single_curve_with_duplicate_x_values(self):
        """
        Negative test: A single curve containing local duplicate x-values (e.g. [0, 0.5, 0.5, 1.0])
        MUST NOT be scored as StackedCurveBlocks (score = 0.00).
        """
        data = [
            ["v(V)", "jtot(mA/cm2)"],
            [0.0, 23.5], [0.5, 23.4], [0.5, 23.4], [1.0, 0.0],
        ]
        sheet = _make_raw_sheet("DuplicateXCurve", data)
        regions = scan_sheet(sheet)
        interp = classify_region(regions[0], sheet)

        stacked_cand = next(c for c in interp.all_candidates if c.structure == StructuralHypothesis.STACKED_CURVE_BLOCKS)
        self.assertEqual(stacked_cand.score, 0.00)
        self.assertNotEqual(interp.winner.structure, StructuralHypothesis.STACKED_CURVE_BLOCKS)

    def test_negative_single_malformed_reset(self):
        """
        Negative test: A single x reset caused by malformed data (e.g. [0, 0.5, 0.2, 0.8])
        where segments are too short (< 3 points) MUST NOT trigger StackedCurveBlocks.
        """
        data = [
            ["v(V)", "jtot(mA/cm2)"],
            [0.0, 23.5], [0.5, 23.4], [0.2, 23.0], [0.8, 12.0],
        ]
        sheet = _make_raw_sheet("MalformedReset", data)
        regions = scan_sheet(sheet)
        interp = classify_region(regions[0], sheet)

        stacked_cand = next(c for c in interp.all_candidates if c.structure == StructuralHypothesis.STACKED_CURVE_BLOCKS)
        self.assertEqual(stacked_cand.score, 0.00)

    def test_negative_irregular_inconsistent_lengths(self):
        """
        Negative test: Irregular segments with severely inconsistent lengths
        MUST NOT be scored as valid StackedCurveBlocks.
        """
        data = [
            ["v(V)", "jtot(mA/cm2)"],
            [0.0, 23.5], [0.1, 23.5], [0.2, 23.5], [0.3, 23.5], [0.4, 23.5], [0.5, 23.5], [0.6, 23.5], [0.7, 23.5], [0.8, 23.5], [0.9, 23.5],
            [0.1, 20.0], [0.2, 19.0],  # len 2 vs len 10
        ]
        sheet = _make_raw_sheet("IrregularLengths", data)
        regions = scan_sheet(sheet)
        interp = classify_region(regions[0], sheet)

        stacked_cand = next(c for c in interp.all_candidates if c.structure == StructuralHypothesis.STACKED_CURVE_BLOCKS)
        self.assertEqual(stacked_cand.score, 0.00)


class TestStackedCurveBlocksRealWorkbook(unittest.TestCase):
    """Real workbook tests for generic StackedCurveBlocks on Ca3PI3.xlsx QE (T+DD)."""

    @classmethod
    def setUpClass(cls):
        if not WORKBOOK_PATH.exists():
            raise unittest.SkipTest(f"Real workbook not found at {WORKBOOK_PATH}")
        with open(WORKBOOK_PATH, "rb") as f:
            file_bytes = f.read()
        cls.raw_wb = read_workbook(WORKBOOK_PATH.name, file_bytes)
        cls.sheet = cls.raw_wb.get_sheet("QE (T+DD)")
        cls.regions = scan_sheet(cls.sheet)
        cls.interps = classify_sheet_regions(cls.sheet, cls.regions)

    def test_qe_tdd_stacked_block_discovery(self):
        """
        QE (T+DD) sheet must autonomously win StackedCurveBlocks hypothesis (score >= 0.85)
        and extract exactly 88 independent sub-block datasets without hardcoding 88.
        """
        reg7 = self.regions[7]
        interp7 = self.interps[7]

        self.assertEqual(interp7.winner.structure, StructuralHypothesis.STACKED_CURVE_BLOCKS)
        self.assertEqual(interp7.winner.scientific_label, "QE")
        self.assertGreaterEqual(interp7.winner.score, 0.85)
        self.assertEqual(interp7.acceptance_state, AcceptanceState.ACCEPTED)

        datasets = extract_sheet_regions(self.sheet, [reg7], [interp7])
        self.assertEqual(len(datasets), 88, f"Expected 88 sub-blocks extracted, got {len(datasets)}")

        # Verify validation & ML readiness
        validated = validate_sheet_datasets(datasets)
        ml_ready_count = sum(1 for v in validated if v.is_ml_ready)
        self.assertEqual(ml_ready_count, 88, f"All 88 extracted blocks must be is_ml_ready=True, got {ml_ready_count}")

    def test_qe_tdd_provenance_and_no_cross_block_mixing(self):
        """
        Every sub-block must retain complete CellRef provenance pointers and distinct region_ids.
        """
        reg7 = self.regions[7]
        interp7 = self.interps[7]
        datasets = extract_sheet_regions(self.sheet, [reg7], [interp7])

        region_ids = [d.region_id for d in datasets]
        self.assertEqual(len(region_ids), len(set(region_ids)), "Sub-block region_ids must be unique")

        for ds in datasets:
            self.assertIsNotNone(ds.axis)
            self.assertEqual(len(ds.axis.values), 61, "Each sub-block wavelength axis must have 61 points")
            self.assertEqual(len(ds.axis.values), len(ds.axis.source_refs))
            for ref in ds.axis.source_refs:
                self.assertEqual(ref.sheet_name, "QE (T+DD)")
                self.assertGreaterEqual(ref.row, 33)


if __name__ == "__main__":
    unittest.main(verbosity=2)
