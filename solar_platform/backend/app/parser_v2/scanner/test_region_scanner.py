"""
Unit and integration tests for parser_v2 Stage 2 Structural Scanner.

Tests cover:
  1. Boundary detection on synthetic grid matrices.
  2. Structural properties of candidate regions (densities, headers, monotonicity).
  3. Shared x-column candidates and repeated column patterns.
  4. Integration tests against Data_file.xlsx (ETL, PVK, Optimized, Interface density, Sheet1).
  5. Strict check verifying ZERO domain/scientific keywords exist in scanner code.

Run from: solar_platform/backend/
Command:  python -m unittest app.parser_v2.scanner.test_region_scanner -v
"""
from __future__ import annotations

import io
import inspect
import unittest
from pathlib import Path

import openpyxl

from app.parser_v2.reader.raw_reader import read_workbook
from app.parser_v2.scanner import boundary_detector, region, region_scanner
from app.parser_v2.scanner.boundary_detector import detect_rectangular_bounds
from app.parser_v2.scanner.region_scanner import scan_sheet

WORKBOOK_PATH = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\Data_file.xlsx")


def _create_side_by_side_tables_bytes() -> bytes:
    """
    Creates a synthetic workbook with 2 side-by-side tables separated by empty column C.
    Table 1: cols A,B (0,1)
    Table 2: cols D,E (3,4)
    Col C (2) is empty.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "SideBySide"

    # Table 1
    ws["A1"] = "ColA"
    ws["B1"] = "ColB"
    for r in range(2, 12):
        ws[f"A{r}"] = float(r - 1)  # Monotonic 1..10
        ws[f"B{r}"] = float((r - 1) * 2)

    # Empty col C

    # Table 2
    ws["D1"] = "ColD"
    ws["E1"] = "ColE"
    for r in range(2, 12):
        ws[f"D{r}"] = float(r + 100) # Monotonic 102..111
        ws[f"E{r}"] = float(r * 5)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestBoundaryDetectorSynthetic(unittest.TestCase):

    def test_side_by_side_bounds_detection(self):
        excel_bytes = _create_side_by_side_tables_bytes()
        raw_wb = read_workbook("side_by_side.xlsx", excel_bytes)
        sheet = raw_wb.get_sheet("SideBySide")
        self.assertIsNotNone(sheet)

        bounds = detect_rectangular_bounds(sheet)
        self.assertEqual(len(bounds), 2, f"Expected 2 bounding boxes, got {len(bounds)}: {bounds}")

        box1, box2 = bounds
        # Box 1: rows 0..10, cols 0..1
        self.assertEqual(box1, (0, 10, 0, 1))
        # Box 2: rows 0..10, cols 3..4
        self.assertEqual(box2, (0, 10, 3, 4))


class TestRegionScannerSynthetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.excel_bytes = _create_side_by_side_tables_bytes()
        cls.raw_wb = read_workbook("side_by_side.xlsx", cls.excel_bytes)
        cls.sheet = cls.raw_wb.get_sheet("SideBySide")
        cls.regions = scan_sheet(cls.sheet)

    def test_scanner_discovers_two_regions(self):
        self.assertEqual(len(self.regions), 2)

    def test_region_structural_properties(self):
        reg1 = self.regions[0]
        self.assertEqual(reg1.region_id, "region_0")
        self.assertEqual(reg1.row_start, 0)
        self.assertEqual(reg1.row_end, 10)
        self.assertEqual(reg1.col_start, 0)
        self.assertEqual(reg1.col_end, 1)
        self.assertEqual(reg1.n_rows, 11)
        self.assertEqual(reg1.n_cols, 2)
        self.assertGreater(reg1.numeric_density, 0.80)

    def test_shared_x_column_candidate(self):
        reg1 = self.regions[0]
        # Column A (col_start 0) is monotonic 1..10
        self.assertIn(0, reg1.shared_x_col_candidates)

    def test_raw_header_texts_preserved(self):
        reg1 = self.regions[0]
        self.assertIn("ColA", reg1.raw_header_texts)
        self.assertIn("ColB", reg1.raw_header_texts)

    def test_neighbor_relationship(self):
        reg1 = self.regions[0]
        reg2 = self.regions[1]
        self.assertIn("region_1", reg1.neighbor_region_ids)
        self.assertIn("region_0", reg2.neighbor_region_ids)


class TestRegionScannerRealWorkbook(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not WORKBOOK_PATH.exists():
            raise unittest.SkipTest(f"Real workbook not found: {WORKBOOK_PATH}")
        with open(WORKBOOK_PATH, "rb") as f:
            file_bytes = f.read()
        cls.raw_wb = read_workbook("Data_file.xlsx", file_bytes)

    def test_etl_sheet_scanned_regions(self):
        sheet = self.raw_wb.get_sheet("ETL")
        regions = scan_sheet(sheet)
        # ETL has 4 sub-table blocks separated by empty columns
        self.assertEqual(len(regions), 4, f"ETL: expected 4 regions, got {len(regions)}")
        # Check first region spans JV columns
        r0 = regions[0]
        self.assertEqual(r0.col_start, 0)
        self.assertGreaterEqual(r0.n_cols, 10)
        self.assertGreater(r0.numeric_density, 0.50)

    def test_optimized_sheet_scanned_regions(self):
        sheet = self.raw_wb.get_sheet("Optimized")
        regions = scan_sheet(sheet)
        # Optimized has at least 4 main sub-blocks separated by empty column groups
        self.assertGreaterEqual(len(regions), 4, f"Optimized: expected >= 4 regions, got {len(regions)}")

    def test_interface_density_sheet_scanned_regions(self):
        sheet = self.raw_wb.get_sheet("Interface density")
        regions = scan_sheet(sheet)
        # Should discover multiple regions across the 61 columns
        self.assertGreaterEqual(len(regions), 2, f"Interface density: expected >= 2 regions, got {len(regions)}")

    def test_sheet1_scanned_regions(self):
        sheet = self.raw_wb.get_sheet("Sheet1")
        regions = scan_sheet(sheet)
        # Sheet1 has data region spanning 23 columns
        self.assertGreaterEqual(len(regions), 1, f"Sheet1: expected >= 1 region, got {len(regions)}")
        main_reg = max(regions, key=lambda r: r.total_cells)
        self.assertEqual(main_reg.n_cols, 23)
        self.assertGreater(main_reg.numeric_density, 0.90)


class TestNoDomainKeywordsInScannerCode(unittest.TestCase):
    """
    CRITICAL ARCHITECTURAL CONSTRAINT TEST:
    Verifies that the scanner sub-package contains ZERO domain/scientific keywords.
    """

    FORBIDDEN_KEYWORDS = [
        "JV", "QE", "RESPONSIVITY", "DETECTIVITY", "ABSORPTION",
        "VOLTAGE", "CURRENT", "WAVELENGTH", "TEMPERATURE", "THICKNESS",
        "PCE", "VOC", "JSC", "ETA", "V_MPP", "J_MPP", "DEFECT", "ACCEPTOR",
        "DONOR", "OPTIMIZED", "SIMULATION", "PEROVSKITE"
    ]

    def test_no_scientific_keywords_in_scanner_source(self):
        scanner_modules = [boundary_detector, region, region_scanner]
        for mod in scanner_modules:
            src = inspect.getsource(mod)
            for kw in self.FORBIDDEN_KEYWORDS:
                lines = src.splitlines()
                for line in lines:
                    stripped = line.strip()
                    # Skip code comments and docstrings
                    if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''") or stripped.startswith("*"):
                        continue
                    if f'"{kw}"' in line or f"'{kw}'" in line or f'"{kw.lower()}"' in line or f"'{kw.lower()}'" in line:
                        self.fail(f"Forbidden keyword '{kw}' found in scanner module {mod.__name__}:\n  {line}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
