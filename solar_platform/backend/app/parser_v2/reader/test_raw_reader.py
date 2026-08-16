"""
Unit tests for Stage 1 parser_v2 RawReader and RawModel.

Tests cover:
  1. Reading in-memory synthetic workbooks with formulas, merged cells,
     formatting, empty cells, and various data types.
  2. Exact cell provenance preservation (sheet, row, col, raw_value).
  3. Reading the real Data_file.xlsx using read_workbook.
  4. Error handling for non-Excel files.

Run from: solar_platform/backend/
Command:  python -m unittest app.parser_v2.reader.test_raw_reader -v
"""
from __future__ import annotations

import io
import unittest
from pathlib import Path

import openpyxl

from app.parser_v2.reader.raw_model import RawCellType, RawWorkbook, RawSheet, RawCell
from app.parser_v2.reader.raw_reader import read_workbook

WORKBOOK_PATH = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\Data_file.xlsx")


def _create_synthetic_excel_bytes() -> bytes:
    """
    Creates a synthetic workbook in memory with known features:
    - Sheet 'TestSheet'
    - Cell A1 (0,0): String header "Header 1", bold
    - Cell B1 (0,1): String header "Header 2", bold
    - Cell A2 (1,0): Numeric 10.0
    - Cell B2 (1,1): Numeric 20.0
    - Cell A3 (2,0): Numeric 30.0
    - Cell B3 (2,1): Formula "=SUM(A2:B2)"
    - Merged cells C1:D2 (0,2 to 1,3) with value "Merged Title"
    - Cell E5 (4,4): Empty cell surrounded by empty space
    - Cell A4 (3,0): Boolean True
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "TestSheet"

    ws["A1"] = "Header 1"
    ws["A1"].font = openpyxl.styles.Font(bold=True)

    ws["B1"] = "Header 2"
    ws["B1"].font = openpyxl.styles.Font(bold=True)

    ws["A2"] = 10.0
    ws["B2"] = 20.0
    ws["A3"] = 30.0
    ws["B3"] = "=SUM(A2:B2)"

    ws.merge_cells("C1:D2")
    ws["C1"] = "Merged Title"

    ws["A4"] = True

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestRawReaderSynthetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.excel_bytes = _create_synthetic_excel_bytes()
        cls.raw_wb = read_workbook("synthetic.xlsx", cls.excel_bytes)

    def test_workbook_structure(self):
        self.assertEqual(self.raw_wb.filename, "synthetic.xlsx")
        self.assertEqual(self.raw_wb.n_sheets, 1)
        self.assertIn("TestSheet", self.raw_wb.sheet_names())

    def test_sheet_dimensions(self):
        sheet = self.raw_wb.get_sheet("TestSheet")
        self.assertIsNotNone(sheet)
        self.assertGreaterEqual(sheet.n_rows, 4)
        self.assertGreaterEqual(sheet.n_cols, 4)

    def test_raw_cell_provenance_and_types(self):
        sheet = self.raw_wb.get_sheet("TestSheet")

        # Cell A1 (row=0, col=0)
        cell_a1 = sheet.get(0, 0)
        self.assertEqual(cell_a1.row, 0)
        self.assertEqual(cell_a1.col, 0)
        self.assertEqual(cell_a1.data_type, RawCellType.STRING)
        self.assertEqual(cell_a1.raw_value, "Header 1")
        self.assertTrue(cell_a1.is_bold)

        # Cell A2 (row=1, col=0)
        cell_a2 = sheet.get(1, 0)
        self.assertEqual(cell_a2.row, 1)
        self.assertEqual(cell_a2.col, 0)
        self.assertEqual(cell_a2.data_type, RawCellType.NUMERIC)
        self.assertEqual(cell_a2.raw_value, 10.0)

        # Cell A4 (row=3, col=0)
        cell_a4 = sheet.get(3, 0)
        self.assertEqual(cell_a4.data_type, RawCellType.BOOLEAN)
        self.assertEqual(cell_a4.raw_value, True)

    def test_formula_preservation(self):
        sheet = self.raw_wb.get_sheet("TestSheet")

        # Cell B3 (row=2, col=1) contains "=SUM(A2:B2)"
        cell_b3 = sheet.get(2, 1)
        self.assertEqual(cell_b3.data_type, RawCellType.FORMULA)
        self.assertEqual(cell_b3.formula_str, "=SUM(A2:B2)")

    def test_merged_cells_preservation(self):
        sheet = self.raw_wb.get_sheet("TestSheet")

        # Anchor cell C1 (row=0, col=2)
        cell_c1 = sheet.get(0, 2)
        self.assertTrue(cell_c1.is_merged)
        self.assertIsNone(cell_c1.merge_parent)
        self.assertEqual(cell_c1.raw_value, "Merged Title")

        # Non-anchor cell D2 (row=1, col=3) in merged region C1:D2
        cell_d2 = sheet.get(1, 3)
        self.assertTrue(cell_d2.is_merged)
        self.assertEqual(cell_d2.merge_parent, (0, 2))

        # Non-merged cell A2 (row=1, col=0)
        cell_a2 = sheet.get(1, 0)
        self.assertFalse(cell_a2.is_merged)
        self.assertIsNone(cell_a2.merge_parent)

    def test_empty_cells_preservation(self):
        sheet = self.raw_wb.get_sheet("TestSheet")

        # Cell C3 (row=2, col=2) is empty inside the bounding box
        cell_c3 = sheet.get(2, 2)
        self.assertEqual(cell_c3.data_type, RawCellType.EMPTY)
        self.assertIsNone(cell_c3.raw_value)
        self.assertEqual(cell_c3.display_str, "")

    def test_out_of_bounds_returns_none(self):
        sheet = self.raw_wb.get_sheet("TestSheet")
        self.assertIsNone(sheet.get(-1, 0))
        self.assertIsNone(sheet.get(1000, 1000))


class TestRawReaderRealWorkbook(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not WORKBOOK_PATH.exists():
            raise unittest.SkipTest(f"Real workbook not found at {WORKBOOK_PATH}")
        with open(WORKBOOK_PATH, "rb") as f:
            cls.file_bytes = f.read()
        cls.raw_wb = read_workbook("Data_file.xlsx", cls.file_bytes)

    def test_sheets_loaded(self):
        self.assertEqual(self.raw_wb.filename, "Data_file.xlsx")
        self.assertEqual(self.raw_wb.n_sheets, 11)
        expected_sheets = ["ETL", "PVK", "HTL", "Temperature", "Defect Density",
                           "Interface density", "Donor Con", "Rsh", "Rs", "Optimized", "Sheet1"]
        self.assertEqual(self.raw_wb.sheet_names(), expected_sheets)

    def test_etl_sheet_grid(self):
        sheet = self.raw_wb.get_sheet("ETL")
        self.assertIsNotNone(sheet)
        self.assertGreaterEqual(sheet.n_rows, 60)
        self.assertGreaterEqual(sheet.n_cols, 40)

        # Check cell (0, 0) -> header 'Jsc'
        cell_0_0 = sheet.get(0, 0)
        self.assertIsNotNone(cell_0_0)
        self.assertEqual(cell_0_0.row, 0)
        self.assertEqual(cell_0_0.col, 0)
        self.assertEqual(cell_0_0.raw_value, "Jsc")

        # Check cell (1, 1) -> numeric 0 (voltage 0.0)
        cell_1_1 = sheet.get(1, 1)
        self.assertIsNotNone(cell_1_1)
        self.assertEqual(cell_1_1.data_type, RawCellType.NUMERIC)
        self.assertEqual(float(cell_1_1.raw_value), 0.0)


class TestRawReaderErrorHandling(unittest.TestCase):

    def test_non_excel_bytes_raises_value_error(self):
        invalid_bytes = b"This is not an excel file content"
        with self.assertRaises(ValueError) as ctx:
            read_workbook("invalid.txt", invalid_bytes)
        self.assertIn("Cannot read workbook", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
