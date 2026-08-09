"""
Golden fixture tests for parser_v2.

These tests assert against the FIXTURE definitions only —
they do NOT call the parser_v2 implementation (which does not exist yet).

Phase 1 of testing:
    Verify that the fixtures themselves are internally consistent and
    that the workbook matches our documented expectations.

Phase 2 of testing (once parser_v2 exists):
    Assert that parser_v2 output satisfies each fixture.

Run from: solar_platform/backend/
Command:  python -m pytest app/parser_v2/fixtures/test_golden_fixtures.py -v
"""
from __future__ import annotations

import sys
import io
import unittest
from pathlib import Path

import openpyxl

# Patch stdout for Windows encoding safety
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.parser_v2.fixtures.golden_fixtures import (
    STRUCTURAL_FIXTURES,
    EXPERIMENT_FIXTURES,
    INVARIANT_FIXTURES,
    PARAMETER_PARSING_FIXTURES,
)

WORKBOOK_PATH = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\Data_file.xlsx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_workbook():
    return openpyxl.load_workbook(WORKBOOK_PATH, data_only=True)


def empty_cols_in_first_n_rows(ws, n_rows=30) -> list:
    """Return 0-indexed column positions empty in first n_rows."""
    return [
        c - 1
        for c in range(1, ws.max_column + 1)
        if all(ws.cell(row=r, column=c).value is None for r in range(1, n_rows + 1))
    ]


def numeric_density(ws, r_start=2, n_rows=50, n_cols=30) -> float:
    total, numeric = 0, 0
    for r in range(r_start, min(ws.max_row + 1, r_start + n_rows)):
        for c in range(1, min(ws.max_column + 1, n_cols + 1)):
            v = ws.cell(row=r, column=c).value
            total += 1
            if isinstance(v, (int, float)):
                numeric += 1
    return numeric / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Phase 1: Workbook ground-truth validation
# Verify that the real workbook matches our documented structural expectations.
# ---------------------------------------------------------------------------

class TestWorkbookStructuralFacts(unittest.TestCase):
    """
    These tests confirm that Data_file.xlsx has the structure we documented.
    They are workbook-level ground truth — not parser tests.
    """

    @classmethod
    def setUpClass(cls):
        if not WORKBOOK_PATH.exists():
            raise unittest.SkipTest(f"Workbook not found: {WORKBOOK_PATH}")
        cls.wb = load_workbook()

    def test_all_expected_sheets_present(self):
        """All 9 fixture sheets must exist in the workbook."""
        expected = {"ETL", "PVK", "Temperature", "Defect Density",
                    "Interface density", "Rsh", "Rs", "Optimized", "Sheet1"}
        actual = set(self.wb.sheetnames)
        missing = expected - actual
        self.assertEqual(missing, set(), f"Missing sheets: {missing}")

    def _fixture_for_sheet(self, sheet_name):
        matches = [f for f in STRUCTURAL_FIXTURES if f["sheet"] == sheet_name]
        self.assertEqual(len(matches), 1, f"Expected exactly 1 structural fixture for {sheet_name}")
        return matches[0]

    def _check_dims(self, sheet_name):
        fixture = self._fixture_for_sheet(sheet_name)
        dims = fixture.get("expected_dims", {})
        ws = self.wb[sheet_name]
        if "n_rows_min" in dims:
            self.assertGreaterEqual(ws.max_row, dims["n_rows_min"],
                f"{sheet_name}: expected >= {dims['n_rows_min']} rows, got {ws.max_row}")
        if "n_rows_max" in dims:
            self.assertLessEqual(ws.max_row, dims["n_rows_max"],
                f"{sheet_name}: expected <= {dims['n_rows_max']} rows, got {ws.max_row}")
        if "n_cols_min" in dims:
            self.assertGreaterEqual(ws.max_column, dims["n_cols_min"],
                f"{sheet_name}: expected >= {dims['n_cols_min']} cols, got {ws.max_column}")
        if "n_cols_max" in dims:
            self.assertLessEqual(ws.max_column, dims["n_cols_max"],
                f"{sheet_name}: expected <= {dims['n_cols_max']} cols, got {ws.max_column}")

    # -- ETL --

    def test_etl_dimensions(self):
        self._check_dims("ETL")

    def test_etl_has_empty_col_separators(self):
        ws = self.wb["ETL"]
        empty = empty_cols_in_first_n_rows(ws)
        self.assertGreaterEqual(len(empty), 4,
            f"ETL: expected >= 4 empty column boundaries, got {len(empty)}: {empty}")

    def test_etl_header_contains_thickness_tokens(self):
        ws = self.wb["ETL"]
        headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
        thickness_headers = [h for h in headers if h and "t=" in str(h).lower()]
        self.assertGreater(len(thickness_headers), 5,
            f"ETL: expected >5 thickness headers, got {len(thickness_headers)}: {thickness_headers[:5]}")

    def test_etl_has_voltage_axis(self):
        ws = self.wb["ETL"]
        headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
        has_voltage = any(h and "v(v)" in str(h).lower() for h in headers)
        self.assertTrue(has_voltage, "ETL: no v(V) axis column found in headers")

    def test_etl_has_wavelength_axis(self):
        ws = self.wb["ETL"]
        headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
        has_wavelength = any(h and "lambda" in str(h).lower() for h in headers)
        self.assertTrue(has_wavelength, "ETL: no lambda(nm) axis column found in headers")

    # -- PVK --

    def test_pvk_dimensions(self):
        self._check_dims("PVK")

    def test_pvk_has_10_thickness_jv_traces(self):
        ws = self.wb["PVK"]
        # First sub-table cols 2-11 should have t=100[nm]..t=1000[nm]
        headers = [ws.cell(row=1, column=c).value for c in range(1, 13)]
        thickness_cols = [h for h in headers if h and "t=" in str(h).lower()]
        self.assertEqual(len(thickness_cols), 10,
            f"PVK JV block: expected 10 t= headers, got {len(thickness_cols)}: {thickness_cols}")

    def test_pvk_scalar_summary_present(self):
        ws = self.wb["PVK"]
        # col 13 (0-idx) = 'thickness(nm)', col 14 = 'Jsc(mA/cm^2)'
        h13 = ws.cell(row=1, column=14).value
        h14 = ws.cell(row=1, column=15).value
        self.assertIsNotNone(h13, "PVK: col 13 header (thickness) missing")
        self.assertIsNotNone(h14, "PVK: col 14 header (Jsc) missing")

    # -- Temperature --

    def test_temperature_dimensions(self):
        self._check_dims("Temperature")

    def test_temperature_has_6_jv_traces(self):
        ws = self.wb["Temperature"]
        headers = [ws.cell(row=1, column=c).value for c in range(3, 10)]
        t_headers = [h for h in headers if h and str(h).startswith("T=")]
        self.assertEqual(len(t_headers), 6,
            f"Temperature JV block: expected 6 T= headers, got {len(t_headers)}: {t_headers}")

    def test_temperature_parameter_values(self):
        ws = self.wb["Temperature"]
        # Extract T= values from header row
        t_values = []
        for c in range(1, ws.max_column + 1):
            h = ws.cell(row=1, column=c).value
            if h and str(h).startswith("T="):
                try:
                    t_values.append(float(str(h).split("=")[1]))
                except (ValueError, IndexError):
                    pass
        self.assertEqual(sorted(set(t_values)), [300, 330, 360, 390, 420, 450],
            f"Temperature: unexpected parameter values: {sorted(set(t_values))}")

    # -- Defect Density --

    def test_defect_density_dimensions(self):
        self._check_dims("Defect Density")

    def test_defect_density_header_format(self):
        """Column headers must use '10^N' format, not numeric values."""
        ws = self.wb["Defect Density"]
        pow_headers = [
            ws.cell(row=1, column=c).value
            for c in range(1, ws.max_column + 1)
            if ws.cell(row=1, column=c).value and "10^" in str(ws.cell(row=1, column=c).value)
        ]
        # Should find 10^13 through 10^18 repeated across 3 sub-tables
        self.assertGreaterEqual(len(pow_headers), 6,
            f"Defect Density: expected >= 6 '10^N' headers, got {len(pow_headers)}: {pow_headers}")

    def test_defect_density_scalar_table_col0_is_strings(self):
        """The scalar summary table col-0 contains string '10^13' etc., not floats."""
        ws = self.wb["Defect Density"]
        # Col 10 (0-idx) is the scalar summary axis column
        col10_vals = [ws.cell(row=r, column=11).value for r in range(2, 9)]
        string_vals = [v for v in col10_vals if isinstance(v, str)]
        self.assertGreater(len(string_vals), 0,
            f"Defect Density scalar col: expected string '10^N' values, got: {col10_vals}")

    # -- Interface density --

    def test_interface_density_dimensions(self):
        self._check_dims("Interface density")

    def test_interface_density_has_two_interface_labels(self):
        """Two interface labels must be present: Sns2/PS and PS/Zn3P2."""
        ws = self.wb["Interface density"]
        all_values = []
        for r in range(1, ws.max_row + 1):
            for c in range(1, ws.max_column + 1):
                v = ws.cell(row=r, column=c).value
                if v:
                    all_values.append(str(v))
        has_sns2_ps = any("Sns2" in v or "sns2" in v.lower() for v in all_values)
        has_ps_zn = any("Zn3P2" in v or "zn3p2" in v.lower() for v in all_values)
        self.assertTrue(has_sns2_ps,
            "Interface density: label 'Sns2/PS' not found in workbook")
        self.assertTrue(has_ps_zn,
            "Interface density: label 'PS/Zn3P2' not found in workbook")

    def test_interface_density_has_two_jv_column_groups(self):
        """Two separate v(V) axis columns must be present (one per experiment)."""
        ws = self.wb["Interface density"]
        vv_cols = [
            c - 1
            for c in range(1, ws.max_column + 1)
            if ws.cell(row=1, column=c).value and "v(v)" in str(ws.cell(row=1, column=c).value).lower()
        ]
        self.assertEqual(len(vv_cols), 2,
            f"Interface density: expected 2 v(V) columns, got {len(vv_cols)}: {vv_cols}")

    # -- Rsh --

    def test_rsh_dimensions(self):
        self._check_dims("Rsh")

    def test_rsh_has_8_jv_traces(self):
        ws = self.wb["Rsh"]
        # cols 2-9 (0-idx) are the Rsh JV traces; header is numeric (1000, 2000, ...)
        trace_cols = [
            ws.cell(row=1, column=c).value
            for c in range(3, 11)
        ]
        numeric_trace_cols = [v for v in trace_cols if isinstance(v, (int, float))]
        self.assertEqual(len(numeric_trace_cols), 8,
            f"Rsh: expected 8 numeric trace column headers, got {len(numeric_trace_cols)}: {numeric_trace_cols}")

    def test_rsh_parameter_values(self):
        ws = self.wb["Rsh"]
        vals = [ws.cell(row=1, column=c).value for c in range(3, 11)]
        self.assertEqual(sorted(vals), [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
            f"Rsh: unexpected parameter values: {vals}")

    # -- Rs --

    def test_rs_dimensions(self):
        self._check_dims("Rs")

    def test_rs_header_encodes_unit(self):
        """Rs headers embed the unit inline: '1 ohm/cm^2'."""
        ws = self.wb["Rs"]
        trace_headers = [ws.cell(row=1, column=c).value for c in range(3, 11)]
        unit_embedded = [h for h in trace_headers if h and "ohm" in str(h).lower()]
        self.assertEqual(len(unit_embedded), 8,
            f"Rs: expected 8 headers with 'ohm' unit, got {len(unit_embedded)}: {unit_embedded}")

    def test_rs_parameter_values_embedded_in_header(self):
        """Numeric part of '1 ohm/cm^2' header is 1, '2 ohm/cm^2' is 2, etc."""
        ws = self.wb["Rs"]
        extracted = []
        for c in range(3, 11):
            h = ws.cell(row=1, column=c).value
            if h:
                try:
                    extracted.append(float(str(h).split()[0]))
                except (ValueError, IndexError):
                    pass
        self.assertEqual(sorted(extracted), [1, 2, 3, 4, 5, 6, 7, 8],
            f"Rs: extracted parameter values: {extracted}")

    # -- Optimized --

    def test_optimized_dimensions(self):
        ws = self.wb["Optimized"]
        self.assertGreater(ws.max_row, 900, "Optimized: expected >900 total rows")
        self.assertGreater(ws.max_column, 20, "Optimized: expected >20 cols")

    def test_optimized_has_four_sub_blocks(self):
        """4 experiment blocks separated by empty column groups."""
        ws = self.wb["Optimized"]
        empty = empty_cols_in_first_n_rows(ws, n_rows=20)
        # Known empty col boundaries: 3, 7, 11, 12, 13, 18
        self.assertIn(3, empty, f"Optimized: expected empty col 3, got {empty}")
        self.assertIn(7, empty, f"Optimized: expected empty col 7, got {empty}")

    def test_optimized_has_jv_qe_responsivity_detectivity_headers(self):
        ws = self.wb["Optimized"]
        all_headers = {ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)}
        all_headers = {str(h).strip().lower() for h in all_headers if h}
        self.assertTrue(any("jtot" in h for h in all_headers), "Optimized: jtot missing")
        self.assertTrue(any("qe" in h for h in all_headers), "Optimized: QE missing")
        self.assertTrue(any("r" in h and ("a/w" in h or "r[" in h or "r(a" in h)
                           for h in all_headers),
                        f"Optimized: Responsivity column missing. Headers: {all_headers}")
        self.assertTrue(any("detect" in h or "d*" in h for h in all_headers),
                        f"Optimized: Detectivity column missing. Headers: {all_headers}")

    # -- Sheet1 --

    def test_sheet1_dimensions(self):
        ws = self.wb["Sheet1"]
        self.assertGreater(ws.max_row, 400, "Sheet1: expected >400 rows")
        self.assertEqual(ws.max_column, 23, f"Sheet1: expected 23 cols, got {ws.max_column}")

    def test_sheet1_has_position_axis(self):
        ws = self.wb["Sheet1"]
        h_col2 = str(ws.cell(row=1, column=2).value or "").strip().lower()
        self.assertIn("x(um)", h_col2,
            f"Sheet1: expected position axis 'x(um)' in col 1, got '{h_col2}'")

    def test_sheet1_high_numeric_density(self):
        ws = self.wb["Sheet1"]
        nd = numeric_density(ws, r_start=3, n_rows=50, n_cols=23)
        self.assertGreater(nd, 0.90,
            f"Sheet1: expected numeric density >0.90, got {nd:.2f}")

    def test_sheet1_has_no_empty_column_boundaries(self):
        ws = self.wb["Sheet1"]
        empty = empty_cols_in_first_n_rows(ws)
        self.assertEqual(empty, [],
            f"Sheet1: expected no empty column separators, got {empty}")


# ---------------------------------------------------------------------------
# Phase 1b: Fixture self-consistency checks
# ---------------------------------------------------------------------------

class TestFixtureSelfConsistency(unittest.TestCase):
    """Verify that the fixture data structures are internally consistent."""

    def test_all_structural_fixtures_have_required_keys(self):
        required = {"sheet", "description"}
        for f in STRUCTURAL_FIXTURES:
            missing = required - set(f.keys())
            self.assertEqual(missing, set(), f"Structural fixture missing keys: {missing}\n{f}")

    def test_all_experiment_fixtures_have_required_keys(self):
        required = {"sheet", "experiment_index", "expected_structure_type",
                    "expected_interpretation_state"}
        for f in EXPERIMENT_FIXTURES:
            missing = required - set(f.keys())
            self.assertEqual(missing, set(), f"Experiment fixture missing keys: {missing}\n{f}")

    def test_parameter_parsing_fixtures_have_value(self):
        for f in PARAMETER_PARSING_FIXTURES:
            self.assertIn("expected_parameter_value", f,
                f"Parameter parsing fixture missing expected_parameter_value: {f}")

    def test_parameter_parsing_power_notation(self):
        """'10^13' must be documented to parse to 1e13."""
        pow_fixtures = [f for f in PARAMETER_PARSING_FIXTURES if "10^" in f.get("header_text", "")]
        self.assertGreater(len(pow_fixtures), 0, "No '10^N' parameter parsing fixtures found")
        for f in pow_fixtures:
            val = f["expected_parameter_value"]
            self.assertIsInstance(val, float,
                f"'10^N' fixture expected_parameter_value must be a float, got {type(val)}")
            self.assertGreater(val, 0,
                f"'10^N' fixture must have positive float value, got {val}")

    def test_rs_unit_embedded_fixture_present(self):
        """The Rs header-with-unit edge case must have a dedicated fixture."""
        rs_fixture = next(
            (f for f in PARAMETER_PARSING_FIXTURES if "ohm/cm^2" in f.get("header_text", "")),
            None
        )
        self.assertIsNotNone(rs_fixture,
            "Missing parameter parsing fixture for '1 ohm/cm^2' header format")

    def test_interface_density_has_two_jv_experiment_fixtures(self):
        """Must have at least 2 JV fixtures for Interface density sheet."""
        jv_fixtures = [
            f for f in EXPERIMENT_FIXTURES
            if f["sheet"] == "Interface density" and f.get("expected_scientific_type") == "JV"
        ]
        self.assertGreaterEqual(len(jv_fixtures), 2,
            f"Expected >= 2 JV experiment fixtures for Interface density, got {len(jv_fixtures)}")

    def test_invariants_have_assertion_field(self):
        for inv in INVARIANT_FIXTURES:
            self.assertIn("assertion", inv,
                f"Invariant fixture missing 'assertion' field: {inv}")

    def test_sheet1_fixture_expects_unresolved(self):
        sheet1_fixture = next(
            (f for f in EXPERIMENT_FIXTURES if f["sheet"] == "Sheet1"),
            None
        )
        self.assertIsNotNone(sheet1_fixture, "No experiment fixture for Sheet1")
        self.assertEqual(sheet1_fixture["expected_interpretation_state"], "UNRESOLVED",
            "Sheet1 must be expected as UNRESOLVED, not any scientific type")
        self.assertIsNone(sheet1_fixture.get("expected_scientific_type"),
            "Sheet1 expected_scientific_type must be None")


if __name__ == "__main__":
    unittest.main(verbosity=2)
