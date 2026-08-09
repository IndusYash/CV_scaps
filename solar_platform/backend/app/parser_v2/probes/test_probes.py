"""
Unit and integration tests for parser_v2 Stage 3 Evidence Probes.

Tests cover:
  1. Unit tests for every probe independently (axis, header, numeric, parameter, repetition, label).
  2. Integration tests on real Data_file.xlsx sheets:
     - ETL J-V region
     - ETL QE region
     - Defect Density J-V region
     - Rs J-V region
     - Optimized J-V region
     - Sheet1 region
  3. CellRef provenance verification.
  4. Explicit check that NO probe directly emits a final ScientificExperiment classification.

Run from: solar_platform/backend/
Command:  python -m unittest app.parser_v2.probes.test_probes -v
"""
from __future__ import annotations

import io
import inspect
import unittest
from pathlib import Path

import openpyxl

from app.parser_v2.reader.raw_reader import read_workbook
from app.parser_v2.scanner.region_scanner import scan_sheet
from app.parser_v2.probes import (
    axis_probe,
    header_probe,
    numeric_probe,
    parameter_probe,
    repetition_probe,
    label_probe,
)
from app.parser_v2.probes.evidence import EvidenceItem, EvidencePolarity

WORKBOOK_PATH = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\Data_file.xlsx")


def _create_synthetic_param_excel_bytes() -> bytes:
    """
    Creates a synthetic workbook with parameter patterns:
    col A: Monotonic voltage 0..1
    col B: t=100[nm] data
    col C: t=200[nm] data
    col D: 1 ohm/cm^2 data
    col E: 10^13 data
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ParamSheet"

    ws["A1"] = "v(V)"
    ws["B1"] = "t=100[nm]"
    ws["C1"] = "t=200[nm]"
    ws["D1"] = "1 ohm/cm^2"
    ws["E1"] = "10^13"

    for r in range(2, 12):
        ws[f"A{r}"] = float(r - 2) * 0.1
        ws[f"B{r}"] = float(r * 2)
        ws[f"C{r}"] = float(r * 3)
        ws[f"D{r}"] = float(r * 4)
        ws[f"E{r}"] = float(r * 5)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestAxisProbeSynthetic(unittest.TestCase):

    def test_axis_probe_monotonic_grid(self):
        excel_bytes = _create_synthetic_param_excel_bytes()
        raw_wb = read_workbook("param.xlsx", excel_bytes)
        sheet = raw_wb.get_sheet("ParamSheet")
        regions = scan_sheet(sheet)
        self.assertEqual(len(regions), 1)

        items = axis_probe.probe_axis(regions[0], sheet)
        self.assertGreater(len(items), 0)

        # Look for axis_monotonicity item
        mono_item = next((it for it in items if it.name == "axis_monotonicity"), None)
        self.assertIsNotNone(mono_item)
        self.assertEqual(mono_item.polarity, EvidencePolarity.POSITIVE)
        self.assertEqual(mono_item.value["direction"], "increasing")
        self.assertGreater(len(mono_item.source_cells), 0)


class TestHeaderProbeSynthetic(unittest.TestCase):

    def test_header_probe_token_extraction(self):
        excel_bytes = _create_synthetic_param_excel_bytes()
        raw_wb = read_workbook("param.xlsx", excel_bytes)
        sheet = raw_wb.get_sheet("ParamSheet")
        regions = scan_sheet(sheet)

        items = header_probe.probe_headers(regions[0], sheet)
        self.assertGreater(len(items), 0)

        hdr_item = next((it for it in items if it.name == "header_row_presence"), None)
        self.assertIsNotNone(hdr_item)
        self.assertEqual(hdr_item.polarity, EvidencePolarity.POSITIVE)
        tokens = hdr_item.value["tokens"]
        self.assertEqual(len(tokens), 5)
        self.assertEqual(tokens[0]["raw_text"], "v(V)")


class TestNumericProbeSynthetic(unittest.TestCase):

    def test_numeric_probe_densities_and_ranges(self):
        excel_bytes = _create_synthetic_param_excel_bytes()
        raw_wb = read_workbook("param.xlsx", excel_bytes)
        sheet = raw_wb.get_sheet("ParamSheet")
        regions = scan_sheet(sheet)

        items = numeric_probe.probe_numerics(regions[0], sheet)
        range_item = next((it for it in items if it.name == "numeric_value_range"), None)
        self.assertIsNotNone(range_item)
        self.assertEqual(range_item.polarity, EvidencePolarity.POSITIVE)
        self.assertGreaterEqual(range_item.value["max"], 50.0)


class TestParameterProbeSynthetic(unittest.TestCase):

    def test_parameter_probe_patterns(self):
        excel_bytes = _create_synthetic_param_excel_bytes()
        raw_wb = read_workbook("param.xlsx", excel_bytes)
        sheet = raw_wb.get_sheet("ParamSheet")
        regions = scan_sheet(sheet)

        items = parameter_probe.probe_parameters(regions[0], sheet)
        param_item = next((it for it in items if it.name == "candidate_parameter_expressions"), None)
        self.assertIsNotNone(param_item)
        self.assertEqual(param_item.polarity, EvidencePolarity.POSITIVE)

        params = param_item.value["parameters"]
        patterns = param_item.value["patterns"]

        # Assert patterns found
        self.assertIn("thickness_t=", patterns)
        self.assertIn("inline_unit", patterns)
        self.assertIn("power_exponent_10^N", patterns)

        # Check parsed values
        t_param = next(p for p in params if p["pattern"] == "thickness_t=")
        self.assertEqual(t_param["parsed_value"], 100.0)
        self.assertEqual(t_param["unit_hint"], "nm")

        pwr_param = next(p for p in params if p["pattern"] == "power_exponent_10^N")
        self.assertEqual(pwr_param["parsed_value"], 1e13)


class TestRepetitionProbeSynthetic(unittest.TestCase):

    def test_repetition_probe_groups(self):
        excel_bytes = _create_synthetic_param_excel_bytes()
        raw_wb = read_workbook("param.xlsx", excel_bytes)
        sheet = raw_wb.get_sheet("ParamSheet")
        regions = scan_sheet(sheet)

        items = repetition_probe.probe_repetition(regions[0], sheet)
        rep_item = next((it for it in items if it.name == "repeated_trace_group"), None)
        self.assertIsNotNone(rep_item)
        self.assertEqual(rep_item.polarity, EvidencePolarity.POSITIVE)
        self.assertEqual(rep_item.value["trace_count"], 4)  # 4 y-columns


class TestLabelProbeSynthetic(unittest.TestCase):

    def test_label_probe_concept_candidates(self):
        excel_bytes = _create_synthetic_param_excel_bytes()
        raw_wb = read_workbook("param.xlsx", excel_bytes)
        sheet = raw_wb.get_sheet("ParamSheet")
        regions = scan_sheet(sheet)

        items = label_probe.probe_labels(regions[0], sheet)
        lbl_item = next((it for it in items if it.name == "candidate_label_concepts"), None)
        self.assertIsNotNone(lbl_item)
        self.assertEqual(lbl_item.polarity, EvidencePolarity.POSITIVE)
        concepts = lbl_item.value["distinct_concepts"]
        self.assertIn("Voltage", concepts)


class TestProbesRealWorkbook(unittest.TestCase):
    """
    Integration tests against representative real sheets from Data_file.xlsx.
    """

    @classmethod
    def setUpClass(cls):
        if not WORKBOOK_PATH.exists():
            raise unittest.SkipTest(f"Real workbook not found at {WORKBOOK_PATH}")
        with open(WORKBOOK_PATH, "rb") as f:
            cls.file_bytes = f.read()
        cls.raw_wb = read_workbook("Data_file.xlsx", cls.file_bytes)

    # 1. ETL J-V region (region 0)
    def test_etl_jv_region_evidence(self):
        sheet = self.raw_wb.get_sheet("ETL")
        regions = scan_sheet(sheet)
        reg_jv = regions[0]

        items_axis = axis_probe.probe_axis(reg_jv, sheet)
        items_hdr = header_probe.probe_headers(reg_jv, sheet)
        items_num = numeric_probe.probe_numerics(reg_jv, sheet)
        items_param = parameter_probe.probe_parameters(reg_jv, sheet)
        items_rep = repetition_probe.probe_repetition(reg_jv, sheet)
        items_lbl = label_probe.probe_labels(reg_jv, sheet)

        self.assertTrue(any(it.polarity == EvidencePolarity.POSITIVE for it in items_axis))
        self.assertTrue(any(it.polarity == EvidencePolarity.POSITIVE for it in items_param))

        lbl_item = next((it for it in items_lbl if it.name == "candidate_label_concepts"), None)
        self.assertIsNotNone(lbl_item)
        concepts = lbl_item.value["distinct_concepts"]
        self.assertIn("Voltage", concepts)
        self.assertTrue(any(c in concepts for c in ["Current", "Jsc"]))

    # 2. ETL QE region (region 2)
    def test_etl_qe_region_evidence(self):
        sheet = self.raw_wb.get_sheet("ETL")
        regions = scan_sheet(sheet)
        reg_qe = regions[2]

        items_lbl = label_probe.probe_labels(reg_qe, sheet)
        lbl_item = next((it for it in items_lbl if it.name == "candidate_label_concepts"), None)
        self.assertIsNotNone(lbl_item)
        concepts = lbl_item.value["distinct_concepts"]
        self.assertIn("Wavelength", concepts)
        self.assertIn("QE", concepts)

    # 3. Defect Density J-V region (region 0)
    def test_defect_density_jv_region_evidence(self):
        sheet = self.raw_wb.get_sheet("Defect Density")
        regions = scan_sheet(sheet)
        reg_dd = regions[0]

        items_param = parameter_probe.probe_parameters(reg_dd, sheet)
        param_item = next((it for it in items_param if it.name == "candidate_parameter_expressions"), None)
        self.assertIsNotNone(param_item)
        self.assertIn("power_exponent_10^N", param_item.value["patterns"])

    # 4. Rs J-V region (region 0)
    def test_rs_jv_region_evidence(self):
        sheet = self.raw_wb.get_sheet("Rs")
        regions = scan_sheet(sheet)
        reg_rs = regions[0]

        items_param = parameter_probe.probe_parameters(reg_rs, sheet)
        param_item = next((it for it in items_param if it.name == "candidate_parameter_expressions"), None)
        self.assertIsNotNone(param_item)
        self.assertIn("inline_unit", param_item.value["patterns"])

    # 5. Optimized J-V region (region 0)
    def test_optimized_jv_region_evidence(self):
        sheet = self.raw_wb.get_sheet("Optimized")
        regions = scan_sheet(sheet)
        reg_opt_jv = regions[0]

        items_lbl = label_probe.probe_labels(reg_opt_jv, sheet)
        lbl_item = next((it for it in items_lbl if it.name == "candidate_label_concepts"), None)
        self.assertIsNotNone(lbl_item)
        concepts = lbl_item.value["distinct_concepts"]
        self.assertTrue(any(c in concepts for c in ["Current", "Detectivity", "Jsc"]))

    # 6. Sheet1 region (region 0)
    def test_sheet1_region_evidence(self):
        sheet = self.raw_wb.get_sheet("Sheet1")
        regions = scan_sheet(sheet)
        main_reg = max(regions, key=lambda r: r.total_cells)

        items_lbl = label_probe.probe_labels(main_reg, sheet)
        lbl_item = next((it for it in items_lbl if it.name == "candidate_label_concepts"), None)
        self.assertIsNotNone(lbl_item)
        concepts = lbl_item.value["distinct_concepts"]
        self.assertIn("Position", concepts)


class TestProbeArchitectureConstraints(unittest.TestCase):
    """
    Verifies that probes NEVER return a ScientificExperiment object or classify an experiment.
    """

    def test_probes_do_not_return_experiment_class(self):
        all_probes = [axis_probe, header_probe, numeric_probe, parameter_probe, repetition_probe, label_probe]
        for p in all_probes:
            # Check function signatures
            funcs = [f for f_name, f in inspect.getmembers(p, inspect.isfunction) if f_name.startswith("probe_")]
            for f in funcs:
                sig = inspect.signature(f)
                return_annotation = str(sig.return_annotation)
                self.assertNotIn("ScientificExperiment", return_annotation,
                    f"Probe function {f.__name__} in {p.__name__} must not return ScientificExperiment")


if __name__ == "__main__":
    unittest.main(verbosity=2)
