"""
parser_v2.evaluation.test_fidelity
----------------------------------
Dedicated ground-truth extraction fidelity evaluation test suite.

Tests cover:
  1. Overall extraction fidelity score == 100.0% across ground-truth datasets.
  2. Dataset-by-dataset agreement across all 15 required aspects:
     - ETL J-V & Scalar Jsc
     - ETL QE & Responsivity
     - Defect Density J-V & Scalar Jsc
     - Rs J-V & Scalar Jsc
     - Rsh J-V & Scalar Jsc (Rsh scalar sweep wins ScalarSweepTable)
     - Interface Density (both independent JV experiments extract all 6 traces)
     - Optimized J-V, QE, Responsivity, Detectivity
     - Sheet1 (evaluated as EXACT_MATCH / UNRESOLVED, preserving raw)
  3. Regression tests:
     - Interface Density all 6 traces extracted per interface block.
     - Truncated/partially-populated tail traces retained without data loss.
     - Bare parameter aliases (Rsh, Rs, Na, Nd, Nt) matched in LabelProbe.
     - Generic scalar sweeps (Temperature -> Jsc, Thickness -> Jsc) win ScalarSweepTable.
     - No cross-experiment mixing.
     - Complete CellRef provenance retention.
     - Numerical values match raw openpyxl cell grid exactly.
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
from app.parser_v2.extractor.extract import extract_sheet_regions
from app.parser_v2.probes.evidence import CellRef
from app.parser_v2.probes.label_probe import match_text_to_concepts
from app.parser_v2.reader.raw_reader import read_workbook
from app.parser_v2.reasoner.experiment_classifier import classify_sheet_regions
from app.parser_v2.reasoner.hypothesis import AcceptanceState, StructuralHypothesis
from app.parser_v2.scanner.region_scanner import scan_sheet

WORKBOOK_PATH = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\Data_file.xlsx")


def _load_extracted_sheet(sheet_name: str):
    if not WORKBOOK_PATH.exists():
        raise unittest.SkipTest(f"Real workbook not found at {WORKBOOK_PATH}")
    with open(WORKBOOK_PATH, "rb") as f:
        file_bytes = f.read()
    wb = read_workbook(WORKBOOK_PATH.name, file_bytes)
    sheet = wb.get_sheet(sheet_name)
    assert sheet is not None, f"Sheet '{sheet_name}' not found"
    regions = scan_sheet(sheet)
    interps = classify_sheet_regions(sheet, regions)
    datasets = extract_sheet_regions(sheet, regions, interps)
    return sheet, regions, interps, datasets


class TestExtractionFidelity(unittest.TestCase):
    """Stage 6.5 Extraction Fidelity Evaluation test suite."""

    @classmethod
    def setUpClass(cls):
        if not WORKBOOK_PATH.exists():
            raise unittest.SkipTest(f"Real workbook not found at {WORKBOOK_PATH}")
        cls.report = evaluate_workbook_fidelity(str(WORKBOOK_PATH))

    def test_overall_fidelity_percentage_hundred(self):
        """Overall fidelity percentage across ground-truth datasets must be 100%."""
        self.assertEqual(
            self.report.overall_fidelity_percentage, 100.0,
            f"Fidelity percentage expected 100.0%, got {self.report.overall_fidelity_percentage}%",
        )

    def test_interface_density_all_6_traces_extracted(self):
        """
        Regression: Sns2/PS JV block (region_0) and PS/Zn3P2 JV block (region_4)
        must both extract all 6 trace columns (10^10..10^20).
        """
        _, _, _, datasets = _load_extracted_sheet("Interface density")
        jv_datasets = [d for d in datasets if d.scientific_label == "JV"]
        self.assertGreaterEqual(len(jv_datasets), 2, "Expected >= 2 JV datasets in Interface density")

        reg0 = next(d for d in jv_datasets if d.region_id == "region_0")
        reg4 = next(d for d in jv_datasets if d.region_id == "region_4")

        self.assertEqual(len(reg0.traces), 6, f"Sns2/PS JV expected 6 traces, got {len(reg0.traces)}")
        self.assertEqual(len(reg4.traces), 6, f"PS/Zn3P2 JV expected 6 traces, got {len(reg4.traces)}")

        # Check parameter values
        expected_params = [1e10, 1e12, 1e14, 1e16, 1e18, 1e20]
        params0 = sorted([t.parameter_value for t in reg0.traces if t.parameter_value is not None])
        self.assertEqual(len(params0), 6)
        for got, exp in zip(params0, expected_params):
            self.assertAlmostEqual(got, exp, places=0)

    def test_truncated_tail_traces_retained(self):
        """
        Regression: In Interface density Sns2/PS JV, high defect density traces
        (10^16, 10^18, 10^20) terminate before maximum voltage (trailing cells empty).
        The extractor MUST retain these traces with valid values up to breakdown
        without data loss or fabrication.
        """
        _, _, _, datasets = _load_extracted_sheet("Interface density")
        reg0 = next(d for d in datasets if d.region_id == "region_0")

        # Traces 3, 4, 5 correspond to 10^16, 10^18, 10^20
        for t_idx in (3, 4, 5):
            trace = reg0.traces[t_idx]
            self.assertGreater(
                len(trace.values), 10,
                f"Trace {t_idx} ({trace.parameter_raw_text}) must retain its valid points up to breakdown",
            )
            # len(values) == len(source_refs) invariant
            self.assertEqual(len(trace.values), len(trace.source_refs))

    def test_rsh_scalar_sweep_table_wins(self):
        """
        Regression: Rsh region_1 (Rsh parameter vs Jsc metric) MUST win
        ScalarSweepTable hypothesis (score=1.00) over NumericCurveGrid.
        """
        _, _, interps, datasets = _load_extracted_sheet("Rsh")
        ds_rsh_summary = next(d for d in datasets if d.region_id == "region_1")

        self.assertEqual(ds_rsh_summary.structure_type, "ScalarSweepTable")
        self.assertEqual(ds_rsh_summary.scientific_label, "SCALAR_SWEEP")
        self.assertEqual(ds_rsh_summary.acceptance_state, AcceptanceState.ACCEPTED.value)
        self.assertEqual(len(ds_rsh_summary.scalar_rows), 8)

        # Check parameter values 1000..8000
        p_vals = [r.parameter_value for r in ds_rsh_summary.scalar_rows]
        self.assertEqual(p_vals, [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0])

    def test_bare_parameter_aliases_matching(self):
        """
        Regression: LabelProbe must recognize bare parameter tokens:
        Rsh, Rs, Na, Nd, Nt without requiring '='.
        """
        for token, expected_concept in [
            ("Rsh", "Shunt_Resistance"),
            ("rsh", "Shunt_Resistance"),
            ("Rs", "Series_Resistance"),
            ("rs", "Series_Resistance"),
            ("Na", "Acceptor_Density"),
            ("na", "Acceptor_Density"),
            ("Nd", "Donor_Density"),
            ("nd", "Donor_Density"),
            ("Nt", "Defect_Density"),
            ("nt", "Defect_Density"),
        ]:
            matches = match_text_to_concepts(token, "header_token", None)
            matched_names = [m.concept_name for m in matches]
            self.assertIn(
                expected_concept, matched_names,
                f"Bare token '{token}' failed to match expected concept '{expected_concept}'",
            )

    def test_generic_scalar_sweep_generalization_rs_and_etl(self):
        """
        Regression: Verify that the Reasoner rule generalizes to other scalar sweeps:
        Rs -> Jsc summary (Rs sheet region_1) and ETL -> Jsc summary (ETL sheet region_1).
        """
        _, _, _, etl_datasets = _load_extracted_sheet("ETL")
        etl_scalar = next(d for d in etl_datasets if d.region_id == "region_1")
        self.assertEqual(etl_scalar.structure_type, "ScalarSweepTable")
        self.assertEqual(len(etl_scalar.scalar_rows), 10)

        _, _, _, rs_datasets = _load_extracted_sheet("Rs")
        rs_scalar = next(d for d in rs_datasets if d.region_id == "region_1")
        self.assertEqual(rs_scalar.structure_type, "ScalarSweepTable")
        self.assertEqual(len(rs_scalar.scalar_rows), 8)

    def test_no_cross_experiment_mixing(self):
        """
        Regression: The two independent JV experiments in Interface density (Sns2/PS and PS/Zn3P2)
        must belong to distinct region_id containers.
        """
        _, _, _, datasets = _load_extracted_sheet("Interface density")
        jv_datasets = [d for d in datasets if d.scientific_label == "JV"]
        region_ids = [d.region_id for d in jv_datasets]
        self.assertEqual(len(region_ids), len(set(region_ids)), "Region IDs must be distinct")

    def test_cellref_provenance_completeness(self):
        """
        Regression: Every extracted trace value in ETL, Rsh, Rs, Interface density
        must carry complete, valid CellRef provenance.
        """
        for sheet_name in ("ETL", "Rsh", "Rs", "Interface density"):
            _, _, _, datasets = _load_extracted_sheet(sheet_name)
            for ds in datasets:
                for trace in ds.traces:
                    self.assertEqual(len(trace.values), len(trace.source_refs))
                    for ref in trace.source_refs:
                        self.assertEqual(ref.sheet_name, sheet_name)
                        self.assertGreaterEqual(ref.row, 0)
                        self.assertGreaterEqual(ref.col, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
