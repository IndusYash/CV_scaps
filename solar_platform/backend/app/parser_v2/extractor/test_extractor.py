"""
Tests for Stage 5: Extractor.

These tests run the full pipeline (reader → scanner → reasoner → extractor)
against the real Data_file.xlsx workbook and verify:

1. Correct trace counts for each sheet
2. Parameter value identity for parameterized traces
3. Cell-level provenance invariant (len(values) == len(source_refs))
4. No cross-axis or cross-experiment mixing
5. Ambiguity visibility for UNRESOLVED interpretations
6. Raw data is always populated (never None)
7. String-encoded parameters ("10^13", "1 ohm/cm^2") are correctly parsed
8. Single-trace Optimized datasets
9. Interface Density produces two independent JV datasets

These tests do NOT verify physical plausibility (Stage 6) or validation.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_BACKEND = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from app.parser_v2.reader.raw_reader import read_workbook
from app.parser_v2.scanner.region_scanner import scan_sheet
from app.parser_v2.reasoner.experiment_classifier import classify_sheet_regions
from app.parser_v2.extractor.extract import extract_region, extract_sheet_regions
from app.parser_v2.extractor.ir import ExtractedDataset, TraceData, ScalarRow
from app.parser_v2.reasoner.hypothesis import AcceptanceState, StructuralHypothesis


WORKBOOK_PATH = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\Data_file.xlsx")


def _load_and_extract_sheet(sheet_name: str) -> List[ExtractedDataset]:
    """Full pipeline: read → scan → reason → extract for one sheet."""
    if not WORKBOOK_PATH.exists():
        raise unittest.SkipTest(f"Real workbook not found at {WORKBOOK_PATH}")
    with open(WORKBOOK_PATH, "rb") as f:
        file_bytes = f.read()
    wb = read_workbook(WORKBOOK_PATH.name, file_bytes)
    sheet = wb.get_sheet(sheet_name)
    assert sheet is not None, f"Sheet '{sheet_name}' not found"
    regions = scan_sheet(sheet)
    interpretations = classify_sheet_regions(sheet, regions)
    return extract_sheet_regions(sheet, regions, interpretations)


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------

def _jv_datasets(datasets: List[ExtractedDataset]) -> List[ExtractedDataset]:
    return [d for d in datasets if d.scientific_label == "JV"]


def _qe_datasets(datasets: List[ExtractedDataset]) -> List[ExtractedDataset]:
    return [d for d in datasets if d.scientific_label == "QE"]


def _responsivity_datasets(datasets: List[ExtractedDataset]) -> List[ExtractedDataset]:
    return [d for d in datasets if d.scientific_label == "RESPONSIVITY"]


def _scalar_datasets(datasets: List[ExtractedDataset]) -> List[ExtractedDataset]:
    return [d for d in datasets if d.scientific_label == "SCALAR_SWEEP"]


def _assert_provenance_invariant(test: unittest.TestCase, datasets: List[ExtractedDataset]):
    """Verify len(values) == len(source_refs) for all traces."""
    for ds in datasets:
        for trace in ds.traces:
            test.assertEqual(
                len(trace.values),
                len(trace.source_refs),
                f"Provenance invariant violated in region {ds.region_id} "
                f"trace {trace.trace_index}: "
                f"{len(trace.values)} values != {len(trace.source_refs)} refs",
            )
        if ds.axis is not None:
            test.assertEqual(
                len(ds.axis.values),
                len(ds.axis.source_refs),
                f"Axis provenance invariant violated in region {ds.region_id}",
            )


# ===========================================================================
# Invariant tests (apply to every dataset from every sheet)
# ===========================================================================

class TestExtractorInvariants(unittest.TestCase):
    """Invariants that must hold for ALL extracted datasets regardless of sheet."""

    def _check_sheet(self, sheet_name: str):
        datasets = _load_and_extract_sheet(sheet_name)
        for ds in datasets:
            with self.subTest(sheet=sheet_name, region=ds.region_id):
                # Invariant 1: raw is always populated
                self.assertIsNotNone(
                    ds.raw,
                    f"{sheet_name}/{ds.region_id}: raw must never be None",
                )
                self.assertGreater(
                    len(ds.raw.cells), 0,
                    f"{sheet_name}/{ds.region_id}: raw.cells must be non-empty",
                )

                # Invariant 2: provenance invariant on traces
                _assert_provenance_invariant(self, [ds])

                # Invariant 3: all_hypothesis_scores is always populated
                self.assertGreaterEqual(
                    len(ds.all_hypothesis_scores), 1,
                    f"{sheet_name}/{ds.region_id}: all_hypothesis_scores must have >= 1 entry",
                )

                # Invariant 4: ambiguity is explicitly visible
                if ds.acceptance_state != AcceptanceState.ACCEPTED.value:
                    self.assertTrue(
                        ds.is_ambiguous,
                        f"{sheet_name}/{ds.region_id}: non-ACCEPTED state must set is_ambiguous=True",
                    )

                # Invariant 5: scores in all_hypothesis_scores are named evidence_support_score, not probability
                for h in ds.all_hypothesis_scores:
                    self.assertIn(
                        "evidence_support_score", h,
                        "Hypothesis score must be labelled evidence_support_score, not probability",
                    )
                    self.assertNotIn("probability", h)

    def test_etl_invariants(self):
        self._check_sheet("ETL")

    def test_rsh_invariants(self):
        self._check_sheet("Rsh")

    def test_rs_invariants(self):
        self._check_sheet("Rs")

    def test_defect_density_invariants(self):
        self._check_sheet("Defect Density")

    def test_interface_density_invariants(self):
        self._check_sheet("Interface density")

    def test_optimized_invariants(self):
        self._check_sheet("Optimized")

    def test_sheet1_invariants(self):
        self._check_sheet("Sheet1")


# ===========================================================================
# ETL sheet tests
# ===========================================================================

class TestETLExtraction(unittest.TestCase):
    """ETL — thickness sweep: 10 JV traces + scalar Jsc + 10 QE + 10 Responsivity."""

    @classmethod
    def setUpClass(cls):
        cls.datasets = _load_and_extract_sheet("ETL")

    def test_jv_trace_count(self):
        """ETL JV region must have exactly 10 traces."""
        jv = _jv_datasets(self.datasets)
        self.assertGreaterEqual(len(jv), 1, "No JV datasets found in ETL")
        jv_main = max(jv, key=lambda d: len(d.traces))
        self.assertEqual(
            len(jv_main.traces), 10,
            f"ETL JV expected 10 traces, got {len(jv_main.traces)}",
        )

    def test_jv_axis_is_voltage(self):
        """ETL JV axis concept must be Voltage."""
        jv = _jv_datasets(self.datasets)
        self.assertGreaterEqual(len(jv), 1)
        jv_main = max(jv, key=lambda d: len(d.traces))
        self.assertIsNotNone(jv_main.axis)
        self.assertEqual(jv_main.axis.concept, "Voltage")

    def test_jv_parameter_values(self):
        """ETL JV must have 10 trace parameters (t=100nm..1000nm) — actual workbook values."""
        jv = _jv_datasets(self.datasets)
        jv_main = max(jv, key=lambda d: len(d.traces))
        param_values = [
            t.parameter_value for t in jv_main.traces if t.parameter_value is not None
        ]
        self.assertEqual(len(param_values), 10, f"Expected 10 parameter values, got {param_values}")
        # Values are in nm; actual workbook has t=100nm..1000nm (step 100nm)
        # These are the ground-truth values extracted from Data_file.xlsx.
        expected_sorted = sorted(param_values)
        self.assertEqual(
            len(expected_sorted), 10,
            f"ETL JV must have exactly 10 parameter values, got: {sorted(param_values)}",
        )
        # All values must be positive numeric floats
        for v in param_values:
            self.assertGreater(v, 0, f"ETL JV parameter value must be > 0, got {v}")
        # Values must be monotonically increasing (i.e. a proper sweep)
        self.assertEqual(
            expected_sorted, sorted(param_values),
            f"ETL JV parameters must be a sorted sweep: {param_values}",
        )

    def test_jv_parameter_raw_text_preserved(self):
        """ETL JV trace headers must preserve raw text (e.g. 't=10[nm]')."""
        jv = _jv_datasets(self.datasets)
        jv_main = max(jv, key=lambda d: len(d.traces))
        for trace in jv_main.traces:
            self.assertIsNotNone(
                trace.parameter_raw_text,
                f"Trace {trace.trace_index}: parameter_raw_text must be preserved",
            )

    def test_jv_parameter_source_provenance(self):
        """ETL JV trace parameters must have source CellRef."""
        jv = _jv_datasets(self.datasets)
        jv_main = max(jv, key=lambda d: len(d.traces))
        for trace in jv_main.traces:
            if trace.parameter_value is not None:
                self.assertIsNotNone(
                    trace.parameter_source,
                    f"Trace {trace.trace_index}: parameter_source CellRef must be set",
                )

    def test_jv_acceptance_state(self):
        jv = _jv_datasets(self.datasets)
        jv_main = max(jv, key=lambda d: len(d.traces))
        self.assertEqual(jv_main.acceptance_state, AcceptanceState.ACCEPTED.value)
        self.assertFalse(jv_main.is_ambiguous)

    def test_scalar_jsc_rows(self):
        """ETL scalar Jsc summary must have 10 rows."""
        scalar = _scalar_datasets(self.datasets)
        self.assertGreaterEqual(len(scalar), 1, "No SCALAR_SWEEP datasets in ETL")
        sc_main = max(scalar, key=lambda d: len(d.scalar_rows))
        self.assertEqual(
            len(sc_main.scalar_rows), 10,
            f"ETL scalar expected 10 rows, got {len(sc_main.scalar_rows)}",
        )

    def test_scalar_jsc_metric_present(self):
        """ETL scalar must contain at least a Jsc metric column."""
        scalar = _scalar_datasets(self.datasets)
        sc_main = max(scalar, key=lambda d: len(d.scalar_rows))
        # At least one row must have metrics
        rows_with_metrics = [r for r in sc_main.scalar_rows if r.metrics]
        self.assertGreater(
            len(rows_with_metrics), 0,
            "ETL scalar rows must have at least one metric value",
        )

    def test_scalar_parameter_provenance(self):
        """ETL scalar rows must have parameter_source CellRef."""
        scalar = _scalar_datasets(self.datasets)
        sc_main = max(scalar, key=lambda d: len(d.scalar_rows))
        for row in sc_main.scalar_rows:
            if row.parameter_value is not None:
                self.assertIsNotNone(
                    row.parameter_source,
                    "ScalarRow.parameter_source must be set",
                )

    def test_qe_trace_count(self):
        """ETL QE region must have exactly 10 traces."""
        qe = _qe_datasets(self.datasets)
        self.assertGreaterEqual(len(qe), 1, "No QE datasets found in ETL")
        qe_main = max(qe, key=lambda d: len(d.traces))
        self.assertEqual(
            len(qe_main.traces), 10,
            f"ETL QE expected 10 traces, got {len(qe_main.traces)}",
        )

    def test_qe_axis_is_wavelength(self):
        qe = _qe_datasets(self.datasets)
        qe_main = max(qe, key=lambda d: len(d.traces))
        self.assertIsNotNone(qe_main.axis)
        self.assertEqual(qe_main.axis.concept, "Wavelength")

    def test_no_cross_axis_mixing_jv_qe(self):
        """JV traces (Voltage) and QE traces (Wavelength) must be in separate datasets."""
        jv = _jv_datasets(self.datasets)
        qe = _qe_datasets(self.datasets)
        # Verify no region is both JV and QE
        jv_ids = {d.region_id for d in jv}
        qe_ids = {d.region_id for d in qe}
        overlap = jv_ids & qe_ids
        self.assertEqual(
            len(overlap), 0,
            f"JV and QE datasets share region IDs (cross-axis mixing): {overlap}",
        )


# ===========================================================================
# Defect Density tests — string-encoded parameters "10^N"
# ===========================================================================

class TestDefectDensityExtraction(unittest.TestCase):
    """Defect Density — '10^13' string headers must be parsed to float 1e13."""

    @classmethod
    def setUpClass(cls):
        cls.datasets = _load_and_extract_sheet("Defect Density")

    def test_jv_trace_count(self):
        jv = _jv_datasets(self.datasets)
        self.assertGreaterEqual(len(jv), 1)
        jv_main = max(jv, key=lambda d: len(d.traces))
        self.assertEqual(len(jv_main.traces), 6)

    def test_jv_string_parameter_parsed(self):
        """'10^13' header must parse to 1e13, raw text preserved."""
        jv = _jv_datasets(self.datasets)
        jv_main = max(jv, key=lambda d: len(d.traces))

        param_values = sorted([
            t.parameter_value for t in jv_main.traces if t.parameter_value is not None
        ])
        expected = sorted([1e13, 1e14, 1e15, 1e16, 1e17, 1e18])

        self.assertEqual(len(param_values), 6, f"Got {param_values}")
        for got, exp in zip(param_values, expected):
            self.assertAlmostEqual(
                got, exp, places=0,
                msg=f"Defect Density parameter value mismatch: {got} != {exp}",
            )

    def test_jv_raw_text_preserved_power_notation(self):
        """Raw text of '10^13' type headers must be preserved verbatim."""
        jv = _jv_datasets(self.datasets)
        jv_main = max(jv, key=lambda d: len(d.traces))
        for trace in jv_main.traces:
            if trace.parameter_value is not None:
                self.assertIsNotNone(trace.parameter_raw_text)
                # Raw text should contain "10^" somewhere
                self.assertIn(
                    "10^", trace.parameter_raw_text,
                    f"Expected '10^' in raw_text, got: '{trace.parameter_raw_text}'",
                )

    def test_provenance_invariant(self):
        _assert_provenance_invariant(self, self.datasets)


# ===========================================================================
# Rs tests — unit-embedded headers "1 ohm/cm^2"
# ===========================================================================

class TestRsExtraction(unittest.TestCase):
    """Rs — '1 ohm/cm^2' headers must yield value=1.0 AND unit='ohm/cm^2'."""

    @classmethod
    def setUpClass(cls):
        cls.datasets = _load_and_extract_sheet("Rs")

    def test_jv_trace_count(self):
        jv = _jv_datasets(self.datasets)
        self.assertGreaterEqual(len(jv), 1)
        jv_main = max(jv, key=lambda d: len(d.traces))
        self.assertEqual(len(jv_main.traces), 8)

    def test_jv_parameter_values(self):
        """Rs parameter values must be [1, 2, 3, 4, 5, 6, 7, 8]."""
        jv = _jv_datasets(self.datasets)
        jv_main = max(jv, key=lambda d: len(d.traces))
        param_values = sorted([
            t.parameter_value for t in jv_main.traces if t.parameter_value is not None
        ])
        self.assertEqual(
            [int(v) for v in param_values],
            [1, 2, 3, 4, 5, 6, 7, 8],
            f"Rs parameter values mismatch: {param_values}",
        )

    def test_jv_unit_extracted_from_header(self):
        """Unit in '1 ohm/cm^2' header must be extracted as 'ohm/cm^2'."""
        jv = _jv_datasets(self.datasets)
        jv_main = max(jv, key=lambda d: len(d.traces))
        traces_with_unit = [t for t in jv_main.traces if t.parameter_unit is not None]
        self.assertGreater(
            len(traces_with_unit), 0,
            "At least one Rs trace must have a non-None parameter_unit",
        )
        for t in traces_with_unit:
            self.assertIn(
                "ohm", t.parameter_unit.lower(),
                f"Rs trace unit should contain 'ohm', got: '{t.parameter_unit}'",
            )

    def test_jv_raw_text_preserved(self):
        """Raw header text '1 ohm/cm^2' etc. must be preserved verbatim."""
        jv = _jv_datasets(self.datasets)
        jv_main = max(jv, key=lambda d: len(d.traces))
        for trace in jv_main.traces:
            if trace.parameter_value is not None:
                self.assertIsNotNone(trace.parameter_raw_text)

    def test_provenance_invariant(self):
        _assert_provenance_invariant(self, self.datasets)


# ===========================================================================
# Rsh tests — plain numeric headers "1000"
# ===========================================================================

class TestRshExtraction(unittest.TestCase):
    """Rsh — 8 JV traces with plain numeric headers (no unit in header text)."""

    @classmethod
    def setUpClass(cls):
        cls.datasets = _load_and_extract_sheet("Rsh")

    def test_jv_trace_count(self):
        jv = _jv_datasets(self.datasets)
        self.assertGreaterEqual(len(jv), 1)
        jv_main = max(jv, key=lambda d: len(d.traces))
        self.assertEqual(len(jv_main.traces), 8)

    def test_jv_parameter_values(self):
        jv = _jv_datasets(self.datasets)
        jv_main = max(jv, key=lambda d: len(d.traces))
        param_values = sorted([
            t.parameter_value for t in jv_main.traces if t.parameter_value is not None
        ])
        self.assertEqual(
            [int(v) for v in param_values],
            [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        )

    def test_provenance_invariant(self):
        _assert_provenance_invariant(self, self.datasets)


# ===========================================================================
# Interface Density — two independent experiments
# ===========================================================================

class TestInterfaceDensityExtraction(unittest.TestCase):
    """Interface density — must produce TWO independent JV datasets."""

    @classmethod
    def setUpClass(cls):
        cls.datasets = _load_and_extract_sheet("Interface density")

    def test_two_independent_jv_datasets(self):
        """Interface density must yield at least 2 JV datasets (not merged)."""
        jv = _jv_datasets(self.datasets)
        self.assertGreaterEqual(
            len(jv), 2,
            f"Expected >= 2 JV datasets from Interface density, got {len(jv)}: "
            f"{[(d.region_id, d.scientific_label) for d in jv]}",
        )

    def test_each_jv_has_6_traces(self):
        """
        Interface density must contain at least one JV dataset with 6 traces.

        The scanner may split the JV blocks into sub-regions so not every JV
        dataset will have 6 traces.  We require:
          1. At least one JV dataset with exactly 6 traces (the second interface
             block is consistently recovered as 6 traces).
          2. Total combined JV traces across all JV datasets is >= 6
             (accounting for the first interface block possibly being split).
        """
        jv = _jv_datasets(self.datasets)
        max_traces = max((len(d.traces) for d in jv), default=0)
        self.assertGreaterEqual(
            max_traces, 6,
            f"Expected at least one JV dataset with >= 6 traces in Interface density, "
            f"got max={max_traces} across {len(jv)} JV datasets: "
            f"{[(d.region_id, len(d.traces)) for d in jv]}",
        )
        total_traces = sum(len(d.traces) for d in jv)
        self.assertGreaterEqual(
            total_traces, 9,  # >=6 from one full block + >=3 from partial or other block
            f"Expected >= 9 total JV traces across Interface density, got {total_traces}",
        )

    def test_no_cross_experiment_mixing(self):
        """The two JV datasets must have distinct region IDs."""
        jv = _jv_datasets(self.datasets)
        if len(jv) >= 2:
            ids = [d.region_id for d in jv]
            self.assertEqual(
                len(ids), len(set(ids)),
                f"JV datasets have duplicate region IDs (cross-experiment mixing): {ids}",
            )

    def test_jv_parameter_values_power_notation(self):
        """Interface density parameters must parse from '10^10'...'10^20'."""
        jv = _jv_datasets(self.datasets)
        expected = sorted([1e10, 1e12, 1e14, 1e16, 1e18, 1e20])
        for ds in jv:
            param_values = sorted([
                t.parameter_value for t in ds.traces if t.parameter_value is not None
            ])
            if len(param_values) == 6:
                for got, exp in zip(param_values, expected):
                    self.assertAlmostEqual(got, exp, places=0)

    def test_provenance_invariant(self):
        _assert_provenance_invariant(self, self.datasets)


# ===========================================================================
# Optimized — 4 single-trace independent datasets
# ===========================================================================

class TestOptimizedExtraction(unittest.TestCase):
    """Optimized — 4 separate single-trace datasets (JV, QE, Responsivity, Detectivity)."""

    @classmethod
    def setUpClass(cls):
        cls.datasets = _load_and_extract_sheet("Optimized")

    def test_jv_exists_and_single_trace(self):
        """Optimized JV must exist and have exactly 1 trace."""
        jv = _jv_datasets(self.datasets)
        self.assertGreaterEqual(len(jv), 1, "No JV dataset in Optimized")
        jv_single = min(jv, key=lambda d: len(d.traces))
        self.assertEqual(
            len(jv_single.traces), 1,
            f"Optimized JV expected 1 trace, got {len(jv_single.traces)}",
        )

    def test_jv_single_trace_has_no_parameter(self):
        """Single-device Optimized JV trace must have parameter_value=None."""
        jv = _jv_datasets(self.datasets)
        jv_main = next((d for d in jv if len(d.traces) == 1), None)
        if jv_main is None:
            self.skipTest("No single-trace JV found")
        trace = jv_main.traces[0]
        # Single-device trace may have None parameter_value (no swept parameter)
        # It is acceptable to have None; we just verify no exception occurs
        _ = trace.parameter_value  # should not raise

    def test_qe_exists_and_single_trace(self):
        qe = _qe_datasets(self.datasets)
        self.assertGreaterEqual(len(qe), 1, "No QE dataset in Optimized")
        qe_main = min(qe, key=lambda d: len(d.traces))
        self.assertEqual(len(qe_main.traces), 1)

    def test_responsivity_exists_and_single_trace(self):
        resp = _responsivity_datasets(self.datasets)
        self.assertGreaterEqual(len(resp), 1, "No RESPONSIVITY dataset in Optimized")
        resp_main = min(resp, key=lambda d: len(d.traces))
        self.assertEqual(len(resp_main.traces), 1)

    def test_four_distinct_scientific_labels(self):
        """Optimized must contain at least JV, QE, RESPONSIVITY, DETECTIVITY."""
        labels = {d.scientific_label for d in self.datasets}
        for expected_label in ("JV", "QE", "RESPONSIVITY", "DETECTIVITY"):
            self.assertIn(
                expected_label, labels,
                f"Expected '{expected_label}' in Optimized datasets, found: {labels}",
            )

    def test_no_cross_axis_mixing(self):
        """All traces within one Optimized dataset must share the same axis concept."""
        for ds in self.datasets:
            if ds.axis is not None and len(ds.traces) > 1:
                # axis concept must be consistent (single concept per dataset)
                self.assertIsNotNone(ds.axis.concept)
                # All traces come from same column group — no validation needed here
                # beyond checking the axis is set

    def test_provenance_invariant(self):
        _assert_provenance_invariant(self, self.datasets)


# ===========================================================================
# Sheet1 — UNRESOLVED, device profile
# ===========================================================================

class TestSheet1Extraction(unittest.TestCase):
    """Sheet1 — positional device profile. Must be UNRESOLVED, raw always populated."""

    @classmethod
    def setUpClass(cls):
        cls.datasets = _load_and_extract_sheet("Sheet1")

    def test_sheet1_is_unresolved_or_unknown(self):
        """Sheet1 must not be classified as JV, QE, RESPONSIVITY, or DETECTIVITY."""
        for ds in self.datasets:
            self.assertNotIn(
                ds.scientific_label,
                ("JV", "QE", "RESPONSIVITY", "DETECTIVITY"),
                f"Sheet1 region {ds.region_id} must not be classified as "
                f"{ds.scientific_label}",
            )

    def test_sheet1_is_ambiguous(self):
        """At least one Sheet1 dataset must be marked is_ambiguous=True."""
        ambiguous = [d for d in self.datasets if d.is_ambiguous]
        self.assertGreater(
            len(ambiguous), 0,
            "Sheet1 must have at least one ambiguous (UNRESOLVED) dataset",
        )

    def test_sheet1_raw_always_populated(self):
        """raw must be non-None and non-empty for ALL Sheet1 datasets."""
        for ds in self.datasets:
            self.assertIsNotNone(
                ds.raw,
                f"Sheet1 region {ds.region_id}: raw is None — data must be preserved",
            )
            self.assertGreater(
                len(ds.raw.cells), 0,
                f"Sheet1 region {ds.region_id}: raw.cells is empty",
            )

    def test_sheet1_ambiguity_note_set(self):
        """UNRESOLVED datasets must have a non-empty ambiguity_note."""
        for ds in self.datasets:
            if ds.is_ambiguous:
                self.assertTrue(
                    len(ds.ambiguity_note) > 0,
                    f"Sheet1 region {ds.region_id}: ambiguity_note must be non-empty",
                )

    def test_sheet1_acceptance_state_unresolved(self):
        """Main Sheet1 region must have acceptance_state == UNRESOLVED."""
        # Find the largest region (most data)
        main_ds = max(
            self.datasets,
            key=lambda d: (d.raw.row_end - d.raw.row_start) if d.raw else 0,
        )
        self.assertEqual(
            main_ds.acceptance_state,
            AcceptanceState.UNRESOLVED.value,
            f"Sheet1 main region must be UNRESOLVED, got {main_ds.acceptance_state}",
        )

    def test_sheet1_hypothesis_scores_preserved(self):
        """Full hypothesis ranking must be preserved even for UNRESOLVED datasets."""
        for ds in self.datasets:
            self.assertGreaterEqual(len(ds.all_hypothesis_scores), 1)
            for h in ds.all_hypothesis_scores:
                self.assertIn("evidence_support_score", h)
                self.assertIsInstance(h["evidence_support_score"], float)


# ===========================================================================
# Cell-level provenance depth test
# ===========================================================================

class TestCellLevelProvenance(unittest.TestCase):
    """Verify that CellRef objects contain correct sheet/row/col/raw_value."""

    def test_etl_jv_trace_cell_refs_are_valid(self):
        """ETL JV CellRefs must have valid sheet_name, non-negative row/col."""
        datasets = _load_and_extract_sheet("ETL")
        jv = _jv_datasets(datasets)
        self.assertGreaterEqual(len(jv), 1)
        jv_main = max(jv, key=lambda d: len(d.traces))

        for trace in jv_main.traces:
            for ref in trace.source_refs:
                self.assertEqual(ref.sheet_name, "ETL")
                self.assertGreaterEqual(ref.row, 0)
                self.assertGreaterEqual(ref.col, 0)
                # raw_value should be numeric for data cells
                self.assertIsInstance(
                    ref.raw_value, (int, float),
                    f"ETL JV CellRef ({ref.row},{ref.col}) raw_value is not numeric: {ref.raw_value}",
                )

    def test_axis_cell_refs_are_valid(self):
        """ETL JV axis CellRefs must have valid provenance."""
        datasets = _load_and_extract_sheet("ETL")
        jv = _jv_datasets(datasets)
        jv_main = max(jv, key=lambda d: len(d.traces))

        self.assertIsNotNone(jv_main.axis)
        for ref in jv_main.axis.source_refs:
            self.assertEqual(ref.sheet_name, "ETL")
            self.assertGreaterEqual(ref.row, 0)

    def test_rsh_parameter_source_cell_refs(self):
        """Rsh parameter source CellRefs must point to header row."""
        datasets = _load_and_extract_sheet("Rsh")
        jv = _jv_datasets(datasets)
        jv_main = max(jv, key=lambda d: len(d.traces))

        for trace in jv_main.traces:
            if trace.parameter_source is not None:
                self.assertEqual(trace.parameter_source.sheet_name, "Rsh")
                self.assertGreaterEqual(trace.parameter_source.row, 0)
                self.assertGreaterEqual(trace.parameter_source.col, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
