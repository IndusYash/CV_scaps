"""
parser_v2.validator.test_validator
----------------------------------
Unit and integration tests for Stage 6 Validator.

Tests cover:
  1. Structural validation: length mismatches, missing axis, trace/ref mismatches.
  2. Semantic validation: axis concept mismatches, inconsistent units.
  3. Numerical validation: invalid numeric values (NaN/Inf), duplicate x-values, non-monotonic grids.
  4. Physical plausibility: warnings raised without mutating data.
  5. Unresolved datasets: Sheet1 & Optimized Detectivity preserved but is_ml_ready=False.
  6. Real workbook datasets: ETL JV, QE, Defect Density JV, Rs JV pass validation.
  7. Non-mutating guarantee: validation NEVER alters extracted values or structures.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import unittest
from pathlib import Path
from typing import List

_HERE = os.path.dirname(__file__)
_BACKEND = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from app.parser_v2.extractor.extract import extract_sheet_regions
from app.parser_v2.extractor.ir import (
    AxisData,
    ExtractedDataset,
    RawRegionData,
    ScalarRow,
    TraceData,
)
from app.parser_v2.probes.evidence import CellRef
from app.parser_v2.reader.raw_model import RawCell, RawCellType
from app.parser_v2.reader.raw_reader import read_workbook
from app.parser_v2.reasoner.experiment_classifier import classify_sheet_regions
from app.parser_v2.scanner.region_scanner import scan_sheet
from app.parser_v2.validator.models import (
    IssueSeverity,
    OverallValidationStatus,
    ValidationLevel,
)
from app.parser_v2.validator.validator import (
    validate_dataset,
    validate_sheet_datasets,
)

WORKBOOK_PATH = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\Data_file.xlsx")


def _load_and_validate_sheet(sheet_name: str):
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
    validated = validate_sheet_datasets(datasets)
    return datasets, validated


# ===========================================================================
# Synthetic Unit Tests for Each Validation Level
# ===========================================================================

class TestStructuralValidation(unittest.TestCase):
    """Level 1 Structural Validation tests."""

    def setUp(self):
        self.ref_a1 = CellRef("Sheet1", 0, 0, 0.0, "0.0")
        self.ref_a2 = CellRef("Sheet1", 1, 0, 0.5, "0.5")
        self.ref_b1 = CellRef("Sheet1", 0, 1, 10.0, "10.0")
        self.ref_b2 = CellRef("Sheet1", 1, 1, 20.0, "20.0")

        self.axis = AxisData(
            concept="Voltage", unit="V",
            values=[0.0, 0.5],
            source_refs=[self.ref_a1, self.ref_a2],
        )

    def test_x_y_length_mismatch(self):
        """Trace with 3 y-values against axis with 2 x-values must trigger X_Y_LENGTH_MISMATCH."""
        ref_b3 = CellRef("Sheet1", 2, 1, 30.0, "30.0")
        bad_trace = TraceData(
            trace_index=0, parameter_value=100.0, parameter_unit="nm",
            parameter_raw_text="t=100[nm]", parameter_source=self.ref_b1,
            values=[10.0, 20.0, 30.0],  # 3 values != 2 axis values
            source_refs=[self.ref_b1, self.ref_b2, ref_b3],
        )
        ds = ExtractedDataset(
            region_id="r0", sheet_name="Sheet1",
            structure_type="NumericCurveGrid", scientific_label="JV",
            axis_concept="Voltage", target_concept="Current",
            acceptance_state="ACCEPTED", winner_evidence_score=0.90,
            winner_runner_up_gap=0.30,
            axis=self.axis, traces=[bad_trace],
        )
        val = validate_dataset(ds)
        self.assertFalse(val.structural_valid)
        self.assertEqual(val.validation_status, OverallValidationStatus.INVALID_STRUCTURE)
        self.assertFalse(val.is_ml_ready)
        codes = [i.code for i in val.issues]
        self.assertIn("X_Y_LENGTH_MISMATCH", codes)

    def test_trace_ref_length_mismatch(self):
        """Trace with 2 values but 1 source_ref must trigger TRACE_REF_MISMATCH."""
        bad_trace = TraceData(
            trace_index=0, parameter_value=100.0, parameter_unit="nm",
            parameter_raw_text="t=100[nm]", parameter_source=self.ref_b1,
            values=[10.0, 20.0],
            source_refs=[self.ref_b1],  # 1 ref != 2 values
        )
        ds = ExtractedDataset(
            region_id="r0", sheet_name="Sheet1",
            structure_type="NumericCurveGrid", scientific_label="JV",
            axis_concept="Voltage", target_concept="Current",
            acceptance_state="ACCEPTED", winner_evidence_score=0.90,
            winner_runner_up_gap=0.30,
            axis=self.axis, traces=[bad_trace],
        )
        val = validate_dataset(ds)
        self.assertFalse(val.structural_valid)
        codes = [i.code for i in val.issues]
        self.assertIn("TRACE_REF_MISMATCH", codes)

    def test_missing_axis_with_traces(self):
        """NumericCurveGrid with traces but axis=None must trigger MISSING_AXIS."""
        trace = TraceData(
            trace_index=0, parameter_value=100.0, parameter_unit="nm",
            parameter_raw_text="t=100[nm]", parameter_source=self.ref_b1,
            values=[10.0, 20.0], source_refs=[self.ref_b1, self.ref_b2],
        )
        ds = ExtractedDataset(
            region_id="r0", sheet_name="Sheet1",
            structure_type="NumericCurveGrid", scientific_label="JV",
            axis_concept="Voltage", target_concept="Current",
            acceptance_state="ACCEPTED", winner_evidence_score=0.90,
            winner_runner_up_gap=0.30,
            axis=None, traces=[trace],
        )
        val = validate_dataset(ds)
        self.assertFalse(val.structural_valid)
        codes = [i.code for i in val.issues]
        self.assertIn("MISSING_AXIS", codes)


class TestSemanticValidation(unittest.TestCase):
    """Level 2 Semantic Validation tests."""

    def test_axis_concept_mismatch(self):
        """Extracted axis concept 'Wavelength' vs Reasoner axis_concept 'Voltage' must trigger AXIS_CONCEPT_MISMATCH."""
        ref = CellRef("Sheet1", 0, 0, 300.0, "300")
        axis = AxisData(concept="Wavelength", unit="nm", values=[300.0, 400.0], source_refs=[ref, ref])
        ds = ExtractedDataset(
            region_id="r0", sheet_name="Sheet1",
            structure_type="NumericCurveGrid", scientific_label="JV",
            axis_concept="Voltage", target_concept="Current",
            acceptance_state="ACCEPTED", winner_evidence_score=0.90,
            winner_runner_up_gap=0.30,
            axis=axis, traces=[],
        )
        val = validate_dataset(ds)
        codes = [i.code for i in val.issues]
        self.assertIn("AXIS_CONCEPT_MISMATCH", codes)

    def test_inconsistent_parameter_units(self):
        """Traces with different units ('nm' vs 'K') must trigger INCONSISTENT_PARAMETER_UNITS."""
        ref = CellRef("Sheet1", 0, 0, 1.0, "1")
        axis = AxisData(concept="Voltage", unit="V", values=[0.0], source_refs=[ref])
        t0 = TraceData(0, 100.0, "nm", "t=100[nm]", ref, [1.0], [ref])
        t1 = TraceData(1, 300.0, "K", "T=300K", ref, [2.0], [ref])
        ds = ExtractedDataset(
            region_id="r0", sheet_name="Sheet1",
            structure_type="NumericCurveGrid", scientific_label="JV",
            axis_concept="Voltage", target_concept="Current",
            acceptance_state="ACCEPTED", winner_evidence_score=0.90,
            winner_runner_up_gap=0.30,
            axis=axis, traces=[t0, t1],
        )
        val = validate_dataset(ds)
        codes = [i.code for i in val.issues]
        self.assertIn("INCONSISTENT_PARAMETER_UNITS", codes)


class TestNumericalValidation(unittest.TestCase):
    """Level 3 Numerical Validation tests."""

    def test_invalid_axis_values_nan(self):
        """Axis containing NaN value must trigger INVALID_NUMERIC_AXIS."""
        ref = CellRef("Sheet1", 0, 0, 0.0, "0.0")
        axis = AxisData(concept="Voltage", unit="V", values=[0.0, float("nan")], source_refs=[ref, ref])
        ds = ExtractedDataset(
            region_id="r0", sheet_name="Sheet1",
            structure_type="NumericCurveGrid", scientific_label="JV",
            axis_concept="Voltage", target_concept="Current",
            acceptance_state="ACCEPTED", winner_evidence_score=0.90,
            winner_runner_up_gap=0.30,
            axis=axis, traces=[],
        )
        val = validate_dataset(ds)
        codes = [i.code for i in val.issues]
        self.assertIn("INVALID_NUMERIC_AXIS", codes)

    def test_duplicate_x_values(self):
        """Axis containing duplicate x-values [0.0, 0.5, 0.5] must trigger DUPLICATE_X_VALUE."""
        ref = CellRef("Sheet1", 0, 0, 0.0, "0.0")
        axis = AxisData(concept="Voltage", unit="V", values=[0.0, 0.5, 0.5], source_refs=[ref, ref, ref])
        ds = ExtractedDataset(
            region_id="r0", sheet_name="Sheet1",
            structure_type="NumericCurveGrid", scientific_label="JV",
            axis_concept="Voltage", target_concept="Current",
            acceptance_state="ACCEPTED", winner_evidence_score=0.90,
            winner_runner_up_gap=0.30,
            axis=axis, traces=[],
        )
        val = validate_dataset(ds)
        codes = [i.code for i in val.issues]
        self.assertIn("DUPLICATE_X_VALUE", codes)

    def test_non_monotonic_axis(self):
        """Axis values [0.0, 0.5, 0.2] must trigger NON_MONOTONIC_AXIS."""
        ref = CellRef("Sheet1", 0, 0, 0.0, "0.0")
        axis = AxisData(concept="Voltage", unit="V", values=[0.0, 0.5, 0.2], source_refs=[ref, ref, ref])
        ds = ExtractedDataset(
            region_id="r0", sheet_name="Sheet1",
            structure_type="NumericCurveGrid", scientific_label="JV",
            axis_concept="Voltage", target_concept="Current",
            acceptance_state="ACCEPTED", winner_evidence_score=0.90,
            winner_runner_up_gap=0.30,
            axis=axis, traces=[],
        )
        val = validate_dataset(ds)
        codes = [i.code for i in val.issues]
        self.assertIn("NON_MONOTONIC_AXIS", codes)

    def test_uncached_formula_value(self):
        """Raw cell containing formula with None raw_value must trigger UNCACHED_FORMULA_VALUE."""
        ref = CellRef("Sheet1", 0, 0, 0.0, "0.0")
        formula_cell = RawCell(
            row=0, col=0, data_type=RawCellType.FORMULA, raw_value=None,
            display_str="", formula_str="=SUM(A1:A5)", is_bold=False,
            background_color=None, is_merged=False, merge_parent=None,
        )
        raw_data = RawRegionData("r0", "Sheet1", 0, 0, 0, 0, [[formula_cell]])
        ds = ExtractedDataset(
            region_id="r0", sheet_name="Sheet1",
            structure_type="NumericCurveGrid", scientific_label="JV",
            axis_concept="Voltage", target_concept="Current",
            acceptance_state="ACCEPTED", winner_evidence_score=0.90,
            winner_runner_up_gap=0.30,
            axis=None, traces=[], raw=raw_data,
        )
        val = validate_dataset(ds)
        codes = [i.code for i in val.issues]
        self.assertIn("UNCACHED_FORMULA_VALUE", codes)


class TestPhysicalPlausibilityValidation(unittest.TestCase):
    """Level 4 Physical Plausibility tests (non-mutating warning tests)."""

    def test_unusual_voltage_range_warning(self):
        """Voltage axis range [-10V, +10V] must trigger UNUSUAL_VOLTAGE_RANGE warning without mutating dataset."""
        ref = CellRef("Sheet1", 0, 0, 0.0, "0.0")
        axis = AxisData(concept="Voltage", unit="V", values=[-10.0, 10.0], source_refs=[ref, ref])
        trace = TraceData(0, 100.0, "nm", "t=100[nm]", ref, [1.0, 2.0], [ref, ref])
        ds = ExtractedDataset(
            region_id="r0", sheet_name="Sheet1",
            structure_type="NumericCurveGrid", scientific_label="JV",
            axis_concept="Voltage", target_concept="Current",
            acceptance_state="ACCEPTED", winner_evidence_score=0.90,
            winner_runner_up_gap=0.30,
            axis=axis, traces=[trace],
        )

        original_copy = copy.deepcopy(ds)
        val = validate_dataset(ds)

        # 1. Check issue raised
        codes = [i.code for i in val.issues]
        self.assertIn("UNUSUAL_VOLTAGE_RANGE", codes)
        self.assertFalse(val.physical_plausible)
        self.assertEqual(val.validation_status, OverallValidationStatus.ACCEPTED_WITH_WARNINGS)
        self.assertTrue(val.is_ml_ready)

        # 2. NON-MUTATING ASSERTION: ds must be identical to original_copy
        self.assertEqual(ds, original_copy, "Validation MUST NOT mutate dataset fields!")

    def test_unusual_qe_range_warning(self):
        """QE trace with max value 200% must trigger UNUSUAL_QE_RANGE warning without data loss."""
        ref = CellRef("Sheet1", 0, 0, 300.0, "300")
        axis = AxisData(concept="Wavelength", unit="nm", values=[300.0, 400.0], source_refs=[ref, ref])
        trace = TraceData(0, 100.0, "nm", "t=100[nm]", ref, [50.0, 200.0], [ref, ref])
        ds = ExtractedDataset(
            region_id="r0", sheet_name="Sheet1",
            structure_type="NumericCurveGrid", scientific_label="QE",
            axis_concept="Wavelength", target_concept="QE",
            acceptance_state="ACCEPTED", winner_evidence_score=0.90,
            winner_runner_up_gap=0.30,
            axis=axis, traces=[trace],
        )

        val = validate_dataset(ds)
        codes = [i.code for i in val.issues]
        self.assertIn("UNUSUAL_QE_RANGE", codes)
        self.assertFalse(val.physical_plausible)
        # Trace data is 100% preserved
        self.assertEqual(len(val.dataset.traces[0].values), 2)
        self.assertEqual(val.dataset.traces[0].values, [50.0, 200.0])


# ===========================================================================
# Real Workbook Integration Tests
# ===========================================================================

class TestRealWorkbookValidation(unittest.TestCase):
    """Validation of real Data_file.xlsx extracted datasets."""

    def test_etl_jv_validation_pass(self):
        """ETL JV dataset must validate as ACCEPTED_VALID or ACCEPTED_WITH_WARNINGS and be is_ml_ready=True."""
        datasets, validated = _load_and_validate_sheet("ETL")
        jv_val = [v for v in validated if v.dataset.scientific_label == "JV"]
        self.assertGreaterEqual(len(jv_val), 1)
        main_jv = max(jv_val, key=lambda v: len(v.dataset.traces))

        self.assertTrue(main_jv.structural_valid)
        self.assertTrue(main_jv.semantic_valid)
        self.assertIn(
            main_jv.validation_status,
            (OverallValidationStatus.ACCEPTED_VALID, OverallValidationStatus.ACCEPTED_WITH_WARNINGS),
        )
        self.assertTrue(main_jv.is_ml_ready)

    def test_etl_qe_validation_pass(self):
        """ETL QE dataset must validate as ACCEPTED and be is_ml_ready=True."""
        datasets, validated = _load_and_validate_sheet("ETL")
        qe_val = [v for v in validated if v.dataset.scientific_label == "QE"]
        self.assertGreaterEqual(len(qe_val), 1)
        main_qe = max(qe_val, key=lambda v: len(v.dataset.traces))

        self.assertTrue(main_qe.structural_valid)
        self.assertTrue(main_qe.is_ml_ready)

    def test_defect_density_jv_validation_pass(self):
        """Defect Density JV dataset must validate successfully and be is_ml_ready=True."""
        datasets, validated = _load_and_validate_sheet("Defect Density")
        jv_val = [v for v in validated if v.dataset.scientific_label == "JV"]
        self.assertGreaterEqual(len(jv_val), 1)
        main_jv = max(jv_val, key=lambda v: len(v.dataset.traces))

        self.assertTrue(main_jv.structural_valid)
        self.assertTrue(main_jv.is_ml_ready)

    def test_rs_jv_validation_pass(self):
        """Rs JV dataset must validate successfully and be is_ml_ready=True."""
        datasets, validated = _load_and_validate_sheet("Rs")
        jv_val = [v for v in validated if v.dataset.scientific_label == "JV"]
        self.assertGreaterEqual(len(jv_val), 1)
        main_jv = max(jv_val, key=lambda v: len(v.dataset.traces))

        self.assertTrue(main_jv.structural_valid)
        self.assertTrue(main_jv.is_ml_ready)

    def test_sheet1_unresolved_preserved_not_ml_ready(self):
        """
        Sheet1 unresolved dataset must:
          - Have validation_status == UNRESOLVED
          - Have is_ml_ready == False
          - Preserve raw data (raw.cells non-empty)
        """
        datasets, validated = _load_and_validate_sheet("Sheet1")
        self.assertGreaterEqual(len(validated), 1)
        main_val = max(
            validated,
            key=lambda v: (v.dataset.raw.row_end - v.dataset.raw.row_start) if v.dataset.raw else 0,
        )

        self.assertEqual(main_val.validation_status, OverallValidationStatus.UNRESOLVED)
        self.assertFalse(main_val.is_ml_ready, "UNRESOLVED dataset MUST NOT be marked is_ml_ready!")
        # Raw data preserved
        self.assertIsNotNone(main_val.dataset.raw)
        self.assertGreater(len(main_val.dataset.raw.cells), 0)

    def test_optimized_detectivity_unresolved_not_ml_ready(self):
        """Optimized Detectivity (UNRESOLVED state) must have is_ml_ready == False while preserving raw."""
        datasets, validated = _load_and_validate_sheet("Optimized")
        unresolved_val = [v for v in validated if v.dataset.acceptance_state == "UNRESOLVED"]
        self.assertGreaterEqual(len(unresolved_val), 1, "Optimized must contain an UNRESOLVED dataset")

        for val in unresolved_val:
            self.assertEqual(val.validation_status, OverallValidationStatus.UNRESOLVED)
            self.assertFalse(val.is_ml_ready, "UNRESOLVED Detectivity MUST NOT be marked is_ml_ready!")
            self.assertIsNotNone(val.dataset.raw)

    def test_validation_never_mutates_extracted_values(self):
        """
        Non-mutating guarantee check on real datasets:
        Assert deep equality of every dataset before and after validate_dataset().
        """
        datasets, validated = _load_and_validate_sheet("ETL")
        for ds in datasets:
            ds_copy = copy.deepcopy(ds)
            _ = validate_dataset(ds)
            self.assertEqual(
                ds, ds_copy,
                f"Validation mutated ExtractedDataset region {ds.region_id}!",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
