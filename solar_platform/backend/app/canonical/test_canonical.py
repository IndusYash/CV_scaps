"""
app.canonical.test_canonical
----------------------------
Comprehensive Unit & Integration Test Suite for Canonical Scientific Dataset Layer.
"""

import json
import tempfile
import unittest
from pathlib import Path

from app.canonical.converter import (
    build_canonical_workbook,
    convert_validated_dataset_to_canonical,
)
from app.canonical.model import (
    CanonicalParameter,
    CanonicalScalarRecord,
    CanonicalTrace,
    CanonicalWorkbook,
    CellRefDTO,
    ScientificDataset,
)
from app.parser_v2.extractor.extract import extract_sheet_regions
from app.parser_v2.reader.raw_reader import read_workbook
from app.parser_v2.reasoner.experiment_classifier import (
    classify_region,
    classify_sheet_regions,
)
from app.parser_v2.scanner.region_scanner import scan_sheet
from app.parser_v2.validator.validator import (
    validate_dataset,
    validate_sheet_datasets,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
WORKBOOK_DATA_FILE = BASE_DIR / "Data File (1).xlsx"
WORKBOOK_CA3PI3 = BASE_DIR / "Ca3PI3.xlsx"


class TestCanonicalModelSerialization(unittest.TestCase):
    """Test model data structures, JSON serialization, and CSV tabular export."""

    def test_cell_ref_dto_serialization(self):
        dto = CellRefDTO(sheet_name="Illumination", row=10, col=2, raw_value=25.5)
        d = dto.to_dict()
        self.assertEqual(d["sheet_name"], "Illumination")
        self.assertEqual(d["row"], 10)
        self.assertEqual(d["col"], 2)

        dto2 = CellRefDTO.from_dict(d)
        self.assertEqual(dto2.sheet_name, dto.sheet_name)
        self.assertEqual(dto2.row, dto.row)
        self.assertEqual(dto2.col, dto.col)
        self.assertEqual(dto2.raw_value, dto.raw_value)

    def test_scientific_dataset_json_roundtrip(self):
        param = CanonicalParameter(
            name="temperature",
            value=300.0,
            unit="K",
            raw_text="T=300K",
            provenance=CellRefDTO(sheet_name="JV", row=0, col=1, raw_value="T=300K"),
        )
        trace = CanonicalTrace(
            trace_index=0,
            trace_id="ds_1_trace_1",
            parameter=param,
            y_values=[25.0, 24.8, 24.5],
            y_provenance=[
                CellRefDTO(sheet_name="JV", row=1, col=1, raw_value=25.0),
                CellRefDTO(sheet_name="JV", row=2, col=1, raw_value=24.8),
                CellRefDTO(sheet_name="JV", row=3, col=1, raw_value=24.5),
            ],
        )
        ds = ScientificDataset(
            dataset_id="test_ds_1",
            workbook_name="TestWb",
            sheet_name="JV",
            region_id="region_0",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            axis_name="Voltage",
            axis_unit="V",
            target_name="CurrentDensity",
            target_unit="mA/cm2",
            x_values=[0.0, 0.2, 0.4],
            x_provenance=[
                CellRefDTO(sheet_name="JV", row=1, col=0, raw_value=0.0),
                CellRefDTO(sheet_name="JV", row=2, col=0, raw_value=0.2),
                CellRefDTO(sheet_name="JV", row=3, col=0, raw_value=0.4),
            ],
            traces=[trace],
            validation_status="ACCEPTED_VALID",
            acceptance_state="ACCEPTED",
            is_ml_ready=True,
            confidence_score=1.0,
            ambiguity_gap=0.8,
        )

        json_str = ds.to_json()
        ds_restored = ScientificDataset.from_json(json_str)

        self.assertEqual(ds_restored.dataset_id, ds.dataset_id)
        self.assertEqual(ds_restored.scientific_type, ds.scientific_type)
        self.assertEqual(ds_restored.x_values, ds.x_values)
        self.assertEqual(len(ds_restored.traces), 1)
        self.assertEqual(ds_restored.traces[0].parameter.value, 300.0)
        self.assertEqual(ds_restored.traces[0].parameter.provenance.sheet_name, "JV")

    def test_tabular_csv_export(self):
        param = CanonicalParameter(name="thickness", value=100.0, unit="nm", raw_text="t=100nm")
        trace = CanonicalTrace(
            trace_index=0,
            trace_id="ds_qe_trace_1",
            parameter=param,
            y_values=[85.0, 90.0],
        )
        ds = ScientificDataset(
            dataset_id="ds_qe",
            workbook_name="Wb",
            sheet_name="QE",
            region_id="reg_0",
            parent_region_id=None,
            scientific_type="QE",
            structural_source="NumericCurveGrid",
            axis_name="Wavelength",
            axis_unit="nm",
            x_values=[400.0, 500.0],
            traces=[trace],
        )

        csv_str = ds.to_csv()
        self.assertIn("dataset_id,scientific_type,trace_id,parameter_name,parameter_value,parameter_unit,Wavelength,y_value", csv_str)
        self.assertIn("ds_qe,QE,ds_qe_trace_1,thickness,100.0,nm,400.0,85.0", csv_str)

    def test_canonical_workbook_csv_folder_export(self):
        param = CanonicalParameter(name="thickness", value=100.0, unit="nm")
        rec = CanonicalScalarRecord(
            record_id="rec_1",
            parameter=param,
            target_name="Jsc",
            target_value=24.5,
            target_unit="mA/cm2",
        )
        ds = ScientificDataset(
            dataset_id="scalar_ds_1",
            workbook_name="Wb",
            sheet_name="Summary",
            region_id="reg_1",
            parent_region_id=None,
            scientific_type="SCALAR_SWEEP",
            structural_source="ScalarSweepTable",
            scalar_records=[rec],
            is_ml_ready=True,
        )
        wb = CanonicalWorkbook(workbook_name="TestWb", datasets=[ds])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_files = wb.to_csv_folder(tmpdir)
            self.assertEqual(len(out_files), 1)
            content = out_files[0].read_text(encoding="utf-8")
            self.assertIn("scalar_ds_1,SCALAR_SWEEP,thickness,100.0,nm,Jsc,24.5", content)


class TestCanonicalConversionFromParserV2(unittest.TestCase):
    """Test real and synthetic conversion across all 5 structural forms."""

    def test_numeric_curve_grid_to_canonical(self):
        """Standard shared-axis NumericCurveGrid (Unoptimized sheet) -> Canonical ScientificDataset."""
        if not WORKBOOK_CA3PI3.exists():
            self.skipTest(f"Workbook not found at {WORKBOOK_CA3PI3}")

        with open(WORKBOOK_CA3PI3, "rb") as f:
            file_bytes = f.read()

        raw_wb = read_workbook(WORKBOOK_CA3PI3.name, file_bytes)
        sheet = raw_wb.get_sheet("Unoptimized")
        regions = scan_sheet(sheet)
        interps = classify_sheet_regions(sheet, regions)
        extracted = extract_sheet_regions(sheet, regions, interps)
        validated = validate_sheet_datasets(extracted)

        val_ds = next(v for v in validated if v.dataset.structure_type == "NumericCurveGrid")
        canonical = convert_validated_dataset_to_canonical(val_ds, workbook_name=WORKBOOK_CA3PI3.name)

        self.assertIn(canonical.scientific_type, ["JV", "QE", "RESPONSIVITY"])
        self.assertEqual(canonical.structural_source, "NumericCurveGrid")
        self.assertIsNotNone(canonical.axis_name)
        self.assertGreaterEqual(len(canonical.traces), 1)
        self.assertEqual(len(canonical.x_values), len(canonical.x_provenance))
        self.assertTrue(canonical.is_ml_ready)

    def test_scalar_sweep_table_to_canonical(self):
        """ScalarSweepTable (Thick+DD summary table) -> Canonical ScientificDataset parameter-scalar records."""
        if not WORKBOOK_DATA_FILE.exists():
            self.skipTest(f"Workbook not found at {WORKBOOK_DATA_FILE}")

        with open(WORKBOOK_DATA_FILE, "rb") as f:
            file_bytes = f.read()

        raw_wb = read_workbook(WORKBOOK_DATA_FILE.name, file_bytes)
        sheet = raw_wb.get_sheet("Thick+DD")
        regions = scan_sheet(sheet)
        interps = classify_sheet_regions(sheet, regions)
        extracted = extract_sheet_regions(sheet, regions, interps)
        validated = validate_sheet_datasets(extracted)

        val_ds = next(v for v in validated if v.dataset.structure_type == "ScalarSweepTable")
        canonical = convert_validated_dataset_to_canonical(val_ds, workbook_name=WORKBOOK_DATA_FILE.name)

        self.assertEqual(canonical.scientific_type, "SCALAR_SWEEP")
        self.assertEqual(canonical.structural_source, "ScalarSweepTable")
        self.assertGreaterEqual(len(canonical.scalar_records), 10)

        # Verify scalar parameter mapping structure
        rec = canonical.scalar_records[0]
        self.assertIsNotNone(rec.parameter)
        self.assertTrue(len(rec.target_name) > 0)
        self.assertIsNotNone(rec.target_provenance)

    def test_stacked_curve_blocks_to_canonical(self):
        """StackedCurveBlocks (QE T+DD) -> 88 canonical ScientificDataset objects."""
        if not WORKBOOK_CA3PI3.exists():
            self.skipTest(f"Workbook not found at {WORKBOOK_CA3PI3}")

        with open(WORKBOOK_CA3PI3, "rb") as f:
            file_bytes = f.read()

        raw_wb = read_workbook(WORKBOOK_CA3PI3.name, file_bytes)
        sheet = raw_wb.get_sheet("QE (T+DD)")
        regions = scan_sheet(sheet)

        stacked_reg = next(r for r in regions if r.n_rows > 1000)
        interp = classify_region(stacked_reg, sheet)
        extracted = extract_sheet_regions(sheet, [stacked_reg], [interp])
        validated = validate_sheet_datasets(extracted)

        self.assertEqual(len(validated), 88)

        canonical_list = [convert_validated_dataset_to_canonical(v, workbook_name="Ca3PI3.xlsx") for v in validated]

        for c in canonical_list:
            self.assertEqual(c.scientific_type, "QE")
            self.assertEqual(c.structural_source, "StackedCurveBlocks")
            self.assertEqual(c.axis_name, "Wavelength")
            self.assertEqual(len(c.x_values), len(c.x_provenance))
            self.assertTrue(c.is_ml_ready)
            self.assertIsNotNone(c.raw_dataset_reference)

    def test_interleaved_curve_grid_to_canonical(self):
        """InterleavedCurveGrid (Illumination) -> 4 canonical curve datasets."""
        if not WORKBOOK_CA3PI3.exists():
            self.skipTest(f"Workbook not found at {WORKBOOK_CA3PI3}")

        with open(WORKBOOK_CA3PI3, "rb") as f:
            file_bytes = f.read()

        raw_wb = read_workbook(WORKBOOK_CA3PI3.name, file_bytes)
        sheet = raw_wb.get_sheet("Illumination")
        regions = scan_sheet(sheet)

        interp = classify_region(regions[0], sheet)
        extracted = extract_sheet_regions(sheet, [regions[0]], [interp])
        validated = validate_sheet_datasets(extracted)

        self.assertEqual(len(validated), 4)

        canonical_list = [convert_validated_dataset_to_canonical(v, workbook_name="Ca3PI3.xlsx") for v in validated]

        for c in canonical_list:
            self.assertIn(c.scientific_type, ["JV", "QE"])
            self.assertEqual(c.structural_source, "InterleavedCurveGrid")
            self.assertEqual(c.axis_name, "Voltage")
            self.assertEqual(c.axis_unit, "V")
            self.assertEqual(len(c.x_values), len(c.x_provenance))
            self.assertTrue(c.is_ml_ready)

        # Check parameter recovery on curves 2, 3, 4 (400 nm, 500 nm, 600 nm)
        self.assertEqual(canonical_list[1].traces[0].parameter.value, 400.0)
        self.assertEqual(canonical_list[2].traces[0].parameter.value, 500.0)
        self.assertEqual(canonical_list[3].traces[0].parameter.value, 600.0)

    def test_no_cross_experiment_mixing_and_raw_recoverability(self):
        """Verify unique dataset IDs, no cross-mixing, and 100% raw recoverability."""
        if not WORKBOOK_DATA_FILE.exists():
            self.skipTest(f"Workbook not found at {WORKBOOK_DATA_FILE}")

        with open(WORKBOOK_DATA_FILE, "rb") as f:
            file_bytes = f.read()

        raw_wb = read_workbook(WORKBOOK_DATA_FILE.name, file_bytes)
        sheet = raw_wb.get_sheet("Thick+DD")
        regions = scan_sheet(sheet)
        interps = classify_sheet_regions(sheet, regions)
        extracted = extract_sheet_regions(sheet, regions, interps)
        validated = validate_sheet_datasets(extracted)

        wb_canonical = build_canonical_workbook("Data File (1).xlsx", validated)
        ds_ids = [d.dataset_id for d in wb_canonical.datasets]

        # Check unique dataset IDs
        self.assertEqual(len(ds_ids), len(set(ds_ids)))

        # Check raw recoverability pointer
        for ds in wb_canonical.datasets:
            self.assertIsNotNone(ds.raw_dataset_reference)
            self.assertIn("region_id", ds.raw_dataset_reference)
            self.assertIn("score", ds.raw_dataset_reference)


if __name__ == "__main__":
    unittest.main(verbosity=2)
