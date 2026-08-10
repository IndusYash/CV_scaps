"""
app.ml.test_parameter_curation
-------------------------------
Unit Test Suite for Parameter Curation Layer (app.ml.parameter_curation).
"""

import unittest
from app.canonical.model import CanonicalParameter, CellRefDTO
from app.ml.parameter_curation import CurationStatus, curate_parameter


class TestParameterCuration(unittest.TestCase):
    """Test suite covering parameter curation, provenance preservation, and status classification."""

    def test_idd_float_cell_curation(self):
        """Sheet IDD float cell 0.00785... -> curated as defect_density 1e13 cm^-3."""
        prov = CellRefDTO(sheet_name="IDD", row=0, col=0, raw_value=0.007857558870967741)
        param = CanonicalParameter(name="thickness", value=0.007857558870967741, raw_text="0.007857558870967741", provenance=prov)

        cp = curate_parameter(param, workbook_name="Data file.xlsx", sheet_name="IDD", region_id="r0")
        self.assertEqual(cp.original_name, "thickness")
        self.assertEqual(cp.original_value, 0.007857558870967741)
        self.assertEqual(cp.original_provenance.sheet_name, "IDD")
        self.assertEqual(cp.curated_name, "defect_density")
        self.assertEqual(cp.curated_value, 1e13)
        self.assertEqual(cp.curated_unit, "cm^-3")
        self.assertEqual(cp.curation_status, CurationStatus.CURATED)

    def test_idd_temperature_header_curation(self):
        """Sheet IDD '1 K' header text -> curated as temperature 300 K."""
        prov = CellRefDTO(sheet_name="IDD", row=0, col=0, raw_value=1.0)
        param = CanonicalParameter(name="temperature", value=1.0, raw_text="1 K", provenance=prov)

        cp = curate_parameter(param, workbook_name="Data file.xlsx", sheet_name="IDD", region_id="r0")
        self.assertEqual(cp.curated_name, "temperature")
        self.assertEqual(cp.curated_value, 300.0)
        self.assertEqual(cp.curated_unit, "K")
        self.assertEqual(cp.curation_status, CurationStatus.CURATED)

    def test_summary_table_index_column_curation(self):
        """Summary table index integer col 0 -> INDEX_COLUMN status."""
        prov = CellRefDTO(sheet_name="NA", row=1, col=0, raw_value=1.0)
        param = CanonicalParameter(name="doping_density", value=1.0, raw_text="1", provenance=prov)

        cp = curate_parameter(param, workbook_name="Ca3AsBr3.xlsx", sheet_name="NA", region_id="region_0_rec_1_Voc")
        self.assertEqual(cp.curated_name, "sweep_index")
        self.assertEqual(cp.curated_value, 1)
        self.assertEqual(cp.curation_status, CurationStatus.INDEX_COLUMN)

    def test_confirmed_source_value_preservation(self):
        """Standard confirmed parameter (e.g. thickness = 100 nm) -> CONFIRMED_SOURCE_VALUE."""
        prov = CellRefDTO(sheet_name="Thick+DD", row=0, col=1, raw_value="t=100[nm]")
        param = CanonicalParameter(name="thickness", value=100.0, unit="nm", raw_text="t=100[nm]", provenance=prov)

        cp = curate_parameter(param, workbook_name="Data File (1).xlsx", sheet_name="Thick+DD", region_id="r0")
        self.assertEqual(cp.original_name, "thickness")
        self.assertEqual(cp.curated_name, "thickness")
        self.assertEqual(cp.curated_value, 100.0)
        self.assertEqual(cp.curation_status, CurationStatus.CONFIRMED_SOURCE_VALUE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
