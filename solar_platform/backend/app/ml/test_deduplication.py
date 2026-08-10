"""
app.ml.test_deduplication
-------------------------
Unit Test Suite for Deduplication & Duplicate Family Resolution Layer (app.ml.deduplication).
"""

import unittest
from app.canonical.model import CanonicalParameter, CanonicalTrace, ScientificDataset
from app.ml.deduplication import (
    DuplicateStatus,
    compute_experiment_family_id,
    resolve_duplicate_families,
)


class TestDeduplication(unittest.TestCase):
    """Test suite covering experiment_family_id generation and duplicate family resolution."""

    def test_compute_experiment_family_id_deterministic(self):
        """Verify experiment_family_id signature is deterministic and independent of dataset_id."""
        ds1 = ScientificDataset(
            dataset_id="ds_wb1_r1",
            workbook_name="Data file.xlsx",
            sheet_name="Thick+DD",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            global_parameters=[CanonicalParameter(name="temp", value=300)],
            x_values=[0.0, 0.1, 0.2],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[25.0, 24.0, 22.0])],
        )

        ds2 = ScientificDataset(
            dataset_id="ds_wb2_r1",
            workbook_name="Data File (1).xlsx",
            sheet_name="Thick+DD",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            global_parameters=[CanonicalParameter(name="temp", value=300)],
            x_values=[0.0, 0.1, 0.2],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[25.0, 24.0, 22.0])],
        )

        fam1 = compute_experiment_family_id(ds1)
        fam2 = compute_experiment_family_id(ds2)
        self.assertEqual(fam1, fam2)

    def test_exact_replica_resolution(self):
        """Cross-workbook exact replicas -> EXACT_REPLICA status and 1 representative selected."""
        ds1 = ScientificDataset(
            dataset_id="ds_wb1_r1",
            workbook_name="Data file.xlsx",
            sheet_name="Thick+DD",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            x_values=[0.0, 0.1],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[25.0, 24.0])],
        )
        ds2 = ScientificDataset(
            dataset_id="ds_wb2_r1",
            workbook_name="Data File (1).xlsx",
            sheet_name="Thick+DD",
            region_id="r1",
            parent_region_id=None,
            scientific_type="JV",
            structural_source="NumericCurveGrid",
            x_values=[0.0, 0.1],
            traces=[CanonicalTrace(trace_index=0, trace_id="t1", y_values=[25.0, 24.0])],
        )

        fams = resolve_duplicate_families([ds1, ds2])
        self.assertEqual(len(fams), 1)
        fam = fams[0]
        self.assertEqual(fam.duplicate_status, DuplicateStatus.EXACT_REPLICA)
        self.assertEqual(len(fam.datasets), 2)
        # Representative should prefer non-"(1)" workbook or first dataset
        self.assertEqual(fam.representative_dataset_id, "ds_wb1_r1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
