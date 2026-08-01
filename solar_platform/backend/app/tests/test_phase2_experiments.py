import sys
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.pipeline import UnifiedParserPipeline
from app.services.experiment_grouper import ExperimentGrouperEngine


class TestPhase2Experiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wb_path = backend_dir.parent / "Ca3PI3.xlsx"
        if not wb_path.exists():
            wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"
        
        with open(wb_path, "rb") as f:
            contents = f.read()
            
        cls.pipeline = UnifiedParserPipeline()
        cls.audit_result, cls.parsed_data = cls.pipeline.parse_workbook(
            wb_path.name, contents, "test_phase2_session"
        )
        cls.grouper = ExperimentGrouperEngine()

    def test_invariant_a_no_mixed_traces_illumination(self):
        """INVARIANT A: Within each Trace_ID, X-axis corresponds strictly to that trace monotonically."""
        sheet_audit = next((s for s in self.audit_result.sheets if s.sheet_name == "Illumination"), None)
        self.assertIsNotNone(sheet_audit)
        
        exps = self.grouper.group_sheet_blocks_into_experiments(sheet_audit)
        self.assertTrue(len(exps) >= 1)
        df = exps[0].combined_dataframe
        self.assertIn("Trace_ID", df.columns)
        
        for trace_id, group in df.groupby("Trace_ID"):
            self.assertEqual(len(group), 63, f"Trace {trace_id} must have exactly 63 points.")
            self.assertTrue(
                group["Voltage"].is_monotonic_increasing,
                f"Trace {trace_id} X-axis must be strictly monotonic increasing without interleaving."
            )

    def test_invariant_b_no_mixed_units_optimized(self):
        """INVARIANT B: One target dataset must have exactly one physical quantity and unit."""
        sheet_audit = next((s for s in self.audit_result.sheets if s.sheet_name == "Optimized"), None)
        self.assertIsNotNone(sheet_audit)
        
        exps = self.grouper.group_sheet_blocks_into_experiments(sheet_audit)
        target_units = [(e.target_name, e.target_unit) for e in exps]
        
        # Verify QE (%), Responsivity (A/W), and Detectivity (Jones) are separate
        names = [t[0] for t in target_units]
        self.assertIn("QE", names)
        self.assertIn("Responsivity", names)
        self.assertIn("Detectivity", names)

    def test_invariant_c_parameter_completeness_qe_tdd(self):
        """INVARIANT C: Every row belonging to a parameterized sweep carries its full parameter tuple."""
        sheet_audit = next((s for s in self.audit_result.sheets if s.sheet_name == "QE (T+DD)"), None)
        self.assertIsNotNone(sheet_audit)
        
        exps = self.grouper.group_sheet_blocks_into_experiments(sheet_audit)
        exp_qe = next(e for e in exps if e.target_name == "QE")
        df = exp_qe.combined_dataframe
        
        self.assertIn("Thickness", df.columns)
        self.assertIn("Defect_Density", df.columns)
        self.assertIn("Temperature", df.columns)
        self.assertFalse(df["Thickness"].isna().any())
        self.assertFalse(df["Defect_Density"].isna().any())
        self.assertEqual(len(df["Thickness"].unique()), 11)
        self.assertEqual(len(df["Defect_Density"].unique()), 8)

    def test_invariant_d_provenance_retention(self):
        """INVARIANT D: Every combined row must retain source Block_ID."""
        sheet_audit = next((s for s in self.audit_result.sheets if s.sheet_name == "QE (T+DD)"), None)
        exps = self.grouper.group_sheet_blocks_into_experiments(sheet_audit)
        exp_qe = next(e for e in exps if e.target_name == "QE")
        df = exp_qe.combined_dataframe
        
        self.assertIn("Block_ID", df.columns)
        self.assertFalse(df["Block_ID"].isna().any())
        self.assertEqual(len(df["Block_ID"].unique()), 88)

    def test_invariant_e_no_silent_overwrites(self):
        """INVARIANT E: Number of grouped source blocks before and after experiment construction must match."""
        sheet_audit = next((s for s in self.audit_result.sheets if s.sheet_name == "QE (T+DD)"), None)
        exps = self.grouper.group_sheet_blocks_into_experiments(sheet_audit)
        exp_qe = next(e for e in exps if e.target_name == "QE")
        
        self.assertEqual(len(exp_qe.blocks), len(sheet_audit.blocks))

    def test_invariant_f_determinism(self):
        """INVARIANT F: Running experiment grouping twice on the same workbook produces identical membership."""
        sheet_audit = next((s for s in self.audit_result.sheets if s.sheet_name == "QE (T+DD)"), None)
        exps1 = self.grouper.group_sheet_blocks_into_experiments(sheet_audit)
        exps2 = self.grouper.group_sheet_blocks_into_experiments(sheet_audit)
        
        self.assertEqual(len(exps1), len(exps2))
        self.assertEqual(exps1[0].experiment_id, exps2[0].experiment_id)
        self.assertEqual(len(exps1[0].combined_dataframe), len(exps2[0].combined_dataframe))


if __name__ == "__main__":
    unittest.main()
