import sys
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.pipeline import UnifiedParserPipeline
from app.services.experiment_grouper import ExperimentGrouperEngine


class TestPhase4ParameterSheets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wb_path = backend_dir.parent / "Ca3PI3.xlsx"
        if not wb_path.exists():
            wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"
            
        with open(wb_path, "rb") as f:
            cls.contents = f.read()

        cls.pipeline = UnifiedParserPipeline()
        cls.audit_result, cls.parsed_data = cls.pipeline.parse_workbook(
            wb_path.name, cls.contents, "test_phase4_session"
        )
        cls.grouper = ExperimentGrouperEngine()

    def test_tk_parameter_and_spectral_experiments(self):
        """Verify T (K) contains parameter-sweep experiments and spectral experiments."""
        sheet_audit = next(s for s in self.audit_result.sheets if s.sheet_name == "T (K)")
        exps = self.grouper.group_sheet_blocks_into_experiments(sheet_audit)
        
        exp_targets = [e.target_name for e in exps]
        self.assertIn("Voc", exp_targets)
        self.assertIn("Jsc", exp_targets)
        self.assertIn("FF", exp_targets)
        self.assertIn("PCE", exp_targets)
        self.assertIn("QE", exp_targets)

        voc_exp = next(e for e in exps if e.target_name == "Voc")
        self.assertEqual(voc_exp.axis_name, "Temperature")
        self.assertEqual(voc_exp.axis_unit, "K")
        self.assertEqual(voc_exp.target_unit, "V")
        self.assertEqual(len(voc_exp.combined_dataframe), 5)

    def test_na_parameter_and_spectral_experiments_with_distinct_traces(self):
        """Verify NA summary experiments exist and spectral QE contains distinct monotonic traces."""
        sheet_audit = next(s for s in self.audit_result.sheets if s.sheet_name == "NA")
        exps = self.grouper.group_sheet_blocks_into_experiments(sheet_audit)
        
        exp_targets = [e.target_name for e in exps]
        self.assertIn("Voc", exp_targets)
        self.assertIn("PCE", exp_targets)
        self.assertIn("QE", exp_targets)

        # Spectral QE experiment vs Wavelength
        qe_exp = next(e for e in exps if e.target_name == "QE" and e.axis_name == "Wavelength")
        df = qe_exp.combined_dataframe
        self.assertIn("Trace_ID", df.columns)
        
        trace_ids = df["Trace_ID"].unique()
        self.assertTrue(len(trace_ids) >= 5)

        # Check monotonic wavelength ordering in every trace
        for t_id in trace_ids:
            sub_df = df[df["Trace_ID"] == t_id]
            x_vals = sub_df["Wavelength"].tolist()
            self.assertEqual(len(x_vals), 61)
            for i in range(1, len(x_vals)):
                self.assertGreaterEqual(x_vals[i], x_vals[i - 1])

    def test_rs_parameter_and_spectral_experiments(self):
        """Verify Rs contains parameter-sweep experiments (Axis: Series_Resistance)."""
        sheet_audit = next(s for s in self.audit_result.sheets if s.sheet_name == "Rs")
        exps = self.grouper.group_sheet_blocks_into_experiments(sheet_audit)
        
        exp_targets = [e.target_name for e in exps]
        self.assertIn("Voc", exp_targets)
        self.assertIn("PCE", exp_targets)
        self.assertIn("QE", exp_targets)

        voc_exp = next(e for e in exps if e.target_name == "Voc")
        self.assertEqual(voc_exp.axis_name, "Series_Resistance")
        self.assertEqual(voc_exp.axis_unit, "ohm.cm^2")

    def test_rsh_parameter_and_spectral_experiments(self):
        """Verify Rsh contains parameter-sweep experiments (Axis: Shunt_Resistance)."""
        sheet_audit = next(s for s in self.audit_result.sheets if s.sheet_name == "Rsh")
        exps = self.grouper.group_sheet_blocks_into_experiments(sheet_audit)
        
        exp_targets = [e.target_name for e in exps]
        self.assertIn("Voc", exp_targets)
        self.assertIn("PCE", exp_targets)
        self.assertIn("QE", exp_targets)

        voc_exp = next(e for e in exps if e.target_name == "Voc")
        self.assertEqual(voc_exp.axis_name, "Shunt_Resistance")
        self.assertEqual(voc_exp.axis_unit, "ohm.cm^2")

    def test_regression_invariants(self):
        """Verify Illumination, QE (T+DD), QE (IDD), and Unoptimized do not regress."""
        # Illumination
        sheet_illum = next(s for s in self.audit_result.sheets if s.sheet_name == "Illumination")
        exps_illum = self.grouper.group_sheet_blocks_into_experiments(sheet_illum)
        self.assertEqual(len(exps_illum), 1)
        self.assertEqual(exps_illum[0].target_name, "Current")
        self.assertEqual(len(exps_illum[0].combined_dataframe["Trace_ID"].unique()), 4)

        # QE (T+DD)
        sheet_tdd = next(s for s in self.audit_result.sheets if s.sheet_name == "QE (T+DD)")
        exps_tdd = self.grouper.group_sheet_blocks_into_experiments(sheet_tdd)
        exp_qe_tdd = next(e for e in exps_tdd if e.target_name == "QE")
        self.assertEqual(len(exp_qe_tdd.blocks), 88)

        # QE (IDD)
        sheet_idd = next(s for s in self.audit_result.sheets if s.sheet_name == "QE (IDD)")
        exps_idd = self.grouper.group_sheet_blocks_into_experiments(sheet_idd)
        exp_qe_idd = next(e for e in exps_idd if e.target_name == "QE")
        self.assertEqual(len(exp_qe_idd.blocks), 16)


if __name__ == "__main__":
    unittest.main()
