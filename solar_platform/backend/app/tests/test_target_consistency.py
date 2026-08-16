import unittest
import sys
from pathlib import Path

# Add backend to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.pipeline import UnifiedParserPipeline
from app.services.ml_engine import register_parsed_experiments, get_parsed_experiments

class TestTargetConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fpath = Path(r"C:\Users\yashv\Desktop\REPOS\Solar_Project\solar_platform\Data_file.xlsx")
        with open(cls.fpath, "rb") as f:
            cls.contents = f.read()
        
        cls.pipeline = UnifiedParserPipeline()
        cls.file_id = "test_target_consistency_id"
        cls.audit_res, cls.parsed_data = cls.pipeline.parse_workbook("Data_file.xlsx", cls.contents, cls.file_id)
        register_parsed_experiments(cls.file_id, cls.audit_res)
        cls.experiments = get_parsed_experiments(cls.file_id, "ETL")

    def test_jv_traces(self):
        # JV Experiment
        jv_exp = next((e for e in self.experiments if e.target_name == "Current" and e.axis_name == "Voltage"), None)
        self.assertIsNotNone(jv_exp, "JV experiment (Current vs Voltage) not found")
        df = jv_exp.combined_dataframe
        # 10 traces: 'Jsc_block_ETL_1', and 't=200[nm]_c2_block_ETL_1' to 't=800[nm]'
        unique_traces = df["Trace_ID"].unique().tolist()
        print("JV Unique Traces:", unique_traces)
        self.assertEqual(len(unique_traces), 10, f"Expected 10 JV traces, got: {unique_traces}")

    def test_qe_traces(self):
        # QE Experiment
        qe_exp = next((e for e in self.experiments if e.target_name == "QE" and e.axis_name == "Wavelength"), None)
        self.assertIsNotNone(qe_exp, "QE experiment (QE vs Wavelength) not found")
        df = qe_exp.combined_dataframe
        unique_traces = df["Trace_ID"].unique().tolist()
        print("QE Unique Traces:", unique_traces)
        
        # Clean trace count should be 10 and trace labels must belong to block_ETL_3
        # Responsivity trace labels should not be in QE
        for t in unique_traces:
            self.assertIn("block_ETL_3", t, f"QE trace {t} should belong to block_ETL_3")
            self.assertNotIn("block_ETL_4", t, f"Responsivity trace {t} was merged into QE!")
        self.assertEqual(len(unique_traces), 10, f"Expected 10 QE traces, got: {unique_traces}")

    def test_responsivity_traces(self):
        # Responsivity Experiment
        resp_exp = next((e for e in self.experiments if e.target_name == "Responsivity" and e.axis_name == "Wavelength"), None)
        self.assertIsNotNone(resp_exp, "Responsivity experiment not found")
        df = resp_exp.combined_dataframe
        unique_traces = df["Trace_ID"].unique().tolist()
        print("Responsivity Unique Traces:", unique_traces)
        
        # Clean trace count should be 10 (all belonging to block_ETL_4)
        for t in unique_traces:
            self.assertIn("block_ETL_4", t, f"Responsivity trace {t} should belong to block_ETL_4")
        self.assertEqual(len(unique_traces), 10, f"Expected 10 Responsivity traces, got: {unique_traces}")

    def test_jsc_lookup_not_qe(self):
        # The lookup block block_ETL_2 has target Jsc/Current, and is not mapped to QE
        # So in the QE experiment, no traces from block_ETL_2 should be merged
        qe_exp = next((e for e in self.experiments if e.target_name == "QE" and e.axis_name == "Wavelength"), None)
        if qe_exp is not None:
            df = qe_exp.combined_dataframe
            unique_traces = df["Trace_ID"].unique().tolist()
            for t in unique_traces:
                self.assertNotIn("block_ETL_2", t, f"Lookup table trace {t} was incorrectly classified as QE!")

    def test_unique_experiment_ids(self):
        # ETL JV and ETL Jsc sweep must receive different experiment IDs
        jv_exp = next((e for e in self.experiments if e.target_name == "Current" and e.axis_name == "Voltage"), None)
        jsc_exp = next((e for e in self.experiments if e.target_name == "Current" and e.axis_name == "Thickness"), None)
        
        self.assertIsNotNone(jv_exp, "JV experiment not found")
        self.assertIsNotNone(jsc_exp, "Jsc sweep experiment not found")
        self.assertNotEqual(jv_exp.experiment_id, jsc_exp.experiment_id, "JV and Jsc sweep shared the same ID!")
        self.assertEqual(jv_exp.semantic_type, "JV", "JV semantic type mismatch")
        self.assertEqual(jsc_exp.semantic_type, "SCALAR_SWEEP", "Jsc sweep semantic type mismatch")
        
        # Verify no duplicate experiment IDs exist in the payload
        all_ids = [e.experiment_id for e in self.experiments]
        self.assertEqual(len(all_ids), len(set(all_ids)), f"Duplicate experiment IDs found in payload: {all_ids}")

    def test_experiment_id_stability(self):
        # Existing JV, QE, Responsivity experiment IDs remain stable (matching legacy IDs)
        jv_exp = next((e for e in self.experiments if e.target_name == "Current" and e.axis_name == "Voltage"), None)
        qe_exp = next((e for e in self.experiments if e.target_name == "QE" and e.axis_name == "Wavelength"), None)
        resp_exp = next((e for e in self.experiments if e.target_name == "Responsivity" and e.axis_name == "Wavelength"), None)
        
        self.assertEqual(jv_exp.experiment_id, "exp_ETL_Current")
        self.assertEqual(qe_exp.experiment_id, "exp_ETL_QE")
        self.assertEqual(resp_exp.experiment_id, "exp_ETL_Responsivity")
        
        # Verify QE and Responsivity semantic types remain QE and RESPONSIVITY
        self.assertEqual(qe_exp.semantic_type, "QE")
        self.assertEqual(resp_exp.semantic_type, "RESPONSIVITY")

    def test_jsc_sweep_structure_and_grouping(self):
        # Jsc-vs-thickness remains a scalar/parameter sweep and is not converted into a JV curve
        jsc_exp = next((e for e in self.experiments if e.target_name == "Current" and e.axis_name == "Thickness"), None)
        self.assertEqual(jsc_exp.experiment_id, "exp_ETL_Current_vs_Thickness")
        self.assertEqual(jsc_exp.semantic_type, "SCALAR_SWEEP")
        
        # Verify traces are grouped together under one experiment
        df = jsc_exp.combined_dataframe
        unique_traces = df["Trace_ID"].unique().tolist()
        # Should be exactly 1 trace containing 10 points
        self.assertEqual(len(unique_traces), 1)
        self.assertEqual(len(df), 10)
        self.assertEqual(jsc_exp.axis_name, "Thickness")
        self.assertEqual(jsc_exp.target_name, "Current")

if __name__ == "__main__":
    unittest.main()
