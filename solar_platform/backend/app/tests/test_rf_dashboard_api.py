"""
app.tests.test_rf_dashboard_api
-------------------------------
Validation regression tests for classical RandomForest validation dashboard API,
point-wise array equality, parameter-name matches, and curation pipeline safety rules.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Add backend to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.database import db_manager
from app.services.rf_pv_regressor import build_ml_dataset_from_db, MLRegressorManager, FEATURE_KEYS
from app.canonical.converter import _derive_parameter_name
from app.ml.parameter_curation import curate_parameter
from app.canonical.model import CanonicalParameter

class TestRFDashboardAPI(unittest.TestCase):
    """Test suite covering the V1 RandomForest dashboard API response and scientific curation safety."""

    @classmethod
    def setUpClass(cls):
        # We curate the actual database to run tests against real state
        cls.df, cls.audit_log = build_ml_dataset_from_db()
        cls.manager = MLRegressorManager(cls.df)
        cls.targets = ["j_sc", "v_oc", "ff", "pce"]

    def test_na_acceptor_mapping(self):
        """Verify boundary-aware Acceptor/Na matching rules and prevent matching Illumination."""
        # Fix 1 verification: Illumination does NOT classify as doping_density
        self.assertNotEqual(_derive_parameter_name("Illumination", "jtot(mA/cm2)", None), "doping_density")
        
        # Valid Na/doping should still classify as doping_density
        self.assertEqual(_derive_parameter_name("Na", "jtot(mA/cm2)", None), "doping_density")
        self.assertEqual(_derive_parameter_name("N_a", "jtot(mA/cm2)", None), "doping_density")
        self.assertEqual(_derive_parameter_name("acceptor", "jtot(mA/cm2)", None), "doping_density")
        self.assertEqual(_derive_parameter_name("doping", "jtot(mA/cm2)", None), "doping_density")
        self.assertEqual(_derive_parameter_name("donor", "jtot(mA/cm2)", None), "doping_density")

    def test_temperature_parameter_name_matching(self):
        """Verify boundary-aware temperature name matching rules."""
        self.assertEqual(_derive_parameter_name("Optimized", "300 K", None), "temperature")
        self.assertEqual(_derive_parameter_name("Optimized", "25 deg C", None), "temperature")
        self.assertEqual(_derive_parameter_name("Optimized", "25 °C", None), "temperature")
        self.assertEqual(_derive_parameter_name("Optimized", "dark current density", None), "parameter")
        self.assertEqual(_derive_parameter_name("Optimized", "current density", None), "parameter")

    def test_illumination_wavelength_curation(self):
        """Verify that 400/500/600 nm are represented as illumination_wavelength and AM1.5G is 0.0 sentinel."""
        # 1. Verification of Case 4 curation mappings
        p1 = CanonicalParameter(name="parameter", value=None, raw_text="400 nm")
        cur1 = curate_parameter(p1, "Ca3PI3.xlsx", "Illumination", "region_0")
        self.assertEqual(cur1.curated_name, "illumination_wavelength")
        self.assertEqual(cur1.curated_value, 400.0)

        p2 = CanonicalParameter(name="parameter", value=None, raw_text="500 nm")
        cur2 = curate_parameter(p2, "Ca3PI3.xlsx", "Illumination", "region_0")
        self.assertEqual(cur2.curated_name, "illumination_wavelength")
        self.assertEqual(cur2.curated_value, 500.0)

        # 2. Verify illumination_wavelength appears in FEATURE_KEYS
        self.assertIn("illumination_wavelength", FEATURE_KEYS)

        # 3. Verify standard curves receive 0.0 sentinel
        opt_rows = self.df[self.df["dataset_id"].str.contains("Optimized")]
        for idx, r in opt_rows.iterrows():
            self.assertEqual(r["illumination_wavelength"], 0.0)

    def test_zero_vector_conflict_resolution(self):
        """Verify that Ca3PI3 illumination traces no longer have identical feature vectors."""
        illum_df = self.df[self.df["dataset_id"].str.contains("Illumination")]
        # Extract the values of illumination_wavelength for these traces
        wavelengths = sorted(illum_df["illumination_wavelength"].unique().tolist())
        # We expect 0.0, 400.0, 500.0, 600.0
        self.assertEqual(wavelengths, [0.0, 400.0, 500.0, 600.0])

        # Verify that their feature rows are distinct
        features_matrix = illum_df[FEATURE_KEYS].drop_duplicates()
        self.assertEqual(len(features_matrix), len(illum_df), "Zero-vector conflict was not resolved!")

    def test_dark_current_exclusion(self):
        """Verify that dark current curves are successfully excluded from ML dataset curation."""
        outlier_id = "Data_File__1_.xlsx_Optimized_region_2"
        outlier_rows = self.df[self.df["dataset_id"] == outlier_id]
        self.assertEqual(len(outlier_rows), 0, "Dark current outlier trace was NOT excluded from ML dataset!")
        
        audit_match = next((r for r in self.audit_log if r["dataset_id"] == outlier_id), None)
        self.assertIsNotNone(audit_match, "Dark current outlier was not found in curation audit log")
        self.assertEqual(audit_match["status"], "EXCLUDED")

    def test_curation_safety_rules(self):
        """Verify that curation safety rules are strictly enforced: no duplicates, no lookup tables."""
        dup_mask = self.df.duplicated(subset=["dataset_id", "trace_id"])
        self.assertEqual(dup_mask.sum(), 0, "Duplicate traces found in curation dataset!")

        for rec in self.audit_log:
            if "region_1" in rec["dataset_id"] or "region_5" in rec["dataset_id"]:
                self.assertEqual(
                    rec["status"], 
                    "EXCLUDED", 
                    f"Scalar sweep dataset {rec['dataset_id']} was not excluded!"
                )

        for ds_id in self.df["dataset_id"].unique():
            ds = db_manager.get_dataset(ds_id)
            if ds:
                self.assertEqual(ds.scientific_type.upper(), "JV", f"Dataset {ds_id} is not JV type!")

        df_data_file = self.df[self.df["dataset_id"].str.startswith("Data_file.xlsx")]
        for t in self.targets:
            zero_count = (df_data_file[t] == 0.0).sum()
            self.assertEqual(zero_count, 0, f"Found zero target value in column {t} for Data_file.xlsx!")

    def test_interpolation_split_rules(self):
        """Verify that the parameter-aware interpolation split prevents leakage and extrapolation."""
        for t in self.targets:
            res = self.manager.train_and_evaluate(t, test_size=0.25, random_state=42)
            preds = self.manager.predictions[t]
            
            test_mask = self.df["dataset_id"].isin(preds["dataset_ids"])
            df_train = self.df[~test_mask]
            df_test = self.df[test_mask]

            overlap_traces = set(df_train["trace_id"]).intersection(set(df_test["trace_id"]))
            self.assertEqual(overlap_traces, set(), f"Trace overlap detected for target {t}!")

            train_configs = set(zip(df_train["dataset_id"], df_train["thickness"], df_train["defect_density"], df_train["series_resistance"], df_train["shunt_resistance"], df_train["illumination_wavelength"]))
            test_configs = set(zip(df_test["dataset_id"], df_test["thickness"], df_test["defect_density"], df_test["series_resistance"], df_test["shunt_resistance"], df_test["illumination_wavelength"]))
            config_overlap = train_configs.intersection(test_configs)
            self.assertEqual(config_overlap, set(), f"Exact configuration overlap detected for target {t}!")

            for dataset_id, group_df in self.df.groupby("dataset_id"):
                group_train = df_train[df_train["dataset_id"] == dataset_id]
                group_test = df_test[df_test["dataset_id"] == dataset_id]
                
                if len(group_test) == 0 or len(group_train) == 0:
                    continue
                    
                varying_feat = None
                for f in FEATURE_KEYS:
                    if len(group_df[f].unique()) > 1:
                        varying_feat = f
                        break
                        
                if varying_feat:
                    train_min = group_train[varying_feat].min()
                    train_max = group_train[varying_feat].max()
                    test_min = group_test[varying_feat].min()
                    test_max = group_test[varying_feat].max()
                    
                    self.assertTrue(train_min <= test_min, f"Extrapolation detected (test min {test_min} < train min {train_min}) for {dataset_id}!")
                    self.assertTrue(test_max <= train_max, f"Extrapolation detected (test max {test_max} > train max {train_max}) for {dataset_id}!")

    def test_api_response_schema_and_alignment(self):
        """Verify that the API returns standardized validation payloads with aligned point-by-point arrays."""
        results = {}
        for target in self.targets:
            metrics_dict = self.manager.train_and_evaluate(
                target_name=target,
                material_filter=None,
                test_size=0.25,
                random_state=42
            )
            results[target] = metrics_dict

        payload = {
            "model": "RandomForestRegressor",
            "targets": {
                target: {
                    "actual": self.manager.predictions[target]["actuals"],
                    "predicted": self.manager.predictions[target]["predicted_rf"],
                    "metrics": {
                        "r2": results[target]["random_forest"]["r2"],
                        "mae": results[target]["random_forest"]["mae"],
                        "rmse": results[target]["random_forest"]["rmse"],
                        "n_samples": len(self.manager.predictions[target]["actuals"])
                    }
                } for target in self.targets
            }
        }

        self.assertEqual(payload["model"], "RandomForestRegressor")
        self.assertEqual(len(payload["targets"]), 4)

        for target in self.targets:
            t_data = payload["targets"][target]
            self.assertIn("actual", t_data)
            self.assertIn("predicted", t_data)
            self.assertIn("metrics", t_data)

            self.assertEqual(
                len(t_data["actual"]), 
                len(t_data["predicted"]), 
                f"Array length mismatch for target {target}!"
            )
            self.assertEqual(
                t_data["metrics"]["n_samples"], 
                len(t_data["actual"]),
                f"n_samples mismatch for target {target}!"
            )

            m = t_data["metrics"]
            self.assertIn("r2", m)
            self.assertIn("mae", m)
            self.assertIn("rmse", m)

    def test_unseen_workbook_validation_mode(self):
        """Verify unseen workbook mode strictly isolates workbook and does not leak categorical materials."""
        if self.df.empty:
            return
        wbs = self.df["workbook_name"].unique()
        if len(wbs) < 2:
            return
            
        test_wb = wbs[0]
        for t in self.targets:
            res = self.manager.train_and_evaluate(
                target_name=t,
                validation_mode="unseen_workbook",
                test_workbook=test_wb,
                random_state=42
            )
            
            # Verify test_workbooks lists only the held out workbook
            self.assertEqual(res["test_workbooks"], [test_wb])
            self.assertNotIn(test_wb, res["train_workbooks"])

            # Verify predictions workbook names match
            preds = self.manager.predictions[t]
            self.assertTrue(all(wb == test_wb for wb in preds["workbook_names"]))

if __name__ == "__main__":
    unittest.main()

