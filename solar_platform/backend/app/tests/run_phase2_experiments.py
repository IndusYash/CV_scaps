import sys
import pandas as pd
import numpy as np
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.pipeline import UnifiedParserPipeline
from app.services.experiment_grouper import ExperimentGrouperEngine
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

with open(wb_path, "rb") as f:
    contents = f.read()

pipeline = UnifiedParserPipeline()
audit_result, parsed_data = pipeline.parse_workbook(wb_path.name, contents, "phase2_diag_session")

grouper = ExperimentGrouperEngine()

print("==========================================================================================")
print("                       PHASE 2 SCIENTIFIC EXPERIMENT REPORT                              ")
print("==========================================================================================")

# 1. QE (T+DD) Analysis
sheet_tdd = next((s for s in audit_result.sheets if s.sheet_name == "QE (T+DD)"), None)
if sheet_tdd:
    exps_tdd = grouper.group_sheet_blocks_into_experiments(sheet_tdd)
    exp = exps_tdd[0]
    df = exp.combined_dataframe
    print(f"\n--- QE (T+DD) GROUPED EXPERIMENT ---")
    print(f"  Experiment ID: {exp.experiment_id}")
    print(f"  Source Block Count: {len(exp.blocks)}")
    print(f"  Target: {exp.target_name} ({exp.target_unit})")
    print(f"  Unique Thickness values: {sorted(df['Thickness'].unique())}")
    print(f"  Unique Defect_Density values: {sorted(df['Defect_Density'].unique())}")
    
    # Check wavelengths per run
    run1_wl = df[df["Block_ID"] == exp.blocks[0].block_id]["Wavelength"].count()
    print(f"  Wavelengths per run: {run1_wl}")
    print(f"  Total Expected Rows (61 * 88): {61 * 88}")
    print(f"  Actual Combined Rows: {len(df)}")
    print(f"  Duplicate Row Count: {df.duplicated().sum()}")
    
    # ML Model Training & Sanity Check (QE = f(Wavelength, Thickness, Defect_Density))
    features = ["Wavelength", "Thickness", "Defect_Density"]
    X = df[features]
    y = df["QE"]
    
    # Grouped Holdout: Withhold entire run 10 (Defect_Density = 1e15)
    holdout_mask = (df["Defect_Density"] == 1e15)
    X_train, y_train = X[~holdout_mask], y[~holdout_mask]
    X_test, y_test = X[holdout_mask], y[holdout_mask]
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"\n  ML MODEL OWNERSHIP & SANITY CHECK:")
    print(f"    Model Owner: file_id='session', sheet='QE (T+DD)', experiment='{exp.experiment_id}', target='QE'")
    print(f"    Features: {features}")
    print(f"    Sample Count: {len(df)}")
    print(f"    Train Strategy: Grouped Parameter Holdout (Withheld Defect_Density=1e15)")
    print(f"    R^2 Score: {r2:.6f}")
    print(f"    MAE: {mae:.6f} %")

# 2. QE (IDD) Analysis
sheet_idd = next((s for s in audit_result.sheets if s.sheet_name == "QE (IDD)"), None)
if sheet_idd:
    exps_idd = grouper.group_sheet_blocks_into_experiments(sheet_idd)
    exp = exps_idd[0]
    df = exp.combined_dataframe
    print(f"\n--- QE (IDD) GROUPED EXPERIMENT ---")
    print(f"  Experiment ID: {exp.experiment_id}")
    print(f"  Source Block Count: {len(exp.blocks)}")
    print(f"  Target: {exp.target_name} ({exp.target_unit})")
    print(f"  Unique Defect_Density values ({len(df['Defect_Density'].unique())}): {sorted(df['Defect_Density'].unique())}")
    print(f"  Actual Combined Rows: {len(df)}")
    print(f"  Duplicate Row Count: {df.duplicated().sum()}")

# 3. Illumination Monotonic Trace Analysis
sheet_ill = next((s for s in audit_result.sheets if s.sheet_name == "Illumination"), None)
if sheet_ill:
    exps_ill = grouper.group_sheet_blocks_into_experiments(sheet_ill)
    exp = exps_ill[0]
    df = exp.combined_dataframe
    print(f"\n--- ILLUMINATION TRACE RECONSTRUCTION ---")
    print(f"  Total Rows: {len(df)}")
    for trace_id, group in df.groupby("Trace_ID"):
        is_mono = group["Voltage"].is_monotonic_increasing
        print(f"  Trace '{trace_id}': Count={len(group)}, Min V={group['Voltage'].min()}, Max V={group['Voltage'].max()}, Monotonic={is_mono}")
        print(f"        First 3 Current values: {list(group['Current'].head(3))}")

# 4. Optimized Multi-Target Separation
sheet_opt = next((s for s in audit_result.sheets if s.sheet_name == "Optimized"), None)
if sheet_opt:
    exps_opt = grouper.group_sheet_blocks_into_experiments(sheet_opt)
    print(f"\n--- OPTIMIZED MULTI-TARGET SEPARATION ---")
    print(f"  Produced {len(exps_opt)} Independent Scientific Experiments:")
    for exp in exps_opt:
        df = exp.combined_dataframe
        t_col = exp.target_name
        min_v = df[t_col].min() if not df.empty else 0.0
        max_v = df[t_col].max() if not df.empty else 0.0
        print(f"  Experiment '{exp.experiment_id}': Target='{t_col}' ({exp.target_unit}), Rows={len(df)}, Min={min_v}, Max={max_v}")
