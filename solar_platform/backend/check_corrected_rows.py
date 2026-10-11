"""
check_corrected_rows.py
Read-only report on the corrected_pv_rows table (the data behind the
"Corrected pipeline" option). It changes nothing.

Run from the backend folder:
    python check_corrected_rows.py
"""
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, ".")
from app.database import db_manager  # noqa: E402
from app.services.corrected_pv_dataset import implausible_reason  # noqa: E402

FEATURES = ["thickness", "defect_density", "temperature",
            "series_resistance", "shunt_resistance", "doping_density"]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

conn = sqlite3.connect(str(db_manager.DB_PATH))
try:
    df = pd.read_sql_query("SELECT * FROM corrected_pv_rows", conn)
except Exception as exc:
    print("Could not read corrected_pv_rows:", exc)
    print("Upload at least one workbook with the new backend running, then try again.")
    sys.exit(1)
finally:
    conn.close()

print(f"Database: {db_manager.DB_PATH}")
print(f"Total rows stored: {len(df)}\n")

print("Rows per workbook and status:")
print(df.groupby(["workbook_name", "status"]).size().unstack(fill_value=0).to_string(), "\n")

eligible = df[df["status"] == "ELIGIBLE"].copy()
eligible["problem"] = [
    implausible_reason({k: r[k] for k in ("j_sc", "v_oc", "ff", "pce")})
    for _, r in eligible.iterrows()
]
bad = eligible[eligible["problem"] != ""]

print(f"Rows marked ELIGIBLE: {len(eligible)}")
print(f"Of these, physically impossible (these are now filtered out before training): {len(bad)}\n")

if len(bad):
    print("Bad rows by workbook and sheet:")
    print(bad.groupby(["workbook_name", "sheet_name"]).size().to_string(), "\n")
    print("First 20 bad rows:")
    show = bad[["workbook_name", "sheet_name", "block_id", "row_index",
                "j_sc", "v_oc", "ff", "pce"] + FEATURES].head(20)
    print(show.to_string(index=False))
    print("\nPlease send this output (or a screenshot) so the cause can be fixed at the source.")
else:
    print("No impossible values found in the stored rows.")