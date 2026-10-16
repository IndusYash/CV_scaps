"""
clear_corrected_rows.py
Lists or clears the data behind the "Corrected pipeline" option, so you can start
a demo from a known set of workbooks.

It ONLY touches the corrected_pv_rows table. Your legacy database tables, your
uploaded sheets and the dashboard graphs are never changed.

Run from the backend folder:

    python clear_corrected_rows.py                    show what is stored (changes nothing)
    python clear_corrected_rows.py --workbook "Ca3PBr3.xlsx"
                                                      remove one workbook (repeat the flag for more)
    python clear_corrected_rows.py --all              remove every workbook

Before anything is deleted, a CSV backup of the removed rows is saved next to the
database, and you are asked to type "yes" to confirm.

To refill the table afterwards, upload the workbooks you want in the dashboard
(uploading is what fills it), then restart the backend and train.
"""
import argparse
import datetime
import pathlib
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, ".")

parser = argparse.ArgumentParser(description="List or clear the corrected_pv_rows table.")
parser.add_argument("--workbook", action="append", default=[],
                    help="workbook file name to remove (repeat for several)")
parser.add_argument("--all", action="store_true", help="remove every workbook")
parser.add_argument("--db", default=None, help="path to the database (default: the project database)")
args = parser.parse_args()

if args.db:
    db_path = pathlib.Path(args.db)
else:
    from app.database import db_manager
    db_path = pathlib.Path(db_manager.DB_PATH)

print(f"Database: {db_path}")

conn = sqlite3.connect(str(db_path))
try:
    df = pd.read_sql_query("SELECT * FROM corrected_pv_rows", conn)
except Exception:
    print("The corrected_pv_rows table does not exist yet (nothing has been uploaded with the new backend).")
    conn.close()
    sys.exit(0)

if df.empty:
    print("The corrected table is already empty.")
    conn.close()
    sys.exit(0)

summary = (df.groupby("workbook_name")["status"]
             .value_counts().unstack(fill_value=0)
             .reindex(columns=["ELIGIBLE", "EXCLUDED"], fill_value=0))
print("\nCurrently stored:")
print(summary.to_string())
print(f"\nTotal: {len(df)} rows from {df['workbook_name'].nunique()} workbook(s)")

if not args.all and not args.workbook:
    print("\nNothing was changed. To remove data, use --workbook \"Name.xlsx\" or --all.")
    conn.close()
    sys.exit(0)

if args.all:
    targets = sorted(df["workbook_name"].unique())
else:
    known = set(df["workbook_name"].unique())
    missing = [w for w in args.workbook if w not in known]
    if missing:
        print(f"\nNot found in the table (check spelling and the .xlsx ending): {missing}")
        print(f"Stored workbooks are: {sorted(known)}")
        conn.close()
        sys.exit(1)
    targets = args.workbook

to_delete = df[df["workbook_name"].isin(targets)]
print(f"\nAbout to remove {len(to_delete)} rows from: {targets}")
answer = input('Type "yes" to continue: ').strip().lower()
if answer != "yes":
    print("Cancelled. Nothing was changed.")
    conn.close()
    sys.exit(0)

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup = db_path.parent / f"corrected_pv_rows_backup_{stamp}.csv"
n = 2
while backup.exists():  # never overwrite an earlier backup
    backup = db_path.parent / f"corrected_pv_rows_backup_{stamp}_{n}.csv"
    n += 1
to_delete.to_csv(backup, index=False)
print(f"Backup saved: {backup}")

conn.executemany("DELETE FROM corrected_pv_rows WHERE workbook_name = ?", [(w,) for w in targets])
conn.commit()
remaining = conn.execute("SELECT COUNT(*) FROM corrected_pv_rows").fetchone()[0]
conn.close()

print(f"Done. Rows remaining in the corrected table: {remaining}")
print("Next: restart the backend, upload the workbooks you want, then train.")