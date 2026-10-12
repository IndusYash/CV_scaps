# SCAPS ML Pipeline

Automatic interpretation of SCAPS-1D photovoltaic simulation workbooks, with Random Forest prediction of solar-cell performance, delivered as an interactive web dashboard.

Built at the Computer Vision Laboratory, Department of Computer Science and Engineering, Madan Mohan Malaviya University of Technology (MMMUT), Gorakhpur. Supervisor: Dr. Satvik Vats.

## What it does

SCAPS-1D exports are Excel workbooks whose layout changes from sheet to sheet: single-parameter sweeps, two-parameter sweeps, several tables side by side, and curve data (J-V, QE, responsivity). Cleaning each one by hand is slow and breaks whenever a new material or sweep appears. This project replaces that with:

1. **Structural parsing.** Each uploaded workbook is read, split into tables, and each table is interpreted (axis, swept parameters, measured quantities) from its own structure, not from a fixed template.
2. **Physical validation.** Reconstructed data is checked for consistency before it is used.
3. **Interactive graphs.** Every experiment in every sheet is plotted in the dashboard.
4. **Per-sheet Random Forest models.** Train, predict a point, or predict a full curve and compare it with the simulated one.
5. **Global Random Forest (V1 ML Reg. tab).** Learns Jsc, Voc, FF and PCE from the swept device parameters across all stored workbooks, with mean and linear baselines and parity plots.

## Quick start (Windows)

You need Python 3.10 or newer and a recent Node.js.

**Backend** (terminal 1):

```
cd solar_platform\backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

**Frontend** (terminal 2):

```
cd solar_platform\frontend
npm install
npm run dev
```

Open http://localhost:3000. The frontend proxies `/api` to the backend on port 8000, so start the backend first.

Sample SCAPS workbooks (Ca3AsBr3, Ca3PBr3, Ca3PI3, Data File and others) are in `solar_platform/`; files named `... copy.xlsx` are duplicates. Upload any of them, or your own SCAPS export, with **Upload New Dataset**. New `.xlsx` files are git-ignored.

## Using the dashboard

- **Worksheets tab:** pick a sheet to see its experiments as interactive plots. **Train All Worksheets** fits a Random Forest for every sheet of the uploaded workbook.
- **V1 ML Reg. tab:** trains the global model.
  - **Validation Mode:** *Within-Dataset Interpolation* holds out random simulation runs. *True Unseen Workbook* holds out a whole workbook (needs at least two stored workbooks).
  - **Training Data Source:** *Legacy database* (default) uses rows derived from J-V curves. *Corrected pipeline* uses the corrected summary sweeps (PCE, Voc, Jsc, FF versus thickness, defect density, temperature, resistances and doping).
  - **Stored Workbooks card** (corrected source only): lists the workbooks the model will train on. Remove one, or clear all, then train again. Upload a workbook to add it.

The corrected table is filled when a workbook is uploaded. Re-uploading a workbook replaces its rows.

## Results

All figures below were measured on real workbooks (Ca3AsBr3, Ca3AsI3, Ca3PBr3, Ca3PI3 and an ETL-style thickness workbook). R-squared (R2) compares the error with the spread of the test values, so quote MAE next to it.

| Setting | Jsc R2 | Voc R2 | FF R2 | PCE R2 |
|---|---|---|---|---|
| Corrected source, 3 workbooks, interpolation (65 test rows) | 0.980 | 0.991 | 0.944 | 0.977 |
| Legacy source, same tab | negative | negative | negative | negative |
| One workbook (Ca3PI3), median of 20 random splits | 0.994 | 0.786 | 0.422 | 0.885 |

On a single workbook the test set is only about 23 rows, so one unlucky split can give a large negative R2 even when the error is small. Jsc is stable across splits.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/upload` | Upload and parse a workbook |
| POST | `/api/curves` | Experiments and traces for a sheet |
| POST | `/api/train` | Train per-sheet Random Forest models |
| POST | `/api/predict/point` | Single-point prediction |
| POST | `/api/predict/curve` | Predicted versus actual curve |
| POST | `/api/ml/train/random-forest` | Train the global model (`data_source=db` or `corrected`, `validation_mode`, `test_workbook`) |
| GET | `/api/ml/results` | Latest global results and parity data |
| GET | `/api/ml/corrected/workbooks` | List workbooks in the corrected data |
| DELETE | `/api/ml/corrected/workbook?name=...` | Remove one workbook from the corrected data |
| DELETE | `/api/ml/corrected/workbooks` | Remove all workbooks from the corrected data |
| GET | `/api/database/status`, `/api/datasets` | Legacy database status and contents |

Interactive documentation is at http://127.0.0.1:8000/docs while the backend runs.

## Project layout

```
solar_platform/
  backend/
    app/
      readers/ layout/ domain/ semantics/ validation/   parsing pipeline
      services/                                          Random Forest training, experiment grouping,
                                                         corrected_pv_dataset.py (global-model data)
      api/                                               FastAPI endpoints
      database/                                          SQLite persistence (legacy source)
      parser_v2/ canonical/                              older second parser feeding the legacy source
    check_corrected_rows.py                              report on the corrected data
    clear_corrected_rows.py                              list or clear the corrected data (with backup)
  frontend/                                              React + TypeScript + Vite, Plotly charts
```

## Data checks on the corrected source

Rows are excluded, and listed with a reason, when they are incomplete, not illuminated (near-zero Jsc), or outside physical limits (Jsc 0.1 to 100 mA/cm2, Voc up to 3 V, FF up to 1, PCE up to 60 %). Sweeps whose parameter has no matching model input (the interface-defect sweeps in IDD sheets) are excluded and reported.

## Maintenance scripts

Run from `solar_platform/backend` with the virtual environment active:

```
python check_corrected_rows.py                     # report on stored rows, changes nothing
python clear_corrected_rows.py                     # list what is stored, changes nothing
python clear_corrected_rows.py --workbook "Name.xlsx"
python clear_corrected_rows.py --all               # asks for confirmation, saves a CSV backup
```

