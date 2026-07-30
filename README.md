# Solar ETL Local Web App

This project now includes a local web app that wraps the ETL notebook workflow in an interactive UI.

## What the app does

- Loads ETL data from `Data_file.xlsx` (or uploaded `.xlsx`)
- Reproduces the same data cleaning and model training flow from the notebook
- Lets users set model/test/input values from controls instead of editing notebook cells
- Shows:
  - model metrics (`R2`, `MAE`)
  - point prediction (`Voltage`, `Thickness`)
  - full predicted J-V curve for any thickness
  - predicted-vs-derived comparison plot

## Run locally

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start app:

```bash
streamlit run app.py
```

4. Open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Notes

- Default data source is `Data_file.xlsx` in the project root.
- If your file or sheet name differs, upload a file and update the sheet name in the sidebar.
