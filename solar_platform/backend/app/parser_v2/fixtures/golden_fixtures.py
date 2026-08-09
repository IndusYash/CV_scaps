"""
Golden fixtures for the new SCAPS parser v2.

These fixtures capture the structural facts and expected experiment
interpretations derived directly from Data_file.xlsx via openpyxl inspection.

Purpose:
    Establish scientific ground truth that parser_v2 must reproduce.
    These tests are independent of implementation details (no cell coordinates
    as expected values unless the coordinate itself is the behaviour under test).

Fixture format:
    Each fixture is a plain dict. Tests should load these and assert
    against the parser_v2 output without referencing old-parser internals.

Workbook: Data_file.xlsx (11 sheets)
Sheets: ETL, PVK, HTL, Temperature, Defect Density, Interface density,
        Donor Con, Rsh, Rs, Optimized, Sheet1
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Helper types (just dicts — no dependencies on parser_v2 yet)
# ---------------------------------------------------------------------------

Fixture = Dict[str, Any]


# ===========================================================================
# STRUCTURAL FIXTURES
# Describe what the scanner/reader must observe about each sheet.
# These are verified before any experiment classification.
# ===========================================================================

STRUCTURAL_FIXTURES: List[Fixture] = [

    # -----------------------------------------------------------------------
    # ETL — Electron Transport Layer thickness sweep
    # -----------------------------------------------------------------------
    {
        "sheet": "ETL",
        "description": (
            "10 JV traces for ETL thickness 10–100 nm, "
            "plus scalar Jsc summary, QE curves, and Responsivity curves "
            "arranged in 4 side-by-side sub-table groups separated by empty columns."
        ),
        "expected_dims": {
            "n_rows_min": 60,
            "n_rows_max": 70,
            "n_cols_min": 40,
            "n_cols_max": 50,
        },
        "expected_empty_col_boundaries_count": 4,  # 4 empty-column barriers
        "expected_sub_table_count": 4,             # JV, Jsc summary, QE, Responsivity
        "expected_header_row_index": 0,            # row 0 is always the header
        "expected_parameter_in_header": "t=",      # column headers contain t=<value>[nm]
        "expected_parameter_unit": "nm",
        "expected_axis_names": ["v(V)", "lambda(nm)"],  # two different x-axes across sub-tables
        "expected_numeric_density_gte": 0.35,
    },

    # -----------------------------------------------------------------------
    # PVK — Perovskite absorber thickness sweep
    # -----------------------------------------------------------------------
    {
        "sheet": "PVK",
        "description": (
            "10 JV traces for PVK thickness 100–1000 nm, "
            "scalar Jsc summary, 10 QE traces, 10 Responsivity traces, "
            "arranged in 4 groups separated by empty columns."
        ),
        "expected_dims": {
            "n_rows_min": 60,
            "n_rows_max": 70,
            "n_cols_min": 40,
            "n_cols_max": 45,
        },
        "expected_empty_col_boundaries_count": 4,
        "expected_sub_table_count": 4,
        "expected_header_row_index": 0,
        "expected_parameter_in_header": "t=",
        "expected_parameter_values": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "expected_parameter_unit": "nm",
        "expected_axis_names": ["v(V)", "lambda(nm)"],
        "expected_numeric_density_gte": 0.35,
    },

    # -----------------------------------------------------------------------
    # Temperature — Temperature sweep
    # -----------------------------------------------------------------------
    {
        "sheet": "Temperature",
        "description": (
            "6 JV traces for T=300,330,360,390,420,450 K, "
            "scalar Jsc summary (col 10-11), 6 QE traces, 6 Responsivity traces."
        ),
        "expected_dims": {
            "n_rows_min": 58,
            "n_rows_max": 65,
            "n_cols_min": 30,
            "n_cols_max": 35,
        },
        "expected_empty_col_boundaries_count": 4,
        "expected_sub_table_count": 4,
        "expected_header_row_index": 0,
        "expected_parameter_in_header": "T=",
        "expected_parameter_values": [300, 330, 360, 390, 420, 450],
        "expected_parameter_unit": "K",
        "expected_axis_names": ["v(V)", "lambda(nm)"],
    },

    # -----------------------------------------------------------------------
    # Defect Density — Bulk defect density sweep
    # -----------------------------------------------------------------------
    {
        "sheet": "Defect Density",
        "description": (
            "6 JV traces for defect densities 10^13 to 10^18 cm^-3, "
            "scalar Jsc summary, 6 QE traces, 6 Responsivity traces. "
            "Column headers encode density as '10^N' strings."
        ),
        "expected_dims": {
            "n_rows_min": 58,
            "n_rows_max": 65,
            "n_cols_min": 30,
            "n_cols_max": 35,
        },
        "expected_empty_col_boundaries_count": 4,
        "expected_sub_table_count": 4,
        "expected_header_row_index": 0,
        "expected_parameter_in_header": "10^",   # '10^13', '10^14', ... in col headers
        "expected_parameter_values_numeric": [1e13, 1e14, 1e15, 1e16, 1e17, 1e18],
        "expected_parameter_unit": "cm^-3",
        "expected_axis_names": ["v(V)", "lambda(nm)"],
    },

    # -----------------------------------------------------------------------
    # Interface density — Interface defect density sweep (TWO separate experiments)
    # -----------------------------------------------------------------------
    {
        "sheet": "Interface density",
        "description": (
            "Two independent repeated experiments side by side in one sheet, "
            "separated by a group of empty columns. "
            "Experiment 1 (SNS2/PS interface): 6 JV traces, Jsc summary, 6 QE, 6 Responsivity. "
            "Experiment 2 (PS/Zn3P2 interface): 6 JV traces, Jsc summary. "
            "Row 2, col 0 contains the label 'Sns2/PS'; row 2, col 35 contains 'PS/Zn3P2'. "
            "This is the critical repeated-experiment case."
        ),
        "expected_dims": {
            "n_rows_min": 58,
            "n_rows_max": 65,
            "n_cols_min": 55,
            "n_cols_max": 65,
        },
        "expected_empty_col_boundaries_count_gte": 8,  # multiple separators
        "expected_independent_experiment_count": 2,    # TWO distinct experiments
        "expected_parameter_in_header": "10^",
        "expected_parameter_values_numeric": [1e10, 1e12, 1e14, 1e16, 1e18, 1e20],
        "expected_parameter_unit": "cm^-2",
        "expected_interface_labels": ["Sns2/PS", "PS/Zn3P2"],
        "expected_axis_names": ["v(V)", "lambda(nm)"],
    },

    # -----------------------------------------------------------------------
    # Rsh — Shunt resistance sweep
    # -----------------------------------------------------------------------
    {
        "sheet": "Rsh",
        "description": (
            "8 JV traces for Rsh = 1000–8000 ohm.cm^2. "
            "Column headers are numeric values with unit (e.g. '1000', '2000'). "
            "Scalar Jsc summary to the right (col 12-13). "
            "Two sub-table groups separated by empty columns at col 10-11."
        ),
        "expected_dims": {
            "n_rows_min": 22,
            "n_rows_max": 28,
            "n_cols_min": 24,
            "n_cols_max": 28,
        },
        "expected_empty_col_boundaries_count": 2,
        "expected_sub_table_count": 2,  # JV curves + Jsc scalar summary
        "expected_header_row_index": 0,
        "expected_parameter_values": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        "expected_parameter_unit": "ohm.cm^2",
        "expected_n_jv_traces": 8,
        "expected_axis_name": "v(V)",
    },

    # -----------------------------------------------------------------------
    # Rs — Series resistance sweep
    # -----------------------------------------------------------------------
    {
        "sheet": "Rs",
        "description": (
            "8 JV traces for Rs = 1, 2, 3, 4, 5, 6, 7, 8 ohm/cm^2. "
            "Column headers include the unit string directly: '1 ohm/cm^2', '2 ohm/cm^2', etc. "
            "Scalar Jsc summary to the right (col 12-13). "
            "Two sub-table groups separated by empty columns at col 10-11."
        ),
        "expected_dims": {
            "n_rows_min": 22,
            "n_rows_max": 28,
            "n_cols_min": 24,
            "n_cols_max": 28,
        },
        "expected_empty_col_boundaries_count": 2,
        "expected_sub_table_count": 2,
        "expected_header_row_index": 0,
        "expected_parameter_values": [1, 2, 3, 4, 5, 6, 7, 8],
        "expected_parameter_unit": "ohm/cm^2",
        "expected_n_jv_traces": 8,
        "expected_axis_name": "v(V)",
        # Note: column header encodes unit inline ('1 ohm/cm^2'), not just a number.
        # The extractor must parse the numeric part AND the unit from the column header.
        "note_header_format": "Parameter value and unit are combined in header: '1 ohm/cm^2'",
    },

    # -----------------------------------------------------------------------
    # Optimized — Multi-block single optimized device characterization
    # -----------------------------------------------------------------------
    {
        "sheet": "Optimized",
        "description": (
            "Multiple independent experiments co-located in one sheet. "
            "Block 1 (cols 1-2): JV curve, axis=v(V), target=jtot(mA/cm2), 1 trace. "
            "Block 2 (cols 5-6): QE spectrum, axis=lambda(nm), target=QE(%), 1 trace. "
            "Block 3 (cols 9-10): Responsivity spectrum, axis=lambda(nm), target=R[A/W], 1 trace. "
            "Block 4 (cols 15-17): Detectivity, axis=V(V), target=D*, with Jsc and dark current. "
            "Separated by empty column groups at cols 3, 7, 11-13, 18. "
            "Sheet has 930 rows but most are empty — actual data is in first ~62 rows. "
            "No swept parameters — single optimized device."
        ),
        "expected_dims": {
            "n_rows_total": 930,
            "n_data_rows_approx": 62,
            "n_cols_total": 27,
        },
        "expected_independent_experiment_count": 4,
        "expected_empty_col_boundaries_in_first_30_rows": [3, 7, 11, 12, 13, 18],
        "expected_experiments": [
            {
                "scientific_type": "JV",
                "axis_name": "Voltage",
                "target_name": "Current",
                "n_traces": 1,
                "has_swept_parameters": False,
            },
            {
                "scientific_type": "QE",
                "axis_name": "Wavelength",
                "target_name": "QE",
                "n_traces": 1,
                "has_swept_parameters": False,
            },
            {
                "scientific_type": "RESPONSIVITY",
                "axis_name": "Wavelength",
                "target_name": "Responsivity",
                "n_traces": 1,
                "has_swept_parameters": False,
            },
            {
                "scientific_type": "DETECTIVITY",
                "axis_name": "Voltage",
                "target_name": "Detectivity",
                "n_traces": 1,
                "has_swept_parameters": False,
            },
        ],
    },

    # -----------------------------------------------------------------------
    # Sheet1 — Internal device profile (not a graphable scientific experiment)
    # -----------------------------------------------------------------------
    {
        "sheet": "Sheet1",
        "description": (
            "SCAPS internal device profile output: position-resolved physical quantities "
            "across the device stack. 480 rows x 23 columns. "
            "Columns include: i (layer index), x(um) (position), y, "
            "Ec(eV), Fn(eV), Fp(eV), Ev(eV), n(/cm3), p(/cm3), "
            "rho(defect), net doping, rho, E(V/cm), jn(mA/cm2), jp(mA/cm2), "
            "jn_tunnel, jp_tunnel, jtot(mA/cm2), generation, recombination, "
            "cumulative generation, cumulative recombination L-to-R, R-to-L. "
            "Row 1 is empty (between header and data). "
            "No swept parameters. No experimental parameter sweep. "
            "This is NOT a parametric experiment — it is a positional profile. "
            "The parser must NOT classify this as JV or any standard curve type."
        ),
        "expected_dims": {
            "n_rows": 480,
            "n_cols": 23,
        },
        "expected_empty_col_boundaries_count": 0,  # no empty columns
        "expected_numeric_density_gte": 0.95,
        "expected_axis_candidate": "x(um)",        # position axis, not voltage or wavelength
        "expected_n_experiments": 1,
        "expected_experiment_type": "UNKNOWN",      # must NOT be classified as JV/QE/etc.
        "expected_interpretation_state": "UNRESOLVED",
        "note": (
            "This sheet's x-axis (position in micrometers) does not match any known "
            "scientific experiment x-axis. Physical plausibility checks must not "
            "misclassify it. The raw data must be preserved even though the "
            "interpretation is UNRESOLVED."
        ),
    },
]


# ===========================================================================
# EXPERIMENT INTERPRETATION FIXTURES
# Describe the expected ScientificExperiment objects that parser_v2 must produce.
# These do NOT encode cell coordinates as expected values.
# ===========================================================================

EXPERIMENT_FIXTURES: List[Fixture] = [

    # -----------------------------------------------------------------------
    # ETL — Expected experiments
    # -----------------------------------------------------------------------
    {
        "sheet": "ETL",
        "experiment_index": 0,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "JV",
        "expected_axis": {
            "name": "Voltage",
            "unit": "V",
            "is_monotonic": True,
            "value_range_min": -0.1,
            "value_range_max": 1.2,
            "n_points_min": 20,
        },
        "expected_target_name": "Current",
        "expected_target_unit": "mA/cm^2",
        "expected_n_traces": 10,
        "expected_trace_parameters": [
            {"name": "Thickness", "unit": "nm", "values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
        ],
        "expected_interpretation_state": "ACCEPTED",
        "expected_confidence_gte": 0.75,
    },
    {
        "sheet": "ETL",
        "experiment_index": 1,
        "expected_structure_type": "ScalarSweepTable",
        "expected_scientific_type": "SCALAR_SWEEP",
        "expected_scalar_parameter_name": "Thickness",
        "expected_scalar_parameter_unit": "nm",
        "expected_scalar_parameter_values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "expected_scalar_metric_names_include": ["Jsc"],
        "expected_interpretation_state": "ACCEPTED",
    },
    {
        "sheet": "ETL",
        "experiment_index": 2,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "QE",
        "expected_axis": {
            "name": "Wavelength",
            "unit": "nm",
            "is_monotonic": True,
            "value_range_min": 290,
            "value_range_max": 910,
        },
        "expected_target_name": "QE",
        "expected_target_unit": "%",
        "expected_n_traces": 10,
        "expected_interpretation_state": "ACCEPTED",
    },
    {
        "sheet": "ETL",
        "experiment_index": 3,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "RESPONSIVITY",
        "expected_axis": {
            "name": "Wavelength",
            "unit": "nm",
            "is_monotonic": True,
            "value_range_min": 290,
            "value_range_max": 910,
        },
        "expected_target_name": "Responsivity",
        "expected_target_unit": "A/W",
        "expected_n_traces": 10,
        "expected_interpretation_state": "ACCEPTED",
    },

    # -----------------------------------------------------------------------
    # PVK — Expected experiments
    # -----------------------------------------------------------------------
    {
        "sheet": "PVK",
        "experiment_index": 0,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "JV",
        "expected_n_traces": 10,
        "expected_trace_parameters": [
            {"name": "Thickness", "unit": "nm",
             "values": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
        ],
        "expected_interpretation_state": "ACCEPTED",
    },
    {
        "sheet": "PVK",
        "experiment_index": 1,
        "expected_structure_type": "ScalarSweepTable",
        "expected_scientific_type": "SCALAR_SWEEP",
        "expected_scalar_parameter_name": "Thickness",
        "expected_scalar_parameter_values": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "expected_scalar_metric_names_include": ["Jsc"],
        "expected_interpretation_state": "ACCEPTED",
    },

    # -----------------------------------------------------------------------
    # Temperature — Expected experiments
    # -----------------------------------------------------------------------
    {
        "sheet": "Temperature",
        "experiment_index": 0,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "JV",
        "expected_n_traces": 6,
        "expected_trace_parameters": [
            {"name": "Temperature", "unit": "K", "values": [300, 330, 360, 390, 420, 450]},
        ],
        "expected_interpretation_state": "ACCEPTED",
    },
    {
        "sheet": "Temperature",
        "experiment_index": 1,
        "expected_structure_type": "ScalarSweepTable",
        "expected_scientific_type": "SCALAR_SWEEP",
        "expected_scalar_parameter_name": "Temperature",
        "expected_scalar_parameter_unit": "K",
        "expected_scalar_parameter_values": [300, 330, 360, 390, 420, 450],
        "expected_scalar_metric_names_include": ["Jsc"],
        "expected_interpretation_state": "ACCEPTED",
    },
    {
        "sheet": "Temperature",
        "experiment_index": 2,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "QE",
        "expected_n_traces": 6,
        "expected_trace_parameters": [
            {"name": "Temperature", "unit": "K", "values": [300, 330, 360, 390, 420, 450]},
        ],
        "expected_interpretation_state": "ACCEPTED",
    },
    {
        "sheet": "Temperature",
        "experiment_index": 3,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "RESPONSIVITY",
        "expected_n_traces": 6,
        "expected_interpretation_state": "ACCEPTED",
    },

    # -----------------------------------------------------------------------
    # Defect Density — Expected experiments
    # -----------------------------------------------------------------------
    {
        "sheet": "Defect Density",
        "experiment_index": 0,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "JV",
        "expected_n_traces": 6,
        "expected_trace_parameters": [
            {"name": "Defect_Density", "unit": "cm^-3",
             "values": [1e13, 1e14, 1e15, 1e16, 1e17, 1e18]},
        ],
        # Header encodes density as '10^13', '10^14', etc.
        # The parameter probe must parse '10^N' → 10**N as a float.
        "note_header_format": "Header '10^13' must be parsed to numeric 1e13",
        "expected_interpretation_state": "ACCEPTED",
    },
    {
        "sheet": "Defect Density",
        "experiment_index": 1,
        "expected_structure_type": "ScalarSweepTable",
        "expected_scientific_type": "SCALAR_SWEEP",
        "expected_scalar_parameter_name": "Defect_Density",
        "expected_scalar_parameter_values_str": ["10^13", "10^14", "10^15", "10^16", "10^17", "10^18"],
        # The scalar table col-0 contains strings like '10^13'.
        # The extractor must convert these to floats for the axis.
        "note": "Column 0 of scalar table contains string '10^13' not the number 1e13.",
        "expected_interpretation_state": "ACCEPTED",
    },

    # -----------------------------------------------------------------------
    # Interface density — Two repeated experiments
    # -----------------------------------------------------------------------
    {
        "sheet": "Interface density",
        "experiment_index": 0,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "JV",
        "expected_n_traces": 6,
        "expected_trace_parameters": [
            {"name": "Interface_Density", "unit": "cm^-2",
             "values": [1e10, 1e12, 1e14, 1e16, 1e18, 1e20]},
        ],
        "expected_interface_label": "Sns2/PS",
        "note": (
            "This is experiment 1 of 2 from this sheet. "
            "The interface label 'Sns2/PS' appears in col 0, row 2. "
            "The scanner must identify this as a separate experiment from "
            "the PS/Zn3P2 block to the right."
        ),
        "expected_interpretation_state": "ACCEPTED",
    },
    {
        "sheet": "Interface density",
        "experiment_index": 1,
        "expected_structure_type": "ScalarSweepTable",
        "expected_scientific_type": "SCALAR_SWEEP",
        "expected_scalar_parameter_name": "Interface_Density",
        "expected_scalar_parameter_values_numeric": [1e10, 1e12, 1e14, 1e16, 1e18, 1e20],
        "expected_interface_label": "Sns2/PS",
        "expected_interpretation_state": "ACCEPTED",
    },
    {
        "sheet": "Interface density",
        "experiment_index": 4,  # experiments 0-3 are from Sns2/PS block; 4-5 from PS/Zn3P2
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "JV",
        "expected_n_traces": 6,
        "expected_trace_parameters": [
            {"name": "Interface_Density", "unit": "cm^-2",
             "values": [1e10, 1e12, 1e14, 1e16, 1e18, 1e20]},
        ],
        "expected_interface_label": "PS/Zn3P2",
        "note": (
            "This is the SECOND independent JV experiment from Interface density. "
            "Same parameter sweep (interface defect density), different interface (PS/Zn3P2). "
            "The parser must produce TWO separate JV experiments from this sheet, "
            "not merge them into one. The interface label distinguishes them."
        ),
        "expected_interpretation_state": "ACCEPTED",
    },

    # -----------------------------------------------------------------------
    # Rsh — Shunt resistance JV sweep
    # -----------------------------------------------------------------------
    {
        "sheet": "Rsh",
        "experiment_index": 0,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "JV",
        "expected_axis": {
            "name": "Voltage",
            "unit": "V",
            "is_monotonic": True,
            "value_range_min": -0.5,
            "value_range_max": 1.3,
        },
        "expected_target_name": "Current",
        "expected_target_unit": "mA/cm^2",
        "expected_n_traces": 8,
        "expected_trace_parameters": [
            {"name": "Shunt_Resistance", "unit": "ohm.cm^2",
             "values": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]},
        ],
        # Column headers are just the numbers: '1000', '2000', etc. — no unit embedded.
        # Unit must be inferred from the sheet context, not the header text.
        "note_unit_inference": "Rsh parameter unit inferred from sheet context, not column header.",
        "expected_interpretation_state": "ACCEPTED",
        "expected_confidence_gte": 0.75,
    },
    {
        "sheet": "Rsh",
        "experiment_index": 1,
        "expected_structure_type": "ScalarSweepTable",
        "expected_scientific_type": "SCALAR_SWEEP",
        "expected_scalar_parameter_name": "Shunt_Resistance",
        "expected_scalar_parameter_unit": "ohm.cm^2",
        "expected_scalar_parameter_values": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        "expected_scalar_metric_names_include": ["Jsc"],
        "expected_interpretation_state": "ACCEPTED",
    },

    # -----------------------------------------------------------------------
    # Rs — Series resistance JV sweep
    # -----------------------------------------------------------------------
    {
        "sheet": "Rs",
        "experiment_index": 0,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "JV",
        "expected_axis": {
            "name": "Voltage",
            "unit": "V",
            "is_monotonic": True,
        },
        "expected_target_name": "Current",
        "expected_target_unit": "mA/cm^2",
        "expected_n_traces": 8,
        "expected_trace_parameters": [
            {"name": "Series_Resistance", "unit": "ohm/cm^2",
             "values": [1, 2, 3, 4, 5, 6, 7, 8]},
        ],
        # Column headers are '1 ohm/cm^2', '2 ohm/cm^2', etc.
        # Both the numeric value AND the unit are embedded in the header text.
        "note_header_format": (
            "Parameter value and unit are combined in the column header: "
            "'1 ohm/cm^2'. The label_probe must parse BOTH the numeric part (1) "
            "AND the unit ('ohm/cm^2') from the same token."
        ),
        "expected_interpretation_state": "ACCEPTED",
        "expected_confidence_gte": 0.75,
    },
    {
        "sheet": "Rs",
        "experiment_index": 1,
        "expected_structure_type": "ScalarSweepTable",
        "expected_scientific_type": "SCALAR_SWEEP",
        "expected_scalar_parameter_name": "Series_Resistance",
        "expected_scalar_parameter_values": [1, 2, 3, 4, 5, 6, 7, 8],
        "expected_scalar_metric_names_include": ["Jsc"],
        "expected_interpretation_state": "ACCEPTED",
    },

    # -----------------------------------------------------------------------
    # Optimized — 4 independent single-device experiments
    # -----------------------------------------------------------------------
    {
        "sheet": "Optimized",
        "experiment_index": 0,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "JV",
        "expected_axis": {
            "name": "Voltage",
            "unit": "V",
            "is_monotonic": True,
            "value_range_min": -0.1,
            "value_range_max": 1.4,
        },
        "expected_target_name": "Current",
        "expected_target_unit": "mA/cm^2",
        "expected_n_traces": 1,
        "expected_trace_parameters": [],   # no swept parameter — single device
        "expected_interpretation_state": "ACCEPTED",
    },
    {
        "sheet": "Optimized",
        "experiment_index": 1,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "QE",
        "expected_axis": {
            "name": "Wavelength",
            "unit": "nm",
            "is_monotonic": True,
            "value_range_min": 290,
            "value_range_max": 910,
        },
        "expected_target_name": "QE",
        "expected_target_unit": "%",
        "expected_n_traces": 1,
        "expected_trace_parameters": [],
        "expected_interpretation_state": "ACCEPTED",
    },
    {
        "sheet": "Optimized",
        "experiment_index": 2,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "RESPONSIVITY",
        "expected_axis": {
            "name": "Wavelength",
            "unit": "nm",
            "is_monotonic": True,
        },
        "expected_target_name": "Responsivity",
        "expected_target_unit": "A/W",
        "expected_n_traces": 1,
        "expected_trace_parameters": [],
        "expected_interpretation_state": "ACCEPTED",
    },
    {
        "sheet": "Optimized",
        "experiment_index": 3,
        "expected_structure_type": "NumericCurveGrid",
        "expected_scientific_type": "DETECTIVITY",
        "expected_axis": {
            "name": "Voltage",
            "unit": "V",
            "is_monotonic": False,  # Detectivity axis may be decreasing
        },
        "expected_target_name": "Detectivity",
        "expected_target_unit": "Jones",
        "expected_n_traces": 1,
        "expected_trace_parameters": [],
        "expected_interpretation_state": "ACCEPTED",
    },

    # -----------------------------------------------------------------------
    # Sheet1 — Device profile, must NOT be classified as an experiment
    # -----------------------------------------------------------------------
    {
        "sheet": "Sheet1",
        "experiment_index": 0,
        "expected_structure_type": "Unknown",
        "expected_scientific_type": None,
        "expected_interpretation_state": "UNRESOLVED",
        "expected_confidence_lt": 0.50,
        "note": (
            "Sheet1 contains position-resolved internal SCAPS device data. "
            "x-axis is device position x(um), not Voltage or Wavelength. "
            "23 columns include band energies, carrier densities, current components. "
            "The parser must not misclassify this as a JV curve even though "
            "jtot(mA/cm2) is present as a column. "
            "The raw data must be fully preserved in RawRegionData."
        ),
        "expected_raw_preserved": True,
        "expected_n_columns": 23,
    },
]


# ===========================================================================
# INVARIANT FIXTURES
# Properties that must hold across ALL experiments from ALL sheets.
# ===========================================================================

INVARIANT_FIXTURES: List[Fixture] = [
    {
        "name": "raw_always_preserved",
        "description": (
            "Every ScientificExperiment, regardless of interpretation_state, "
            "must have a non-None raw field with a complete cell grid."
        ),
        "assertion": "for every experiment: experiment.raw is not None",
    },
    {
        "name": "provenance_complete",
        "description": (
            "Every DataTrace must have len(x_refs) == len(x) and len(y_refs) == len(y). "
            "No extracted value may lack a source CellRef."
        ),
        "assertion": (
            "for every experiment, for every trace: "
            "len(trace.x_refs) == len(trace.x) and len(trace.y_refs) == len(trace.y)"
        ),
    },
    {
        "name": "hypothesis_ranking_never_discarded",
        "description": (
            "all_hypothesis_scores must contain at least 2 entries for every region "
            "that had more than 1 hypothesis generated. The full ranking is always preserved."
        ),
        "assertion": (
            "for every experiment: len(experiment.all_hypothesis_scores) >= 1"
        ),
    },
    {
        "name": "physical_violation_does_not_discard",
        "description": (
            "If is_physically_plausible is False, the traces and raw must still be populated. "
            "Physical violations never cause data loss."
        ),
        "assertion": (
            "for every experiment where is_physically_plausible is False: "
            "experiment.raw is not None and "
            "(len(experiment.traces) > 0 or len(experiment.scalar_rows) > 0)"
        ),
    },
    {
        "name": "unresolved_not_empty",
        "description": (
            "UNRESOLVED experiments must still have their raw data populated. "
            "The fact that we could not classify it does not mean we discard the cells."
        ),
        "assertion": (
            "for every experiment where interpretation_state == 'UNRESOLVED': "
            "experiment.raw is not None"
        ),
    },
    {
        "name": "no_cross_axis_mixing_in_one_experiment",
        "description": (
            "No single NumericCurveGrid experiment may have traces from "
            "different x-axis types (e.g. some traces with Voltage axis, "
            "some with Wavelength axis). Such blocks must produce separate experiments."
        ),
        "assertion": (
            "for every experiment of structure_type NumericCurveGrid: "
            "all traces share the same axis name and unit"
        ),
    },
    {
        "name": "interface_density_two_experiments",
        "description": (
            "The 'Interface density' sheet must produce exactly 2 independent "
            "JV experiments (one per interface: Sns2/PS and PS/Zn3P2). "
            "They must not be merged into a single experiment."
        ),
        "assertion": (
            "len([e for e in experiments('Interface density') "
            "     if e.scientific_type == 'JV']) == 2"
        ),
    },
    {
        "name": "sheet1_not_jv",
        "description": (
            "Sheet1 must never be classified as JV or any spectral experiment. "
            "Presence of jtot(mA/cm2) column is not sufficient evidence on its own."
        ),
        "assertion": (
            "for every experiment from 'Sheet1': "
            "scientific_type not in ['JV', 'QE', 'RESPONSIVITY', 'DETECTIVITY']"
        ),
    },
]


# ===========================================================================
# PARAMETER PARSING EDGE CASES
# Specific header formats the parameter probe must handle correctly.
# ===========================================================================

PARAMETER_PARSING_FIXTURES: List[Fixture] = [
    {
        "header_text": "t=100[nm]",
        "expected_parameter_name": "Thickness",
        "expected_parameter_value": 100.0,
        "expected_parameter_unit": "nm",
    },
    {
        "header_text": "T=300",
        "expected_parameter_name": "Temperature",
        "expected_parameter_value": 300.0,
        "expected_parameter_unit": "K",
    },
    {
        "header_text": "10^13",
        "expected_parameter_name": None,  # name comes from sheet context, not this token
        "expected_parameter_value": 1e13,
        "expected_parameter_unit": None,  # unit comes from sheet context
        "note": "'10^N' strings encode a numeric value only; name and unit from context.",
    },
    {
        "header_text": "1 ohm/cm^2",
        "expected_parameter_value": 1.0,
        "expected_parameter_unit": "ohm/cm^2",
        "note": "Value and unit combined in one token; both must be extracted.",
    },
    {
        "header_text": "1000",
        "expected_parameter_value": 1000.0,
        "expected_parameter_unit": None,  # unit from sheet context
        "note": "Plain integer header; unit must come from sheet-level context.",
    },
    {
        "header_text": "10^20",
        "expected_parameter_value": 1e20,
        "expected_parameter_unit": None,
        "note": "Large exponent encoding for interface/defect density.",
    },
]


if __name__ == "__main__":
    print(f"Structural fixtures: {len(STRUCTURAL_FIXTURES)}")
    print(f"Experiment fixtures: {len(EXPERIMENT_FIXTURES)}")
    print(f"Invariant fixtures:  {len(INVARIANT_FIXTURES)}")
    print(f"Parameter parsing fixtures: {len(PARAMETER_PARSING_FIXTURES)}")
    print()
    sheets_covered = {f["sheet"] for f in STRUCTURAL_FIXTURES}
    print(f"Sheets covered by structural fixtures: {sorted(sheets_covered)}")
    exp_sheets = {f["sheet"] for f in EXPERIMENT_FIXTURES}
    print(f"Sheets covered by experiment fixtures: {sorted(exp_sheets)}")
