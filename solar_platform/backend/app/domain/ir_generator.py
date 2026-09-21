from __future__ import annotations

import re

from typing import Dict, List, Optional, Tuple

from app.domain.audit import (
    BoundingBox,
    DataBlockAudit,
    DecisionTraceEntry,
    BlockCategory,
)

from app.domain.document import DocumentSheet

from app.domain.ir import (
    DependentVariable,
    IndependentAxis,
    IntermediateRepresentationV2,
    SweptParameter,
)

from app.services.preprocessor import extract_numeric_value


class IRGeneratorEngine:

    def generate_ir_for_block(
        self,
        sheet: DocumentSheet,
        block: DataBlockAudit,
        source_filename: Optional[str] = None,
    ) -> IntermediateRepresentationV2:
        """Generates IntermediateRepresentationV2 with structural curve purity,
        trace identity, and parameter banner extraction.
        """

        bbox = block.bounding_box

        # ------------------------------------------------------------
        # 2-D PARAMETER SWEEP DETECTION
        #
        # Some summary tables are a flattened 2-D sweep across two
        # named parameter columns ("bp 1" / "bp 2"), not a single 1-D
        # sweep. This used to be gated to one exact workbook+sheet
        # name ("Ca3AsBr3.xlsx" -> "T+DD"), which meant any other
        # workbook with the same genuine bp1/bp2 structure (e.g.
        # Ca3AsI3.xlsx -> T+DD) fell through to the 1-D summary path
        # instead, treating "bp 2" as if it were a dependent metric
        # and producing a sawtooth "bp 2 vs Parameter" chart.
        #
        # The gate is now structural: it fires whenever a block's own
        # headers contain both a "bp 1" and a "bp 2" column, regardless
        # of workbook/sheet name. Any block without both columns falls
        # through to the existing 1-D summary logic exactly as before.
        # ------------------------------------------------------------

        rows = list(range(bbox.start_row, bbox.end_row + 1))
        cols = list(range(bbox.start_col, bbox.end_col + 1))

        if len(rows) < 2 or len(cols) < 1:
            return IntermediateRepresentationV2()

        # ============================================================
        # Step 1: Detect Real Table Header Row
        # ============================================================

        header_row_idx = bbox.start_row
        header_evidence = "Row 0 assumed as default top header."

        for r in range(
            bbox.start_row,
            min(
                bbox.start_row + 40,
                max(bbox.start_row + 1, bbox.end_row - 4),
            ),
        ):
            c_curr = sheet.get_cell(r, cols[0])

            if not c_curr or c_curr.is_numeric():
                continue

            n_count = sum(
                1
                for off in range(1, 7)
                if (
                    r + off <= bbox.end_row
                    and sheet.get_cell(r + off, cols[0])
                    and sheet.get_cell(r + off, cols[0]).is_numeric()
                )
            )

            if n_count >= 3:
                header_row_idx = r
                header_evidence = (
                    f"Row {r} selected as curve table header."
                )
                break

        block.decision_traces.append(
            DecisionTraceEntry(
                decision_type="HEADER_LOCATION",
                decision=f"Header assigned to Row {header_row_idx}",
                evidence=header_evidence,
                confidence=1.0,
            )
        )

        header_texts = [
            (
                sheet.get_cell(header_row_idx, c).formatted_value
                if sheet.get_cell(header_row_idx, c)
                else f"Col_{c}"
            )
            for c in cols
        ]

        # ============================================================
        # Step 2: Detect Parameter-Sweep Summary Tables
        # ============================================================

        sample_h_lower = [h.lower() for h in header_texts]

        summary_metric_keywords = [
            "voc",
            "jsc",
            "ff",
            "eta",
            "pce",
            "v_mpp",
            "j_mpp",
        ]

        matched_summary_keys = [
            h
            for h in sample_h_lower
            if any(k in h for k in summary_metric_keywords)
        ]

        if len(matched_summary_keys) >= 2:

            block.category = BlockCategory.SUMMARY
            data_start_row = header_row_idx + 1

            # ----------------------------------------------------------
            # Determine which column actually holds the swept-parameter
            # values. cols[0] is the conventional position, but some
            # sheets stack one or more non-data leading columns there
            # first: a text title/label (e.g. "ETL/PS", no numeric rows
            # at all) and/or a row/run index column (header "i"/"run"/
            # "index"/"no"/"#", numeric but not a real experimental
            # variable). Skip forward past any such columns to find the
            # first genuine numeric sweep column instead of assuming
            # position 0 is always right.
            #
            # For every sheet where cols[0] is already the real sweep
            # column (the normal case, e.g. Ca3AsBr3.xlsx -> T+DD where
            # cols[0] is "bp 1"), this loop terminates immediately and
            # axis_col == cols[0], so behavior is unchanged.
            # ----------------------------------------------------------
            _index_header_terms = {"i", "run", "index", "no", "no.", "#"}
            _leading_title_text = ""
            axis_col_pos = 0
            while axis_col_pos < len(cols) - 1:
                _probe_col = cols[axis_col_pos]
                _probe_header = sample_h_lower[axis_col_pos].strip()
                _probe_has_numeric = any(
                    sheet.get_cell(r, _probe_col)
                    and sheet.get_cell(r, _probe_col).is_numeric()
                    for r in range(
                        data_start_row,
                        min(data_start_row + 5, bbox.end_row + 1),
                    )
                )
                if not _probe_has_numeric:
                    # A genuine text title/label column (not a run-index
                    # column) -- remember it. When a sheet has two
                    # side-by-side sweep tables whose real axis columns
                    # happen to share the exact same raw header (e.g.
                    # both literally called "bp 1"), this title is the
                    # only thing that distinguishes them (e.g. "ETL/PS"
                    # vs "PS/HTL"). Without it they'd resolve to the
                    # same axis name and get incorrectly merged into one
                    # experiment/trace by the grouper downstream.
                    _title_candidate = header_texts[
                        axis_col_pos
                    ].strip()
                    if _title_candidate:
                        _leading_title_text = _title_candidate
                    axis_col_pos += 1
                    continue
                if _probe_header in _index_header_terms:
                    axis_col_pos += 1
                    continue
                break
            axis_col = cols[axis_col_pos]

            data_rows = [
                r
                for r in range(
                    data_start_row,
                    bbox.end_row + 1,
                )
                if (
                    sheet.get_cell(r, axis_col)
                    and sheet.get_cell(r, axis_col).is_numeric()
                )
            ]

            # --------------------------------------------------------
            # Existing summary statistics logic.
            #
            # Skip the axis column itself (and any leading title/index
            # columns skipped over above, idx <= axis_col_pos) -- these
            # are the independent/swept variable, not a performance
            # metric, and including them here previously caused bogus
            # "i" / "bp 1" cards to show up as if they were metrics
            # (e.g. on the IDD sheet, whose axis columns leak straight
            # through as raw header text). Every genuine metric column
            # (idx > axis_col_pos) is completely unaffected.
            # --------------------------------------------------------

            metrics_dict: Dict[str, Dict[str, float]] = {}

            for idx, col in enumerate(cols):

                if idx <= axis_col_pos:
                    continue

                col_name = header_texts[idx]

                col_vals = [
                    sheet.get_cell(r, col).to_float()
                    for r in data_rows
                    if (
                        sheet.get_cell(r, col)
                        and sheet.get_cell(r, col).to_float() is not None
                    )
                ]

                if col_vals:
                    metrics_dict[col_name] = {
                        "mean": float(sum(col_vals) / len(col_vals)),
                        "max": float(max(col_vals)),
                        "min": float(min(col_vals)),
                        "latest": float(col_vals[-1]),
                    }

            # ========================================================
            # 2-D PARAMETER SWEEP HANDLING (bp 1 / bp 2)
            #
            # Structural gate: run this whenever the block's headers
            # actually contain both a bp1 and bp2 column. No longer
            # tied to a specific workbook/sheet name (see note above).
            # ========================================================

            bp1_idx: Optional[int] = None
            bp2_idx: Optional[int] = None

            for idx, header in enumerate(header_texts):

                normalized_header = re.sub(
                    r"\s+",
                    "",
                    header.lower().strip(),
                )

                if normalized_header in {"bp1", "bp_1"}:
                    bp1_idx = idx

                elif normalized_header in {"bp2", "bp_2"}:
                    bp2_idx = idx

            # ----------------------------------------------------
            # A genuine 2-D parameter sweep requires both columns.
            # ----------------------------------------------------

            if bp1_idx is not None and bp2_idx is not None:

                bp1_col = cols[bp1_idx]
                bp2_col = cols[bp2_idx]

                bp1_unique: List[float] = []
                bp2_unique: List[float] = []

                for r in data_rows:

                    c1 = sheet.get_cell(r, bp1_col)
                    c2 = sheet.get_cell(r, bp2_col)

                    v1 = c1.to_float() if c1 else None
                    v2 = c2.to_float() if c2 else None

                    if (
                        v1 is not None
                        and v1 not in bp1_unique
                    ):
                        bp1_unique.append(float(v1))

                    if (
                        v2 is not None
                        and v2 not in bp2_unique
                    ):
                        bp2_unique.append(float(v2))

                bp1_unique.sort()
                bp2_unique.sort()

                # ----------------------------------------------------
                # "T+DD" sheets in this workbook family are Thickness
                # + Defect Density sweeps (bp 1 = Thickness in um,
                # bp 2 = Defect Density in cm^-3) -- same convention
                # already used for this family's "QE (T+DD)" sheet.
                # This naming stays local to this structural 2-D-sweep
                # branch and only applies when both bp1/bp2 columns
                # were actually found.
                # ----------------------------------------------------
                axis_name, axis_unit = "Thickness", "um"
                param2_name, param2_unit = "Defect_Density", "cm^-3"

                param_axis = IndependentAxis(
                    axis_id="axis_param_01",
                    name=axis_name,
                    unit=axis_unit,
                    is_regular_sampling=False,
                    values=bp1_unique,
                )

                # NOTE: only bp2 (Defect_Density) goes in "parameters"
                # here, not bp1/Thickness. Thickness is already the
                # axis (set correctly per-row via x_name/x_val below in
                # the dataframe builder); listing it again as a
                # "parameter" makes the dataframe builder re-index it
                # by trace/column position instead of row position,
                # silently overwriting the correct axis value with the
                # wrong one and collapsing every trace's x to a single
                # repeated number.
                tdd_parameters = [
                    SweptParameter(
                        parameter_id="param_bp2",
                        name=param2_name,
                        unit=param2_unit,
                        values=bp2_unique,
                    ),
                ]

                row_by_pair: Dict[
                    Tuple[float, float],
                    int,
                ] = {}

                for r in data_rows:

                    c1 = sheet.get_cell(r, bp1_col)
                    c2 = sheet.get_cell(r, bp2_col)

                    v1 = c1.to_float() if c1 else None
                    v2 = c2.to_float() if c2 else None

                    if v1 is None or v2 is None:
                        continue

                    row_by_pair[
                        (float(v1), float(v2))
                    ] = r

                summary_targets: List[
                    DependentVariable
                ] = []

                for c_idx, col in enumerate(cols):

                    if c_idx in {bp1_idx, bp2_idx}:
                        continue

                    h_name = header_texts[c_idx]

                    has_numeric = any(
                        sheet.get_cell(r, col)
                        and sheet.get_cell(r, col).to_float()
                        is not None
                        for r in data_rows
                    )

                    if not has_numeric:
                        continue

                    t_name, t_unit = (
                        self._identify_metric_target(
                            h_name
                        )
                    )

                    y_matrix: List[List[float]] = []

                    for bp1 in bp1_unique:

                        row_values: List[float] = []

                        for bp2 in bp2_unique:

                            source_row = row_by_pair.get(
                                (bp1, bp2)
                            )

                            if source_row is None:
                                row_values.append(float("nan"))
                                continue

                            cell = sheet.get_cell(
                                source_row,
                                col,
                            )

                            value = (
                                cell.to_float()
                                if cell
                                else None
                            )

                            row_values.append(
                                float(value)
                                if value is not None
                                else float("nan")
                            )

                        y_matrix.append(row_values)

                    trace_labels = [
                        f"{param2_name} = {v:.1e} {param2_unit}"
                        for v in bp2_unique
                    ]

                    summary_targets.append(
                        DependentVariable(
                            target_id=f"target_sum_{c_idx}",
                            name=t_name,
                            unit=t_unit,
                            is_sparse=False,
                            data_shape=[
                                len(bp1_unique),
                                len(bp2_unique),
                            ],
                            dense_matrix=y_matrix,
                            column_labels=trace_labels,
                        )
                    )

                return IntermediateRepresentationV2(
                    ir_version="v2.0",
                    axes=[param_axis],
                    parameters=tdd_parameters,
                    targets=summary_targets,
                    summary_metrics=metrics_dict,
                    metadata_attributes={
                        "sheet_name": sheet.sheet_name,
                        "header_row": header_row_idx,
                        "is_summary_table": True,
                        "tdd_2d_sweep_handling": True,
                        "tdd_parameter_1": axis_name,
                        "tdd_parameter_2": param2_name,
                        "tdd_axis_mode": (
                            "bp1_vs_metric_grouped_by_bp2"
                        ),
                        "tdd_bp1_count": len(bp1_unique),
                        "tdd_bp2_count": len(bp2_unique),
                    },
                )


            # ========================================================
            # EXISTING SUMMARY LOGIC
            #
            # Retained for every other workbook/sheet.
            # ========================================================

            param_x_vals = [
                sheet.get_cell(r, axis_col).to_float()
                for r in data_rows
                if (
                    sheet.get_cell(r, axis_col)
                    and sheet.get_cell(r, axis_col).to_float()
                    is not None
                )
            ]

            p_axis_name, p_axis_unit = (
                self._identify_sheet_param_axis(
                    sheet.sheet_name,
                    header_texts[axis_col_pos],
                )
            )

            # If a leading title column was found (see skip-loop above),
            # fold it into the axis name so two side-by-side sweep
            # tables that happen to share an identical raw axis header
            # (e.g. both literally "bp 1") don't collapse into the same
            # experiment. Sheets with no such title (axis_col_pos == 0,
            # the normal case for NA/Rs/Rsh/Temperature/T+DD/etc.) are
            # completely unaffected.
            if _leading_title_text:
                _safe_title = (
                    _leading_title_text.replace("/", "_")
                    .replace(" ", "_")
                    .strip("_")
                )
                if _safe_title:
                    p_axis_name = f"{_safe_title}_{p_axis_name}"

            param_axis = IndependentAxis(
                axis_id="axis_param_01",
                name=p_axis_name,
                unit=p_axis_unit,
                is_regular_sampling=False,
                values=param_x_vals,
            )

            summary_targets: List[
                DependentVariable
            ] = []

            for c_idx in range(axis_col_pos + 1, len(cols)):

                col = cols[c_idx]
                h_name = header_texts[c_idx]

                t_name, t_unit = (
                    self._identify_metric_target(
                        h_name
                    )
                )

                y_vals = [
                    sheet.get_cell(r, col).to_float()
                    for r in data_rows
                ]

                y_matrix = [
                    [
                        v
                        if v is not None
                        else float("nan")
                    ]
                    for v in y_vals
                ]

                summary_targets.append(
                    DependentVariable(
                        target_id=f"target_sum_{c_idx}",
                        name=t_name,
                        unit=t_unit,
                        is_sparse=False,
                        data_shape=[
                            len(param_x_vals),
                            1,
                        ],
                        dense_matrix=y_matrix,
                        column_labels=[t_name],
                    )
                )

            return IntermediateRepresentationV2(
                ir_version="v2.0",
                axes=[param_axis],
                parameters=[],
                targets=summary_targets,
                summary_metrics=metrics_dict,
                metadata_attributes={
                    "sheet_name": sheet.sheet_name,
                    "header_row": header_row_idx,
                    "is_summary_table": True,
                },
            )

        # ============================================================
        # Step 3: Curve Block Axis & Sub-Table Detection
        #
        # UNCHANGED
        # ============================================================

        x_col_idx = cols[0]

        x_name = "Voltage"
        x_unit = "V"

        first_row_numeric = sum(
            1
            for r in range(
                header_row_idx + 1,
                min(
                    header_row_idx + 5,
                    bbox.end_row + 1,
                ),
            )
            if (
                sheet.get_cell(r, cols[0])
                and sheet.get_cell(r, cols[0]).is_numeric()
            )
        )

        if first_row_numeric == 0 and len(cols) > 1:
            x_col_idx = cols[1]

        x_detected = False

        for idx, col in enumerate(cols):

            h_text = header_texts[idx].lower()

            if any(
                k in h_text
                for k in [
                    "lambda",
                    "wavelength",
                    "l(nm)",
                    "w(nm)",
                ]
            ):
                x_col_idx = col
                x_name = "Wavelength"
                x_unit = "nm"
                x_detected = True
                break

            elif any(
                k in h_text
                for k in [
                    "v(v)",
                    "voltage",
                ]
            ):
                x_col_idx = col
                x_name = "Voltage"
                x_unit = "V"
                x_detected = True
                break

        if not x_detected:

            h_raw = header_texts[
                cols.index(x_col_idx)
            ]

            p_name, p_unit = (
                self._identify_param_name(
                    h_raw
                )
            )

            if p_name:

                x_name = p_name

                x_unit = (
                    "nm"
                    if "nm" in h_raw.lower()
                    else p_unit
                )

        # ============================================================
        # Step 4: Structural Curve Row Extraction
        # ============================================================

        data_start_row = header_row_idx + 1

        valid_data_rows: List[int] = []

        for r in range(
            data_start_row,
            bbox.end_row + 1,
        ):

            x_cell = sheet.get_cell(
                r,
                x_col_idx,
            )

            x_val = (
                x_cell.to_float()
                if x_cell
                else None
            )

            if x_val is None:

                if not valid_data_rows:
                    continue

                else:
                    break

            row_text = " ".join(
                [
                    sheet.get_cell(
                        r,
                        c,
                    ).formatted_value
                    for c in cols
                    if (
                        sheet.get_cell(
                            r,
                            c,
                        )
                        and sheet.get_cell(
                            r,
                            c,
                        ).formatted_value
                    )
                ]
            ).lower()

            if any(
                desc_kw in row_text
                for desc_kw in [
                    "dark current",
                    "recombination",
                    "voc",
                    "jsc",
                    "pce",
                    "eta",
                ]
            ):
                break

            valid_data_rows.append(r)

        x_values = [
            sheet.get_cell(
                r,
                x_col_idx,
            ).to_float()
            for r in valid_data_rows
            if (
                sheet.get_cell(
                    r,
                    x_col_idx,
                )
                and sheet.get_cell(
                    r,
                    x_col_idx,
                ).to_float()
                is not None
            )
        ]

        if (
            x_values
            and max(x_values) > 100.0
            and x_name == "Voltage"
        ):
            x_name = "Wavelength"
            x_unit = "nm"

        axis = IndependentAxis(
            axis_id="axis_01",
            name=x_name,
            unit=x_unit,
            is_regular_sampling=True,
            values=x_values,
        )

        # ============================================================
        # Step 5: Parameter Banner & Header Extraction
        # ============================================================

        swept_parameters = (
            self._extract_parameters(
                sheet=sheet,
                block=block,
                header_row_idx=header_row_idx,
                cols=cols,
                x_col_idx=x_col_idx,
                header_texts=header_texts,
            )
        )

        # ============================================================
        # Step 6: Dependent Variable Target Extraction
        # ============================================================

        targets = (
            self._extract_dependent_variables(
                sheet=sheet,
                bbox=bbox,
                valid_data_rows=valid_data_rows,
                cols=cols,
                x_col_idx=x_col_idx,
                header_row_idx=header_row_idx,
                header_texts=header_texts,
                x_name=x_name,
                x_values_len=len(x_values),
            )
        )

        return IntermediateRepresentationV2(
            ir_version="v2.0",
            axes=[axis],
            parameters=swept_parameters,
            targets=targets,
            metadata_attributes={
                "sheet_name": sheet.sheet_name,
                "header_row": header_row_idx,
                "valid_rows_count": len(
                    valid_data_rows
                ),
            },
        )

    # ================================================================
    # PARAMETER AXIS IDENTIFICATION
    # ================================================================

    def _identify_sheet_param_axis(
        self,
        sheet_name: str,
        col0_header: str,
    ) -> Tuple[str, str]:

        s_lower = sheet_name.strip().lower()

        # ------------------------------------------------------------
        # STRICT TEMPERATURE FIX
        #
        # ONLY affects a sheet whose name is exactly "Temperature".
        #
        # This does NOT alter:
        #   - T+DD
        #   - Thickness_PVK
        #   - Rs
        #   - Rsh
        #   - Acceptor Density
        #   - Defect Density
        #   - any other sheet
        # ------------------------------------------------------------
        if s_lower == "temperature":
            return "Temperature", "K"

        # ------------------------------------------------------------
        # Existing logic below remains unchanged.
        # ------------------------------------------------------------

        if "t (" in s_lower or "t(k)" in s_lower:
            return "Temperature", "K"

        elif "na" in s_lower:
            return "Acceptor_Density", "cm^-3"

        elif "rs" in s_lower and "rsh" not in s_lower:
            return "Series_Resistance", "ohm.cm^2"

        elif "rsh" in s_lower:
            return "Shunt_Resistance", "ohm.cm^2"

        c_clean = col0_header.strip()

        if (
            c_clean
            and c_clean.lower() != "bp 1"
        ):
            return c_clean, "meta"

        return "Parameter", "unit"

    # ================================================================
    # METRIC TARGET IDENTIFICATION
    # ================================================================

    def _identify_metric_target(
        self,
        col_header: str,
    ) -> Tuple[str, str]:

        h_lower = col_header.lower().strip()

        if (
            "eta" in h_lower
            or "pce" in h_lower
        ):
            return "PCE", "%"

        elif "voc" in h_lower:
            return "Voc", "V"

        elif "jsc" in h_lower:
            return "Jsc", "mA/cm^2"

        elif "ff" in h_lower:
            return "FF", "%"

        elif "v_mpp" in h_lower:
            return "V_MPP", "V"

        elif "j_mpp" in h_lower:
            return "J_MPP", "mA/cm^2"

        elif "qe" in h_lower:
            return "QE", "%"

        elif (
            "r (" in h_lower
            or "r(" in h_lower
        ):
            return "Responsivity", "A/W"

        return (
            col_header.strip()
            or "Metric",
            "unit",
        )

    # ================================================================
    # PARAMETER EXTRACTION
    # ================================================================

    def _extract_parameters(
        self,
        sheet: DocumentSheet,
        block: DataBlockAudit,
        header_row_idx: int,
        cols: List[int],
        x_col_idx: int,
        header_texts: List[str],
    ) -> List[SweptParameter]:

        extracted_params: Dict[
            str,
            Tuple[str, List[float]],
        ] = {}

        bbox = block.bounding_box

        # ------------------------------------------------------------
        # Existing special handling
        # ------------------------------------------------------------

        if sheet.sheet_name in [
            "QE (T+DD)",
            "QE (IDD)",
        ]:

            block_num = 1

            if "_" in block.block_id:

                try:
                    block_num = int(
                        block.block_id.rsplit(
                            "_",
                            1,
                        )[-1]
                    )

                except ValueError:
                    block_num = 1

            if sheet.sheet_name == "QE (T+DD)":

                t_idx = (
                    block_num - 1
                ) // 8

                dd_idx = (
                    block_num - 1
                ) % 8

                thick_val = round(
                    0.3 + t_idx * 0.1,
                    4,
                )

                dd_val = float(
                    10 ** (
                        11 + dd_idx
                    )
                )

                return [
                    SweptParameter(
                        parameter_id="param_1",
                        name="Temperature",
                        unit="K",
                        values=[300.0],
                    ),
                    SweptParameter(
                        parameter_id="param_2",
                        name="Thickness",
                        unit="um",
                        values=[thick_val],
                    ),
                    SweptParameter(
                        parameter_id="param_3",
                        name="Defect_Density",
                        unit="cm^-3",
                        values=[dd_val],
                    ),
                ]

            elif sheet.sheet_name == "QE (IDD)":

                dd_val = float(
                    10 ** (
                        8 + (
                            block_num - 1
                        )
                    )
                )

                return [
                    SweptParameter(
                        parameter_id="param_1",
                        name="Temperature",
                        unit="K",
                        values=[300.0],
                    ),
                    SweptParameter(
                        parameter_id="param_2",
                        name="Defect_Density",
                        unit="cm^-3",
                        values=[dd_val],
                    ),
                ]

        scan_start_row = max(
            0,
            bbox.start_row - 10,
        )

        # ------------------------------------------------------------
        # Scan parameter metadata above the table.
        # ------------------------------------------------------------

        for r in range(
            scan_start_row,
            header_row_idx,
        ):

            search_cols = [0] + [
                c
                for c in cols
                if c != 0
            ]

            for key_col in search_cols:

                cell_key = sheet.get_cell(
                    r,
                    key_col,
                )

                if (
                    not cell_key
                    or not cell_key.formatted_value
                ):
                    continue

                txt_key = (
                    cell_key.formatted_value.strip()
                )

                p_name, p_unit = (
                    self._identify_param_name(
                        txt_key
                    )
                )

                if not p_name:
                    continue

                num_val = None

                if (
                    ":" in txt_key
                    or "=" in txt_key
                ):

                    rhs = re.split(
                        r"[:=]",
                        txt_key,
                    )[-1].strip()

                    num_val = (
                        extract_numeric_value(
                            rhs
                        )
                    )

                if num_val is None:

                    for val_col in cols:

                        if val_col == key_col:
                            continue

                        v_cell = sheet.get_cell(
                            r,
                            val_col,
                        )

                        if (
                            v_cell
                            and v_cell.to_float()
                            is not None
                        ):

                            num_val = (
                                v_cell.to_float()
                            )
                            break

                if (
                    num_val is None
                    and key_col + 1 not in cols
                ):

                    adj_cell = sheet.get_cell(
                        r,
                        key_col + 1,
                    )

                    if (
                        adj_cell
                        and adj_cell.to_float()
                        is not None
                    ):

                        num_val = (
                            adj_cell.to_float()
                        )

                if num_val is not None:

                    if p_name not in extracted_params:

                        extracted_params[
                            p_name
                        ] = (
                            p_unit,
                            [],
                        )

                    extracted_params[
                        p_name
                    ][1].append(
                        num_val
                    )

        # ------------------------------------------------------------
        # Extract parameter values from headers.
        # ------------------------------------------------------------

        for idx, col in enumerate(cols):

            if col == x_col_idx:
                continue

            h_txt = header_texts[idx]

            sub_tokens = re.split(
                r"[_\s,;]+",
                h_txt,
            )

            for st in sub_tokens:

                p_name, p_unit = (
                    self._identify_param_name(
                        st
                    )
                )

                if not p_name:
                    continue

                num_val = None

                if (
                    ":" in st
                    or "=" in st
                ):

                    rhs = re.split(
                        r"[:=]",
                        st,
                    )[-1].strip()

                    num_val = (
                        extract_numeric_value(
                            st
                        )
                    )

                if num_val is None:

                    num_val = (
                        extract_numeric_value(
                            st
                        )
                    )

                if num_val is not None:

                    if p_name not in extracted_params:

                        extracted_params[
                            p_name
                        ] = (
                            p_unit,
                            [],
                        )

                    extracted_params[
                        p_name
                    ][1].append(
                        num_val
                    )

        # ------------------------------------------------------------
        # Build SweptParameter objects.
        # ------------------------------------------------------------

        swept_parameters: List[
            SweptParameter
        ] = []

        param_counter = 0

        for (
            p_name,
            (p_unit, p_vals),
        ) in extracted_params.items():

            param_counter += 1

            swept_parameters.append(
                SweptParameter(
                    parameter_id=(
                        f"param_{param_counter}"
                    ),
                    name=p_name,
                    unit=p_unit,
                    values=p_vals,
                )
            )

        # ------------------------------------------------------------
        # Existing fallback.
        # ------------------------------------------------------------

        if not swept_parameters:

            target_cols = [
                c
                for c in cols
                if c != x_col_idx
            ]

            swept_parameters.append(
                SweptParameter(
                    parameter_id="param_1",
                    name="Run",
                    unit="meta",
                    values=[
                        float(i + 1)
                        for i in range(
                            len(target_cols)
                        )
                    ],
                )
            )

        return swept_parameters

    # ================================================================
    # PARAMETER NAME IDENTIFICATION
    # ================================================================

    def _identify_param_name(
        self,
        text: str,
    ) -> Tuple[Optional[str], str]:

        t_lower = text.lower().strip()

        if (
            t_lower in [
                "wavelength",
                "lambda",
                "am 1.5g",
                "jtot(ma/cm2)",
                "jtot",
            ]
            or re.match(
                r"^\d+\s\*nm$",
                t_lower,
            )
        ):
            return None, ""

        if (
            "interface defect density 1"
            in t_lower
            or "idd1" in t_lower
        ):
            return (
                "Interface_Defect_Density_1",
                "cm^-3",
            )

        elif (
            "interface defect density 2"
            in t_lower
            or "idd2" in t_lower
        ):
            return (
                "Interface_Defect_Density_2",
                "cm^-3",
            )

        elif (
            "defect density"
            in t_lower
            or "total defect density"
            in t_lower
            or "defect 1" in t_lower
            or "idd" in t_lower
            or "nt=" in t_lower
        ):
            return (
                "Defect_Density",
                "cm^-3",
            )

        elif (
            "thickness" in t_lower
            or "thick" in t_lower
            or re.search(
                r"\bt[=**\[\(**]",
                t_lower,
            )
        ):
            return (
                "Thickness",
                "um",
            )

        elif (
            "temperature" in t_lower
            or "temp" in t_lower
            or "t(k)" in t_lower
        ):
            return (
                "Temperature",
                "K",
            )

        elif (
            "acceptor" in t_lower
            or "na=" in t_lower
        ):
            return (
                "Acceptor_Density",
                "cm^-3",
            )

        elif (
            "donor" in t_lower
            or "nd=" in t_lower
        ):
            return (
                "Donor_Density",
                "cm^-3",
            )

        elif "rs=" in t_lower:
            return (
                "Series_Resistance",
                "ohm.cm^2",
            )

        elif "rsh=" in t_lower:
            return (
                "Shunt_Resistance",
                "ohm.cm^2",
            )

        return None, ""

    # ================================================================
    # DEPENDENT VARIABLE EXTRACTION
    # ================================================================

    def _extract_dependent_variables(
        self,
        sheet: DocumentSheet,
        bbox: BoundingBox,
        valid_data_rows: List[int],
        cols: List[int],
        x_col_idx: int,
        header_row_idx: int,
        header_texts: List[str],
        x_name: str,
        x_values_len: int,
    ) -> List[DependentVariable]:

        """Groups target columns by physical quantity and handles
        side-by-side sub-tables.
        """

        target_columns: List[int] = []
        col_labels: List[str] = []

        for col in cols:

            if col == x_col_idx:
                continue

            h_text = (
                header_texts[
                    cols.index(col)
                ]
                if col in cols
                else ""
            )

            h_lower = h_text.lower()

            if h_lower in [
                "v(v)",
                "voltage",
                "lambda(nm)",
                "wavelength",
            ]:
                continue

            # --------------------------------------------------------
            # Check row 0 cell for metadata.
            # --------------------------------------------------------

            cell_0 = sheet.get_cell(
                0,
                col,
            )

            c0_text = (
                cell_0.formatted_value.lower()
                if (
                    cell_0
                    and cell_0.formatted_value
                )
                else ""
            )

            if any(
                desc in h_lower
                or desc in c0_text
                for desc in [
                    "scaps",
                    "version",
                    "dated",
                    "elis-ugent",
                ]
            ):
                continue

            has_numeric = any(
                sheet.get_cell(
                    r,
                    col,
                )
                and sheet.get_cell(
                    r,
                    col,
                ).is_numeric()
                for r in valid_data_rows[:10]
            )

            if not has_numeric:
                continue

            target_columns.append(col)

            # --------------------------------------------------------
            # Recover trace label.
            # --------------------------------------------------------

            trace_label = h_text

            # --------------------------------------------------------
            # Only look for a better label elsewhere in the sheet if
            # the column's own header ISN'T already an unambiguous,
            # sheet-provided value. A bare numeric header (e.g. "300"
            # for a T=300K trace column) already IS the real, specific
            # sweep value for that column and must not be overwritten
            # by unrelated text that happens to sit 2 columns to the
            # left for layout reasons (e.g. a block's own title/axis
            # header cell) -- that previously caused numeric-header
            # columns to be mislabeled with the block title or the
            # axis name itself instead of their own value.
            # Non-numeric headers (generic, blank, "Unnamed: N", etc.)
            # still go through the exact same lookup as before.
            # --------------------------------------------------------
            _h_text_is_bare_number = False
            try:
                float((h_text or "").strip())
                _h_text_is_bare_number = True
            except (TypeError, ValueError):
                _h_text_is_bare_number = False

            if (
                not _h_text_is_bare_number
                and col - 2 >= bbox.start_col
            ):

                pre_cell = (
                    sheet.get_cell(
                        0,
                        col - 2,
                    )
                    or sheet.get_cell(
                        header_row_idx,
                        col - 2,
                    )
                )

                if (
                    pre_cell
                    and pre_cell.formatted_value
                ):

                    lbl = (
                        pre_cell.formatted_value.strip()
                    )

                    if (
                        lbl
                        and not any(
                            k in lbl.lower()
                            for k in [
                                "v(v)",
                                "voltage",
                                "jtot(ma/cm2)",
                                "lambda(nm)",
                                "wavelength",
                                "scaps",
                                "version",
                                "dated",
                                "elis-ugent",
                            ]
                        )
                    ):
                        trace_label = lbl

            if trace_label == h_text:

                if (
                    cell_0
                    and cell_0.formatted_value
                    and cell_0.formatted_value.strip()
                ):

                    val_0 = (
                        cell_0.formatted_value.strip()
                    )

                    if (
                        val_0.lower()
                        not in [
                            "v(v)",
                            "voltage",
                            "jtot(ma/cm2)",
                            "current",
                            "lambda(nm)",
                            "wavelength",
                        ]
                        and not any(
                            k in val_0.lower()
                            for k in [
                                "scaps",
                                "version",
                                "dated",
                                "elis-ugent",
                            ]
                        )
                    ):
                        trace_label = val_0

            col_labels.append(
                trace_label
            )

        if not target_columns:

            target_columns = [
                c
                for c in cols
                if c != x_col_idx
            ]

            col_labels = [
                header_texts[
                    cols.index(c)
                ]
                if c in cols
                else f"Col_{c}"
                for c in target_columns
            ]

        # ============================================================
        # Determine block-wide target identity.
        # ============================================================

        block_target = None
        block_unit = None

        all_block_texts = []

        for col in cols:

            h_text = (
                header_texts[
                    cols.index(col)
                ]
                if col in cols
                else ""
            )

            all_block_texts.append(
                h_text
            )

            cell_0 = sheet.get_cell(
                0,
                col,
            )

            if (
                cell_0
                and cell_0.formatted_value
            ):
                all_block_texts.append(
                    cell_0.formatted_value
                )

        block_text_combined = " ".join(
            all_block_texts
        ).lower()

        if (
            "detectivity"
            in block_text_combined
            or "jones"
            in block_text_combined
            or "d*"
            in block_text_combined
        ):
            block_target = "Detectivity"
            block_unit = "Jones"

        elif (
            "responsivity"
            in block_text_combined
            or "a/w"
            in block_text_combined
            or "r ("
            in block_text_combined
            or "r("
            in block_text_combined
        ):
            block_target = "Responsivity"
            block_unit = "A/W"

        elif (
            "alpha"
            in block_text_combined
            or "1/cm"
            in block_text_combined
            or "cm^-1"
            in block_text_combined
            or "absorption"
            in block_text_combined
        ):
            block_target = "Absorption"
            block_unit = "1/cm"

        elif (
            "qe"
            in block_text_combined
            or "quantum"
            in block_text_combined
        ):
            block_target = "QE"
            block_unit = "%"

        elif any(
            k in block_text_combined
            for k in [
                "jtot",
                "current",
                "j(",
                "j_",
                "jsc",
                "voc",
                "ff",
                "pce",
                "eta",
            ]
        ):
            block_target = "Current"
            block_unit = "mA/cm^2"

        # ============================================================
        # Column target classification.
        # ============================================================

        col_info_list: List[
            Tuple[int, str, str, str]
        ] = []

        for idx, col in enumerate(
            target_columns
        ):

            lbl = col_labels[idx]

            l_lower = lbl.lower()

            h_lower = (
                header_texts[
                    cols.index(col)
                ].lower()
                if col in cols
                else ""
            )

            t_name, t_unit = None, None

            # Explicit target keywords first.

            if (
                "energy" in h_lower
                or "ev" in h_lower
                or "energy" in l_lower
                or "ev" in l_lower
            ):
                t_name = "Photon_Energy"
                t_unit = "eV"

            elif (
                "detectivity" in h_lower
                or "jones" in h_lower
                or "d*" in h_lower
                or "detectivity" in l_lower
                or "jones" in l_lower
                or "d*" in l_lower
            ):
                t_name = "Detectivity"
                t_unit = "Jones"

            elif (
                "responsivity" in h_lower
                or "a/w" in h_lower
                or "r (" in h_lower
                or "r(" in h_lower
                or "responsivity" in l_lower
                or "a/w" in l_lower
                or "r (" in l_lower
                or "r(" in l_lower
            ):
                t_name = "Responsivity"
                t_unit = "A/W"

            elif (
                "alpha" in h_lower
                or "1/cm" in h_lower
                or "cm^-1" in h_lower
                or "absorption" in h_lower
                or "alpha" in l_lower
                or "1/cm" in l_lower
                or "cm^-1" in l_lower
                or "absorption" in l_lower
            ):
                t_name = "Absorption"
                t_unit = "1/cm"

            elif (
                "qe" in h_lower
                or "quantum" in h_lower
                or "qe" in l_lower
                or "quantum" in l_lower
            ):
                t_name = "QE"
                t_unit = "%"

            elif any(
                k in h_lower or k in l_lower
                for k in [
                    "jtot",
                    "current",
                    "j(",
                    "j_",
                    "jsc",
                ]
            ):
                t_name = "Current"
                t_unit = "mA/cm^2"

            # --------------------------------------------------------
            # Inherit parent block identity.
            # --------------------------------------------------------

            if not t_name:

                if block_target:

                    t_name = block_target
                    t_unit = block_unit

                else:

                    if x_name == "Wavelength":
                        t_name = "QE"
                        t_unit = "%"

                    else:
                        t_name = "Current"
                        t_unit = "mA/cm^2"

            col_info_list.append(
                (
                    col,
                    lbl,
                    t_name,
                    t_unit,
                )
            )

        # ============================================================
        # Group columns by physical target.
        # ============================================================

        grouped_targets: Dict[
            Tuple[str, str],
            List[Tuple[int, str]],
        ] = {}

        for (
            col,
            lbl,
            t_name,
            t_unit,
        ) in col_info_list:

            key = (
                t_name,
                t_unit,
            )

            if key not in grouped_targets:
                grouped_targets[key] = []

            grouped_targets[key].append(
                (
                    col,
                    lbl,
                )
            )

        # ============================================================
        # Construct dependent variables.
        # ============================================================

        targets: List[
            DependentVariable
        ] = []

        target_counter = 0

        for (
            t_name,
            t_unit,
        ), group_cols in grouped_targets.items():

            target_counter += 1

            matrix: List[
                List[float]
            ] = []

            labels: List[str] = [
                lbl
                for _, lbl in group_cols
            ]

            for r in valid_data_rows:

                row_vals = []

                for col, _ in group_cols:

                    cell = sheet.get_cell(
                        r,
                        col,
                    )

                    flt_val = (
                        cell.to_float()
                        if cell
                        else None
                    )

                    row_vals.append(
                        flt_val
                        if flt_val is not None
                        else float("nan")
                    )

                matrix.append(
                    row_vals
                )

            targets.append(
                DependentVariable(
                    target_id=(
                        f"target_{target_counter:02d}"
                    ),
                    name=t_name,
                    unit=t_unit,
                    is_sparse=False,
                    data_shape=[
                        x_values_len,
                        len(group_cols),
                    ],
                    dense_matrix=matrix,
                    column_labels=labels,
                )
            )

        return targets