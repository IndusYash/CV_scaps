from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from app.domain.audit import BoundingBox, DataBlockAudit, DecisionTraceEntry, BlockCategory
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
    ) -> IntermediateRepresentationV2:
        """Generates IntermediateRepresentationV2 with token-based semantic parameter extraction."""
        bbox = block.bounding_box
        rows = list(range(bbox.start_row, bbox.end_row + 1))
        cols = list(range(bbox.start_col, bbox.end_col + 1))

        if len(rows) < 2 or len(cols) < 1:
            return IntermediateRepresentationV2()

        # Step 1: Detect Header Row
        header_row_idx = bbox.start_row
        header_evidence = "Row 0 assumed as default top header."

        for r in rows[:5]:
            numeric_count = sum(
                1 for c in cols
                if sheet.get_cell(r, c) and sheet.get_cell(r, c).is_numeric()
            )
            total = len(cols)
            if numeric_count / total < 0.5:
                header_row_idx = r
                header_evidence = f"Row {r} selected as header (numeric ratio = {numeric_count/total:.2f} < 0.50)."
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
            sheet.get_cell(header_row_idx, c).formatted_value if sheet.get_cell(header_row_idx, c) else f"Col_{c}"
            for c in cols
        ]
        sample_h_lower = [h.lower() for h in header_texts]

        # Step 2: Detect Summary Metrics Tables (Voc, Jsc, FF, Eta / PCE)
        summary_metric_keywords = ["voc", "jsc", "ff", "eta", "pce", "v_mpp", "j_mpp"]
        matched_summary_keys = [h for h in sample_h_lower if any(k in h for k in summary_metric_keywords)]

        if len(matched_summary_keys) >= 2:
            block.category = BlockCategory.SUMMARY
            data_start_row = header_row_idx + 1
            data_rows = list(range(data_start_row, bbox.end_row + 1))

            metrics_dict: Dict[str, Dict[str, float]] = {}
            for idx, col in enumerate(cols):
                col_name = header_texts[idx]
                col_vals = [
                    sheet.get_cell(r, col).to_float()
                    for r in data_rows
                    if sheet.get_cell(r, col) and sheet.get_cell(r, col).to_float() is not None
                ]
                if col_vals:
                    metrics_dict[col_name] = {
                        "mean": float(sum(col_vals) / len(col_vals)),
                        "max": float(max(col_vals)),
                        "min": float(min(col_vals)),
                        "latest": float(col_vals[-1]),
                    }

            ir_summary = IntermediateRepresentationV2(
                ir_version="v2.0",
                axes=[],
                parameters=[],
                targets=[],
                summary_metrics=metrics_dict,
                metadata_attributes={
                    "sheet_name": sheet.sheet_name,
                    "header_row": header_row_idx,
                    "is_summary_table": True,
                },
            )
            return ir_summary

        # Step 3: Curve Block Axis & Target Processing
        data_start_row = header_row_idx + 1
        data_rows = list(range(data_start_row, bbox.end_row + 1))

        x_col_idx = cols[0]
        x_name = "Voltage"
        x_unit = "V"

        for idx, col in enumerate(cols):
            h_text = header_texts[idx].lower()
            if any(k in h_text for k in ["lambda", "wavelength", "l(nm)", "w(nm)"]):
                x_col_idx = col
                x_name = "Wavelength"
                x_unit = "nm"
                break
            elif any(k in h_text for k in ["v(v)", "voltage", "v"]):
                x_col_idx = col
                x_name = "Voltage"
                x_unit = "V"
                break

        x_values = [
            sheet.get_cell(r, x_col_idx).to_float()
            for r in data_rows
            if sheet.get_cell(r, x_col_idx) and sheet.get_cell(r, x_col_idx).to_float() is not None
        ]

        if x_values and max(x_values) > 100.0 and x_name == "Voltage":
            x_name = "Wavelength"
            x_unit = "nm"

        axis = IndependentAxis(
            axis_id="axis_01",
            name=x_name,
            unit=x_unit,
            is_regular_sampling=True,
            values=x_values,
        )

        # Step 4: Sub-Token Parameter Extraction
        target_columns: List[int] = []
        param_tuples: List[List[float]] = []
        param_token_names: List[List[Tuple[str, str]]] = []

        for idx, col in enumerate(cols):
            if col == x_col_idx:
                continue

            h_text = header_texts[idx]
            sub_tokens = re.split(r"[_\s,;]+", h_text)
            parsed_nums = []
            token_info = []

            for st in sub_tokens:
                st_lower = st.lower()
                num_matches = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", st)
                if num_matches:
                    m = num_matches[0]
                    nv = extract_numeric_value(m)
                    if nv is not None:
                        parsed_nums.append(nv)
                        if "nt=" in st_lower or "dd" in st_lower or "10^" in st_lower or "cm^-3" in st_lower or "1e" in st_lower:
                            token_info.append(("Defect_Density", "cm^-3"))
                        elif re.search(r"(?:^|[^a-z])t=", st_lower) or "thick" in st_lower or "nm" in st_lower:
                            token_info.append(("Thickness", "nm"))
                        elif "temp" in st_lower or re.search(r"(?:^|[^a-z])t=", st_lower) or "k" in st_lower:
                            token_info.append(("Temperature", "K"))
                        elif "na=" in st_lower or "acceptor" in st_lower:
                            token_info.append(("Acceptor_Density", "cm^-3"))
                        elif "nd=" in st_lower or "donor" in st_lower:
                            token_info.append(("Donor_Density", "cm^-3"))
                        elif "rs=" in st_lower:
                            token_info.append(("Series_Resistance", "ohm.cm^2"))
                        elif "rsh=" in st_lower:
                            token_info.append(("Shunt_Resistance", "ohm.cm^2"))
                        else:
                            token_info.append(("Parameter_1", "scalar"))

            if parsed_nums:
                param_tuples.append(parsed_nums)
                param_token_names.append(token_info)
                target_columns.append(col)

        swept_parameters: List[SweptParameter] = []
        if param_tuples and param_token_names:
            max_p_cnt = max(len(pt) for pt in param_tuples)
            for p_idx in range(max_p_cnt):
                p_vals = [pt[p_idx] if p_idx < len(pt) else 1.0 for pt in param_tuples]
                p_name, p_unit = param_token_names[0][p_idx] if (param_token_names and p_idx < len(param_token_names[0])) else (f"Parameter_{p_idx+1}", "")
                swept_parameters.append(
                    SweptParameter(
                        parameter_id=f"param_{p_idx+1}",
                        name=p_name,
                        unit=p_unit,
                        values=p_vals,
                    )
                )
        else:
            target_columns = [c for c in cols if c != x_col_idx]
            swept_parameters.append(
                SweptParameter(
                    parameter_id="param_1",
                    name="Run",
                    unit="meta",
                    values=[1.0] * len(target_columns),
                )
            )

        # Step 5: Precise Block Target Variable & Unit Deduction
        col_headers_str = " ".join([header_texts[cols.index(c)] for c in target_columns if c in cols]).lower()
        sheet_l = sheet.sheet_name.lower()

        if "alpha" in col_headers_str or "1/cm" in col_headers_str or "cm^-1" in col_headers_str or "absorption" in sheet_l:
            target_name = "Absorption"
            target_unit = "1/cm"
        elif "responsivity" in col_headers_str or "r(a/w)" in col_headers_str:
            target_name = "Responsivity"
            target_unit = "A/W"
        elif "detectivity" in col_headers_str or "d*" in col_headers_str or "jones" in col_headers_str:
            target_name = "Detectivity"
            target_unit = "Jones"
        elif "qe" in col_headers_str or "quantum" in col_headers_str or "qe" in sheet_l or x_name == "Wavelength":
            target_name = "QE"
            target_unit = "%"
        else:
            target_name = "Current"
            target_unit = "mA/cm^2"

        # Build Target Matrix
        target_matrix: List[List[float]] = []
        for r in data_rows:
            row_vals = []
            for col in target_columns:
                cell = sheet.get_cell(r, col)
                flt_val = cell.to_float() if cell else None
                row_vals.append(flt_val if flt_val is not None else float("nan"))
            target_matrix.append(row_vals)

        target = DependentVariable(
            target_id="target_01",
            name=target_name,
            unit=target_unit,
            is_sparse=False,
            data_shape=[len(x_values), len(target_columns)],
            dense_matrix=target_matrix,
        )

        ir = IntermediateRepresentationV2(
            ir_version="v2.0",
            axes=[axis],
            parameters=swept_parameters,
            targets=[target],
            metadata_attributes={
                "sheet_name": sheet.sheet_name,
                "header_row": header_row_idx,
            },
        )

        return ir
