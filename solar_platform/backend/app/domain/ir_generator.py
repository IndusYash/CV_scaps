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
        """Generates IntermediateRepresentationV2 with structural curve purity, trace identity, and parameter banner extraction."""
        bbox = block.bounding_box
        rows = list(range(bbox.start_row, bbox.end_row + 1))
        cols = list(range(bbox.start_col, bbox.end_col + 1))

        if len(rows) < 2 or len(cols) < 1:
            return IntermediateRepresentationV2()

        # Step 1: Detect Real Table Header Row
        header_row_idx = bbox.start_row
        header_evidence = "Row 0 assumed as default top header."

        for r in range(bbox.start_row, min(bbox.start_row + 40, max(bbox.start_row + 1, bbox.end_row - 4))):
            c_curr = sheet.get_cell(r, cols[0])
            if not c_curr or c_curr.is_numeric():
                continue

            n_count = sum(
                1 for off in range(1, 7)
                if r + off <= bbox.end_row and sheet.get_cell(r + off, cols[0]) and sheet.get_cell(r + off, cols[0]).is_numeric()
            )
            if n_count >= 3:
                header_row_idx = r
                header_evidence = f"Row {r} selected as curve table header."
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

        # Step 2: Detect Parameter-Sweep Summary Tables (Voc, Jsc, FF, Eta / PCE vs Parameter P)
        sample_h_lower = [h.lower() for h in header_texts]
        summary_metric_keywords = ["voc", "jsc", "ff", "eta", "pce", "v_mpp", "j_mpp"]
        matched_summary_keys = [h for h in sample_h_lower if any(k in h for k in summary_metric_keywords)]

        if len(matched_summary_keys) >= 2:
            block.category = BlockCategory.SUMMARY
            data_start_row = header_row_idx + 1
            data_rows = [
                r for r in range(data_start_row, bbox.end_row + 1)
                if sheet.get_cell(r, cols[0]) and sheet.get_cell(r, cols[0]).is_numeric()
            ]

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

            # Build 1D Independent Axis for the Swept Parameter (Col 0)
            param_x_vals = [
                sheet.get_cell(r, cols[0]).to_float()
                for r in data_rows
                if sheet.get_cell(r, cols[0]) and sheet.get_cell(r, cols[0]).to_float() is not None
            ]

            p_axis_name, p_axis_unit = self._identify_sheet_param_axis(sheet.sheet_name, header_texts[0])

            param_axis = IndependentAxis(
                axis_id="axis_param_01",
                name=p_axis_name,
                unit=p_axis_unit,
                is_regular_sampling=False,
                values=param_x_vals,
            )

            # Build 1D Target Variables for each metric column (Cols 1+)
            summary_targets: List[DependentVariable] = []
            for c_idx in range(1, len(cols)):
                col = cols[c_idx]
                h_name = header_texts[c_idx]
                t_name, t_unit = self._identify_metric_target(h_name)

                y_vals = [
                    sheet.get_cell(r, col).to_float()
                    for r in data_rows
                ]
                y_matrix = [[v if v is not None else float("nan")] for v in y_vals]

                summary_targets.append(
                    DependentVariable(
                        target_id=f"target_sum_{c_idx}",
                        name=t_name,
                        unit=t_unit,
                        is_sparse=False,
                        data_shape=[len(param_x_vals), 1],
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

        # Step 3: Curve Block Axis & Sub-Table Detection
        x_col_idx = cols[0]
        x_name = "Voltage"
        x_unit = "V"

        first_row_numeric = sum(
            1 for r in range(header_row_idx + 1, min(header_row_idx + 5, bbox.end_row + 1))
            if sheet.get_cell(r, cols[0]) and sheet.get_cell(r, cols[0]).is_numeric()
        )
        if first_row_numeric == 0 and len(cols) > 1:
            x_col_idx = cols[1]

        for idx, col in enumerate(cols):
            h_text = header_texts[idx].lower()
            if any(k in h_text for k in ["lambda", "wavelength", "l(nm)", "w(nm)"]):
                x_col_idx = col
                x_name = "Wavelength"
                x_unit = "nm"
                break
            elif any(k in h_text for k in ["v(v)", "voltage"]):
                x_col_idx = col
                x_name = "Voltage"
                x_unit = "V"
                break

        # Step 4: Structural Curve Row Extraction
        data_start_row = header_row_idx + 1
        valid_data_rows: List[int] = []

        for r in range(data_start_row, bbox.end_row + 1):
            x_cell = sheet.get_cell(r, x_col_idx)
            x_val = x_cell.to_float() if x_cell else None
            
            if x_val is None:
                if not valid_data_rows:
                    continue
                else:
                    break

            row_text = " ".join([
                sheet.get_cell(r, c).formatted_value
                for c in cols if sheet.get_cell(r, c) and sheet.get_cell(r, c).formatted_value
            ]).lower()

            if any(desc_kw in row_text for desc_kw in ["dark current", "recombination", "voc", "jsc", "pce", "eta"]):
                break

            valid_data_rows.append(r)

        x_values = [
            sheet.get_cell(r, x_col_idx).to_float()
            for r in valid_data_rows
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

        # Step 5: Parameter Banner & Header Extraction
        swept_parameters: List[SweptParameter] = self._extract_parameters(
            sheet=sheet,
            block=block,
            header_row_idx=header_row_idx,
            cols=cols,
            x_col_idx=x_col_idx,
            header_texts=header_texts,
        )

        # Step 6: Dependent Variable Target Extraction & Separation
        targets: List[DependentVariable] = self._extract_dependent_variables(
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

        return IntermediateRepresentationV2(
            ir_version="v2.0",
            axes=[axis],
            parameters=swept_parameters,
            targets=targets,
            metadata_attributes={
                "sheet_name": sheet.sheet_name,
                "header_row": header_row_idx,
                "valid_rows_count": len(valid_data_rows),
            },
        )

    def _identify_sheet_param_axis(self, sheet_name: str, col0_header: str) -> Tuple[str, str]:
        s_lower = sheet_name.lower()
        if "t (" in s_lower or "t(k)" in s_lower:
            return "Temperature", "K"
        elif "na" in s_lower:
            return "Acceptor_Density", "cm^-3"
        elif "rs" in s_lower and "rsh" not in s_lower:
            return "Series_Resistance", "ohm.cm^2"
        elif "rsh" in s_lower:
            return "Shunt_Resistance", "ohm.cm^2"
        
        c_clean = col0_header.strip()
        if c_clean and c_clean.lower() != "bp 1":
            return c_clean, "meta"
        return "Parameter", "unit"

    def _identify_metric_target(self, col_header: str) -> Tuple[str, str]:
        h_lower = col_header.lower().strip()
        if "eta" in h_lower or "pce" in h_lower:
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
        elif "r (" in h_lower or "r(" in h_lower:
            return "Responsivity", "A/W"
        return col_header.strip() or "Metric", "unit"

    def _extract_parameters(
        self,
        sheet: DocumentSheet,
        block: DataBlockAudit,
        header_row_idx: int,
        cols: List[int],
        x_col_idx: int,
        header_texts: List[str],
    ) -> List[SweptParameter]:
        extracted_params: Dict[str, Tuple[str, List[float]]] = {}
        bbox = block.bounding_box

        if sheet.sheet_name in ["QE (T+DD)", "QE (IDD)"]:
            block_num = 1
            if "_" in block.block_id:
                try:
                    block_num = int(block.block_id.rsplit("_", 1)[-1])
                except ValueError:
                    block_num = 1

            if sheet.sheet_name == "QE (T+DD)":
                t_idx = (block_num - 1) // 8
                dd_idx = (block_num - 1) % 8
                thick_val = round(0.3 + t_idx * 0.1, 4)
                dd_val = float(10 ** (11 + dd_idx))

                return [
                    SweptParameter(parameter_id="param_1", name="Temperature", unit="K", values=[300.0]),
                    SweptParameter(parameter_id="param_2", name="Thickness", unit="um", values=[thick_val]),
                    SweptParameter(parameter_id="param_3", name="Defect_Density", unit="cm^-3", values=[dd_val]),
                ]

            elif sheet.sheet_name == "QE (IDD)":
                dd_val = float(10 ** (8 + (block_num - 1)))
                return [
                    SweptParameter(parameter_id="param_1", name="Temperature", unit="K", values=[300.0]),
                    SweptParameter(parameter_id="param_2", name="Defect_Density", unit="cm^-3", values=[dd_val]),
                ]

        scan_start_row = max(0, bbox.start_row - 10)

        for r in range(scan_start_row, header_row_idx):
            search_cols = [0] + [c for c in cols if c != 0]
            
            for key_col in search_cols:
                cell_key = sheet.get_cell(r, key_col)
                if not cell_key or not cell_key.formatted_value:
                    continue

                txt_key = cell_key.formatted_value.strip()

                p_name, p_unit = self._identify_param_name(txt_key)
                if not p_name:
                    continue

                num_val = None
                if ":" in txt_key or "=" in txt_key:
                    rhs = re.split(r"[:=]", txt_key)[-1].strip()
                    num_val = extract_numeric_value(rhs)

                if num_val is None:
                    for val_col in cols:
                        if val_col == key_col:
                            continue
                        v_cell = sheet.get_cell(r, val_col)
                        if v_cell and v_cell.to_float() is not None:
                            num_val = v_cell.to_float()
                            break

                if num_val is None and key_col + 1 not in cols:
                    adj_cell = sheet.get_cell(r, key_col + 1)
                    if adj_cell and adj_cell.to_float() is not None:
                        num_val = adj_cell.to_float()

                if num_val is not None:
                    if p_name not in extracted_params:
                        extracted_params[p_name] = (p_unit, [])
                    extracted_params[p_name][1].append(num_val)

        for idx, col in enumerate(cols):
            if col == x_col_idx:
                continue
            h_txt = header_texts[idx]
            sub_tokens = re.split(r"[_\s,;]+", h_txt)

            for st in sub_tokens:
                p_name, p_unit = self._identify_param_name(st)
                if not p_name:
                    continue
                
                num_val = None
                if ":" in st or "=" in st:
                    rhs = re.split(r"[:=]", st)[-1].strip()
                    num_val = extract_numeric_value(st)
                if num_val is None:
                    num_val = extract_numeric_value(st)

                if num_val is not None:
                    if p_name not in extracted_params:
                        extracted_params[p_name] = (p_unit, [])
                    extracted_params[p_name][1].append(num_val)

        swept_parameters: List[SweptParameter] = []
        param_counter = 0

        for p_name, (p_unit, p_vals) in extracted_params.items():
            param_counter += 1
            swept_parameters.append(
                SweptParameter(
                    parameter_id=f"param_{param_counter}",
                    name=p_name,
                    unit=p_unit,
                    values=p_vals,
                )
            )

        if not swept_parameters:
            target_cols = [c for c in cols if c != x_col_idx]
            swept_parameters.append(
                SweptParameter(
                    parameter_id="param_1",
                    name="Run",
                    unit="meta",
                    values=[float(i + 1) for i in range(len(target_cols))],
                )
            )

        return swept_parameters

    def _identify_param_name(self, text: str) -> Tuple[Optional[str], str]:
        t_lower = text.lower().strip()
        if t_lower in ["wavelength", "lambda", "am 1.5g", "jtot(ma/cm2)", "jtot"] or re.match(r"^\d+\s*nm$", t_lower):
            return None, ""

        if "interface defect density 1" in t_lower or "idd1" in t_lower:
            return "Interface_Defect_Density_1", "cm^-3"
        elif "interface defect density 2" in t_lower or "idd2" in t_lower:
            return "Interface_Defect_Density_2", "cm^-3"
        elif "defect density" in t_lower or "total defect density" in t_lower or "defect 1" in t_lower or "idd" in t_lower or "nt=" in t_lower:
            return "Defect_Density", "cm^-3"
        elif "thickness" in t_lower or "thick" in t_lower or re.search(r"\bt[=\[\(]", t_lower):
            return "Thickness", "um"
        elif "temperature" in t_lower or "temp" in t_lower or "t(k)" in t_lower:
            return "Temperature", "K"
        elif "acceptor" in t_lower or "na=" in t_lower:
            return "Acceptor_Density", "cm^-3"
        elif "donor" in t_lower or "nd=" in t_lower:
            return "Donor_Density", "cm^-3"
        elif "rs=" in t_lower:
            return "Series_Resistance", "ohm.cm^2"
        elif "rsh=" in t_lower:
            return "Shunt_Resistance", "ohm.cm^2"
        return None, ""

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
        """Groups target columns by physical quantity and handles side-by-side sub-tables."""
        target_columns: List[int] = []
        col_labels: List[str] = []

        for col in cols:
            if col == x_col_idx:
                continue

            h_text = header_texts[cols.index(col)] if col in cols else ""
            h_lower = h_text.lower()

            if h_lower in ["v(v)", "voltage", "lambda(nm)", "wavelength"]:
                continue

            # Check row 0 cell text for metadata / description strings
            cell_0 = sheet.get_cell(0, col)
            c0_text = cell_0.formatted_value.lower() if (cell_0 and cell_0.formatted_value) else ""

            # Skip non-data description header columns (e.g. SCAPS version/footer text strings)
            if any(desc in h_lower or desc in c0_text for desc in ["scaps", "version", "dated", "elis-ugent"]):
                continue

            has_numeric = any(
                sheet.get_cell(r, col) and sheet.get_cell(r, col).is_numeric()
                for r in valid_data_rows[:10]
            )

            if not has_numeric:
                continue

            target_columns.append(col)

            # Recover trace label: check col - 2 cell at row 0 first for side-by-side sub-tables
            trace_label = h_text
            if col - 2 >= bbox.start_col:
                pre_cell = sheet.get_cell(0, col - 2) or sheet.get_cell(header_row_idx, col - 2)
                if pre_cell and pre_cell.formatted_value:
                    lbl = pre_cell.formatted_value.strip()
                    if lbl and not any(k in lbl.lower() for k in ["v(v)", "voltage", "jtot(ma/cm2)", "lambda(nm)", "wavelength", "scaps", "version", "dated", "elis-ugent"]):
                        trace_label = lbl

            if trace_label == h_text:
                if cell_0 and cell_0.formatted_value and cell_0.formatted_value.strip():
                    val_0 = cell_0.formatted_value.strip()
                    if val_0.lower() not in ["v(v)", "voltage", "jtot(ma/cm2)", "current", "lambda(nm)", "wavelength"] and not any(k in val_0.lower() for k in ["scaps", "version", "dated", "elis-ugent"]):
                        trace_label = val_0

            col_labels.append(trace_label)

        if not target_columns:
            target_columns = [c for c in cols if c != x_col_idx]
            col_labels = [header_texts[cols.index(c)] if c in cols else f"Col_{c}" for c in target_columns]

        col_info_list: List[Tuple[int, str, str, str]] = []
        for idx, col in enumerate(target_columns):
            lbl = col_labels[idx]
            l_lower = lbl.lower()
            h_lower = header_texts[cols.index(col)].lower() if col in cols else ""

            if "energy" in h_lower or "ev" in h_lower or "energy" in l_lower or "ev" in l_lower:
                t_name, t_unit = "Photon_Energy", "eV"
            elif "detectivity" in h_lower or "jones" in h_lower or "d*" in h_lower or "detectivity" in l_lower or "jones" in l_lower or "d*" in l_lower:
                t_name, t_unit = "Detectivity", "Jones"
            elif "responsivity" in h_lower or "a/w" in h_lower or "r (" in h_lower or "r(" in h_lower or "responsivity" in l_lower or "a/w" in l_lower or "r (" in l_lower or "r(" in l_lower:
                t_name, t_unit = "Responsivity", "A/W"
            elif "alpha" in h_lower or "1/cm" in h_lower or "cm^-1" in h_lower or "absorption" in h_lower or "alpha" in l_lower or "1/cm" in l_lower or "cm^-1" in l_lower or "absorption" in l_lower:
                t_name, t_unit = "Absorption", "1/cm"
            elif "qe" in h_lower or "quantum" in h_lower or "qe" in l_lower or "quantum" in l_lower:
                t_name, t_unit = "QE", "%"
            else:
                t_name, t_unit = ("QE", "%") if x_name == "Wavelength" else ("Current", "mA/cm^2")

            col_info_list.append((col, lbl, t_name, t_unit))

        grouped_targets: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
        for col, lbl, t_name, t_unit in col_info_list:
            key = (t_name, t_unit)
            if key not in grouped_targets:
                grouped_targets[key] = []
            grouped_targets[key].append((col, lbl))

        targets: List[DependentVariable] = []
        target_counter = 0

        for (t_name, t_unit), group_cols in grouped_targets.items():
            target_counter += 1
            matrix: List[List[float]] = []
            labels: List[str] = [lbl for _, lbl in group_cols]

            for r in valid_data_rows:
                row_vals = []
                for col, _ in group_cols:
                    cell = sheet.get_cell(r, col)
                    flt_val = cell.to_float() if cell else None
                    row_vals.append(flt_val if flt_val is not None else float("nan"))
                matrix.append(row_vals)

            targets.append(
                DependentVariable(
                    target_id=f"target_{target_counter:02d}",
                    name=t_name,
                    unit=t_unit,
                    is_sparse=False,
                    data_shape=[x_values_len, len(group_cols)],
                    dense_matrix=matrix,
                    column_labels=labels,
                )
            )

        return targets
