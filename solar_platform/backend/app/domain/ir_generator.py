from __future__ import annotations

import re
from typing import List, Optional, Tuple
from app.domain.audit import BoundingBox, DataBlockAudit, DecisionTraceEntry
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
        """Generates an IntermediateRepresentationV2 object from a DocumentSheet and DataBlock bounding box."""
        bbox = block.bounding_box
        rows = list(range(bbox.start_row, bbox.end_row + 1))
        cols = list(range(bbox.start_col, bbox.end_col + 1))

        if len(rows) < 2 or len(cols) < 2:
            return IntermediateRepresentationV2()

        # Step 1: Detect Header Row (First row containing strings followed by numeric rows)
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

        # Extract Column Header Text
        header_texts = [
            sheet.get_cell(header_row_idx, c).formatted_value if sheet.get_cell(header_row_idx, c) else f"Col_{c}"
            for c in cols
        ]

        # Step 2: Identify Independent Axis (First column matching Voltage V or Wavelength Lambda)
        data_start_row = header_row_idx + 1
        data_rows = list(range(data_start_row, bbox.end_row + 1))

        x_col_idx = cols[0]
        x_name = "Voltage"
        x_unit = "V"

        for idx, col in enumerate(cols):
            h_text = header_texts[idx].lower()
            if "lambda" in h_text or "wavelength" in h_text:
                x_col_idx = col
                x_name = "Wavelength"
                x_unit = "nm"
                break
            elif "v(" in h_text or h_text in ["voltage", "v"]:
                x_col_idx = col
                x_name = "Voltage"
                x_unit = "V"
                break

        x_values = [
            sheet.get_cell(r, x_col_idx).to_float()
            for r in data_rows
            if sheet.get_cell(r, x_col_idx) and sheet.get_cell(r, x_col_idx).to_float() is not None
        ]

        axis = IndependentAxis(
            axis_id="axis_01",
            name=x_name,
            unit=x_unit,
            is_regular_sampling=True,
            values=x_values,
        )

        # Step 3: Extract Swept Parameters and Target Variables
        param_values: List[float] = []
        target_columns: List[int] = []
        param_name = "Parameter"
        param_unit = ""

        for idx, col in enumerate(cols):
            if col == x_col_idx:
                continue

            h_text = header_texts[idx]
            num_val = extract_numeric_value(h_text)
            if num_val is not None:
                param_values.append(num_val)
                target_columns.append(col)

        if not param_values:
            # Fallback for single-run or metrics
            param_values = [1.0]
            target_columns = [c for c in cols if c != x_col_idx]

        parameter = SweptParameter(
            parameter_id="param_01",
            name=param_name,
            unit=param_unit,
            values=sorted(list(set(param_values))),
        )

        # Build Dependent Target Matrices
        target_matrix: List[List[float]] = []
        for r in data_rows:
            row_vals = []
            for col in target_columns:
                cell = sheet.get_cell(r, col)
                flt_val = cell.to_float() if cell else 0.0
                row_vals.append(flt_val if flt_val is not None else 0.0)
            target_matrix.append(row_vals)

        target = DependentVariable(
            target_id="target_01",
            name="Current" if x_name == "Voltage" else "QE",
            unit="mA/cm^2" if x_name == "Voltage" else "%",
            is_sparse=False,
            data_shape=[len(x_values), len(param_values)],
            dense_matrix=target_matrix,
        )

        ir = IntermediateRepresentationV2(
            ir_version="v2.0",
            axes=[axis],
            parameters=[parameter],
            targets=[target],
            metadata_attributes={
                "sheet_name": sheet.sheet_name,
                "header_row": header_row_idx,
            },
        )

        return ir
