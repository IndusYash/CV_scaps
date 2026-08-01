from __future__ import annotations

import re
from typing import Dict, List, Optional
import pandas as pd
from app.domain.ir import IntermediateRepresentationV2


class RandomForestMLAdapter:
    def transform_to_long_dataframe(
        self,
        ir: IntermediateRepresentationV2,
        target_index: int = 0,
        block_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Transforms an N-dimensional IntermediateRepresentationV2 payload into a standard long DataFrame.

        Supports multi-parameter sweeps: [X_Axis, P1, P2, ..., Target].
        Enforces independent Trace_ID metadata for every dependent-variable column in multi-column curves.
        """
        if not ir.axes or not ir.targets or target_index >= len(ir.targets):
            return pd.DataFrame()

        target = ir.targets[target_index]
        if not target.dense_matrix:
            return pd.DataFrame()

        axis = ir.axes[0]
        x_name = axis.name
        target_name = target.name
        x_vals = axis.values
        matrix = target.dense_matrix
        params = ir.parameters if ir.parameters else []
        column_labels = getattr(target, "column_labels", [])

        # Deduplicate target columns if column labels are identical, redundant, or padded with empty strings
        num_target_cols = target.data_shape[1] if len(target.data_shape) > 1 else 1
        if column_labels and len(column_labels) > 1:
            clean_labels = [re.sub(r"_c\d+$", "", str(lbl)).strip() for lbl in column_labels]
            non_empty = [lbl for lbl in clean_labels if lbl]
            if len(set(clean_labels)) == 1 or len(set(non_empty)) <= 1:
                num_target_cols = 1

        records = []
        for r_idx, x_val in enumerate(x_vals):
            if r_idx < len(matrix):
                row_targets = matrix[r_idx][:num_target_cols]
                for c_idx, t_val in enumerate(row_targets):
                    if pd.notna(t_val):
                        rec = {x_name: x_val}

                        # Add Physical Swept Parameters
                        for p_idx, param in enumerate(params):
                            p_vals = param.values
                            p_val = p_vals[c_idx] if c_idx < len(p_vals) else (p_vals[0] if p_vals else 1.0)
                            rec[param.name] = p_val

                        # Determine base trace label
                        if column_labels and c_idx < len(column_labels) and column_labels[c_idx]:
                            base_label = column_labels[c_idx]
                            # Disambiguate duplicate labels within the same block
                            if column_labels.count(base_label) > 1:
                                base_label = f"{base_label}_c{c_idx+1}"
                        else:
                            base_label = f"trace_{c_idx+1:03d}"

                        # Disambiguate Trace_ID using block_id when multiple blocks exist
                        if block_id and num_target_cols > 1:
                            rec["Trace_ID"] = f"{base_label}_{block_id}"
                        else:
                            rec["Trace_ID"] = base_label

                        # Add Block_ID metadata
                        if block_id:
                            rec["Block_ID"] = block_id

                        rec[target_name] = t_val
                        records.append(rec)

        df_long = pd.DataFrame(records)
        return df_long
