from __future__ import annotations

from typing import Dict
import pandas as pd
from app.domain.ir import IntermediateRepresentationV2


class RandomForestMLAdapter:
    def transform_to_long_dataframe(
        self,
        ir: IntermediateRepresentationV2,
    ) -> pd.DataFrame:
        """Transforms an N-dimensional IntermediateRepresentationV2 payload into a standard long DataFrame.

        Supports multi-parameter sweeps: [X_Axis, P1, P2, ..., Target].
        """
        if not ir.axes or not ir.targets or not ir.targets[0].dense_matrix:
            return pd.DataFrame()

        axis = ir.axes[0]
        target = ir.targets[0]

        x_name = axis.name
        target_name = target.name

        x_vals = axis.values
        matrix = target.dense_matrix
        params = ir.parameters if ir.parameters else []

        records = []
        for r_idx, x_val in enumerate(x_vals):
            if r_idx < len(matrix):
                row_targets = matrix[r_idx]
                for c_idx, t_val in enumerate(row_targets):
                    if pd.notna(t_val):
                        rec = {x_name: x_val}
                        for p_idx, param in enumerate(params):
                            p_vals = param.values
                            p_val = p_vals[c_idx] if c_idx < len(p_vals) else 1.0
                            rec[param.name] = p_val
                        rec[target_name] = t_val
                        records.append(rec)

        df_long = pd.DataFrame(records)
        return df_long
