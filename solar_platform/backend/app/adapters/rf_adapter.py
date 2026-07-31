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

        Schema: [X_Axis, Parameter, Target] (e.g. Voltage, Thickness, Current).
        """
        if not ir.axes or not ir.targets or not ir.targets[0].dense_matrix:
            return pd.DataFrame()

        axis = ir.axes[0]
        target = ir.targets[0]
        param = ir.parameters[0] if ir.parameters else None

        x_name = axis.name
        param_name = param.name if param else "Parameter"
        target_name = target.name

        x_vals = axis.values
        param_vals = param.values if param else [1.0]

        records = []
        matrix = target.dense_matrix

        for r_idx, x_val in enumerate(x_vals):
            if r_idx < len(matrix):
                row_targets = matrix[r_idx]
                for p_idx, p_val in enumerate(param_vals):
                    if p_idx < len(row_targets):
                        t_val = row_targets[p_idx]
                        records.append({
                            x_name: x_val,
                            param_name: p_val,
                            target_name: t_val
                        })

        df_long = pd.DataFrame(records)
        return df_long
