from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd

from app.domain.audit import SheetAudit, DataBlockAudit
from app.domain.ir import IntermediateRepresentationV2, DependentVariable
from app.adapters.rf_adapter import RandomForestMLAdapter


@dataclass
class ScientificExperiment:
    experiment_id: str
    sheet_name: str
    target_name: str
    target_unit: str
    axis_name: str
    axis_unit: str
    swept_parameter_names: List[str]
    blocks: List[DataBlockAudit] = field(default_factory=list)
    combined_dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def semantic_type(self) -> str:
        if self.target_name == "Current" and self.axis_name == "Voltage":
            return "JV"
        elif self.target_name == "Current" and self.axis_name not in ["Voltage", "Wavelength"]:
            return "SCALAR_SWEEP"
        elif self.target_name == "QE":
            return "QE"
        elif self.target_name == "Responsivity":
            return "RESPONSIVITY"
        return self.target_name


class ExperimentGrouperEngine:
    def group_sheet_blocks_into_experiments(
        self,
        sheet_audit: SheetAudit,
    ) -> List[ScientificExperiment]:
        """Groups sheet blocks into scientifically distinct experiment datasets based on IR compatibility."""
        if not sheet_audit.blocks:
            return []

        rf_adapter = RandomForestMLAdapter()
        experiments_dict: Dict[Tuple[str, str, str, str], ScientificExperiment] = {}

        for b in sheet_audit.blocks:
            ir = b.intermediate_representation
            if not ir or not ir.axes or not ir.targets:
                continue

            axis = ir.axes[0]
            axis_name = axis.name
            axis_unit = axis.unit

            # Process each target in the block separately (MULTI-TARGET DOWNSTREAM SEPARATION)
            for t_idx, target in enumerate(ir.targets):
                t_name = target.name
                t_unit = target.unit

                # Extract swept parameter names
                p_names = tuple(sorted([p.name for p in ir.parameters if p.name != "Run"]))

                exp_key = (sheet_audit.sheet_name, t_name, t_unit, axis_name)

                if exp_key not in experiments_dict:
                    safe_sheet = sheet_audit.sheet_name.replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
                    if axis_name not in ["Wavelength", "Voltage"]:
                        exp_id = f"exp_{safe_sheet}_{t_name}_vs_{axis_name}"
                    else:
                        exp_id = f"exp_{safe_sheet}_{t_name}"

                    experiments_dict[exp_key] = ScientificExperiment(
                        experiment_id=exp_id,
                        sheet_name=sheet_audit.sheet_name,
                        target_name=t_name,
                        target_unit=t_unit,
                        axis_name=axis_name,
                        axis_unit=axis_unit,
                        swept_parameter_names=list(p_names),
                        blocks=[],
                    )

                exp = experiments_dict[exp_key]
                exp.blocks.append(b)

                # Transform block IR target to long DataFrame
                block_df = rf_adapter.transform_to_long_dataframe(
                    ir, target_index=t_idx, block_id=b.block_id
                )

                if not block_df.empty:
                    if exp.combined_dataframe.empty:
                        exp.combined_dataframe = block_df
                    else:
                        exp.combined_dataframe = pd.concat(
                            [exp.combined_dataframe, block_df], ignore_index=True
                        )

        # Only return experiments that have actual data
        return [e for e in experiments_dict.values() if not e.combined_dataframe.empty]
