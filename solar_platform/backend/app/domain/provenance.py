from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class TransformationStep(BaseModel):
    step_name: str
    applied_rule: str
    input_value: Any = None
    output_value: Any = None


class CellLineage(BaseModel):
    source_document: str
    source_sheet: str
    cell_range: str
    raw_text_entry: str = ""
    transformation_history: List[TransformationStep] = Field(default_factory=list)

    def add_step(self, step_name: str, applied_rule: str, input_value: Any, output_value: Any) -> None:
        self.transformation_history.append(
            TransformationStep(
                step_name=step_name,
                applied_rule=applied_rule,
                input_value=input_value,
                output_value=output_value,
            )
        )
