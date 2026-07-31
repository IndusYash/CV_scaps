from __future__ import annotations

from app.domain.audit import BlockCategory, CandidateEvaluation
from app.domain.ir import IntermediateRepresentationV2
from app.semantics.base_plugin import ISemanticPlugin


class QEPlugin(ISemanticPlugin):
    @property
    def plugin_id(self) -> str:
        return "QEPlugin_v1.0"

    @property
    def specificity_rank(self) -> int:
        return 8

    @property
    def category(self) -> BlockCategory:
        return BlockCategory.CURVE

    def can_handle(self, ir: IntermediateRepresentationV2) -> CandidateEvaluation:
        match_reasons = []
        rejection_reasons = []
        score = 0.0

        if not ir.axes:
            return CandidateEvaluation(
                plugin_id=self.plugin_id,
                score=0.0,
                rejection_reasons=["No independent axis present in IR."],
            )

        axis = ir.axes[0]
        if axis.name.lower() in ["wavelength", "lambda"] or axis.unit.lower() == "nm":
            score += 0.50
            match_reasons.append("Primary axis matches Wavelength grid [200, 2000]nm.")
        else:
            rejection_reasons.append(f"Axis name '{axis.name}' is not Wavelength.")

        if ir.targets and "qe" in ir.targets[0].name.lower():
            score += 0.40
            match_reasons.append("Target variable matches Quantum Efficiency.")

        return CandidateEvaluation(
            plugin_id=self.plugin_id,
            score=min(1.0, score),
            match_reasons=match_reasons,
            rejection_reasons=rejection_reasons,
        )

    def classify(self, ir: IntermediateRepresentationV2) -> str:
        return "QE"
