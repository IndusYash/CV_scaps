from __future__ import annotations

from app.domain.audit import BlockCategory, CandidateEvaluation
from app.domain.ir import IntermediateRepresentationV2
from app.semantics.base_plugin import ISemanticPlugin


class AbsorptionPlugin(ISemanticPlugin):
    @property
    def plugin_id(self) -> str:
        return "AbsorptionPlugin_v1.0"

    @property
    def specificity_rank(self) -> int:
        return 9  # Higher priority than generic QE

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
            score += 0.40
            match_reasons.append("Axis matches Wavelength grid.")

        if ir.targets:
            t_name = ir.targets[0].name.lower()
            t_unit = ir.targets[0].unit.lower()
            if "absorption" in t_name or "alpha" in t_name or "1/cm" in t_unit or "cm^-1" in t_unit:
                score += 0.55
                match_reasons.append("Target matches Absorption Coefficient (1/cm).")

        return CandidateEvaluation(
            plugin_id=self.plugin_id,
            score=min(1.0, score),
            match_reasons=match_reasons,
            rejection_reasons=rejection_reasons,
        )

    def classify(self, ir: IntermediateRepresentationV2) -> str:
        return "ABSORPTION"
