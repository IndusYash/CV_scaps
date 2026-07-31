from __future__ import annotations

from app.domain.audit import BlockCategory, CandidateEvaluation
from app.domain.ir import IntermediateRepresentationV2
from app.semantics.base_plugin import ISemanticPlugin


class GenericParametricPlugin(ISemanticPlugin):
    @property
    def plugin_id(self) -> str:
        return "GenericParametricPlugin_v1.0"

    @property
    def specificity_rank(self) -> int:
        return 1  # Lowest priority fallback

    @property
    def category(self) -> BlockCategory:
        return BlockCategory.CURVE

    def can_handle(self, ir: IntermediateRepresentationV2) -> CandidateEvaluation:
        return CandidateEvaluation(
            plugin_id=self.plugin_id,
            score=0.40,
            match_reasons=["Universal fallback plugin for unclassified parametric sweeps."],
            rejection_reasons=[],
        )

    def classify(self, ir: IntermediateRepresentationV2) -> str:
        return "GENERIC_PARAMETRIC"
