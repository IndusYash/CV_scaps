from __future__ import annotations

from app.domain.audit import BlockCategory, CandidateEvaluation
from app.domain.ir import IntermediateRepresentationV2
from app.semantics.base_plugin import ISemanticPlugin


class SummaryMetricsPlugin(ISemanticPlugin):
    @property
    def plugin_id(self) -> str:
        return "SummaryMetricsPlugin_v1.0"

    @property
    def specificity_rank(self) -> int:
        return 6

    @property
    def category(self) -> BlockCategory:
        return BlockCategory.SUMMARY

    def can_handle(self, ir: IntermediateRepresentationV2) -> CandidateEvaluation:
        match_reasons = []
        rejection_reasons = []
        score = 0.0

        if ir.summary_metrics or (not ir.axes and ir.targets):
            score += 0.70
            match_reasons.append("Block contains scalar performance metrics (Voc, Jsc, FF, Eta).")

        return CandidateEvaluation(
            plugin_id=self.plugin_id,
            score=min(1.0, score),
            match_reasons=match_reasons,
            rejection_reasons=rejection_reasons,
        )

    def classify(self, ir: IntermediateRepresentationV2) -> str:
        return "SUMMARY_METRICS"
