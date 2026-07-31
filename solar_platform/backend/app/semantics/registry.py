from __future__ import annotations

from typing import List, Tuple
from app.domain.audit import CandidateEvaluation, DataBlockAudit, PluginSelectionAudit
from app.domain.ir import IntermediateRepresentationV2
from app.semantics.base_plugin import ISemanticPlugin
from app.semantics.plugins.absorption_plugin import AbsorptionPlugin
from app.semantics.plugins.detectivity_plugin import DetectivityPlugin
from app.semantics.plugins.generic_plugin import GenericParametricPlugin
from app.semantics.plugins.jv_plugin import JVPlugin
from app.semantics.plugins.qe_plugin import QEPlugin
from app.semantics.plugins.responsivity_plugin import ResponsivityPlugin
from app.semantics.plugins.summary_plugin import SummaryMetricsPlugin


class SemanticRegistry:
    def __init__(self):
        self._plugins: List[ISemanticPlugin] = []
        # Register core domain plugins (higher specificity ranks evaluated first)
        self.register_plugin(AbsorptionPlugin())
        self.register_plugin(JVPlugin())
        self.register_plugin(QEPlugin())
        self.register_plugin(ResponsivityPlugin())
        self.register_plugin(DetectivityPlugin())
        self.register_plugin(SummaryMetricsPlugin())
        self.register_plugin(GenericParametricPlugin())

    def register_plugin(self, plugin: ISemanticPlugin) -> None:
        """Registers a new ISemanticPlugin extension plugin."""
        self._plugins.append(plugin)

    def evaluate_and_tag(
        self,
        ir: IntermediateRepresentationV2,
        block_audit: DataBlockAudit,
    ) -> str:
        """Evaluates all registered plugins against an IR object and attaches PluginSelectionAudit to block_audit."""
        evaluations: List[Tuple[ISemanticPlugin, CandidateEvaluation]] = []

        for plugin in self._plugins:
            eval_res = plugin.can_handle(ir)
            evaluations.append((plugin, eval_res))

        # 3-Level Deterministic Tie-Breaking Logic:
        # Level 1: Highest Score
        # Level 2: Specificity Rank
        # Level 3: Registration Order Index
        evaluations.sort(
            key=lambda item: (
                item[1].score,
                item[0].specificity_rank,
                -self._plugins.index(item[0]),
            ),
            reverse=True,
        )

        selected_plugin, best_eval = evaluations[0]

        # Attach plugin evaluation trace to audit
        candidate_audits = [ev[1] for ev in evaluations]
        block_audit.plugin_selection = PluginSelectionAudit(
            candidate_evaluations=candidate_audits,
            selected_plugin_id=selected_plugin.plugin_id,
            selection_rationale=f"Selected '{selected_plugin.plugin_id}' with score {best_eval.score:.2f} (Rank {selected_plugin.specificity_rank}).",
            selection_confidence=best_eval.score,
        )

        block_audit.category = selected_plugin.category
        semantic_label = selected_plugin.classify(ir)
        block_audit.semantic_label = semantic_label

        return semantic_label
