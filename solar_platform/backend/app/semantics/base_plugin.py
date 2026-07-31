from __future__ import annotations

from abc import ABC, abstractmethod
from app.domain.audit import BlockCategory, CandidateEvaluation
from app.domain.ir import IntermediateRepresentationV2


class ISemanticPlugin(ABC):
    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """Returns unique identifier for this plugin (e.g. 'JVPlugin_v1.0')."""
        pass

    @property
    @abstractmethod
    def specificity_rank(self) -> int:
        """Returns priority rank (higher numbers win tie-breaks)."""
        pass

    @property
    @abstractmethod
    def category(self) -> BlockCategory:
        """Returns block category ('CURVE', 'TABLE', 'SUMMARY')."""
        pass

    @abstractmethod
    def can_handle(self, ir: IntermediateRepresentationV2) -> CandidateEvaluation:
        """Evaluates match score (0.00 to 1.00) and rationale for an IR object."""
        pass

    @abstractmethod
    def classify(self, ir: IntermediateRepresentationV2) -> str:
        """Returns semantic label (e.g. 'JV', 'QE', 'RESPONSIVITY')."""
        pass
