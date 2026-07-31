from __future__ import annotations

from typing import List
from app.domain.audit import DataBlockAudit
from app.domain.ir import IntermediateRepresentationV2


class ScientificValidator:
    def validate_ir(
        self,
        ir: IntermediateRepresentationV2,
        block_audit: DataBlockAudit,
    ) -> float:
        """Executes non-mutating physical bounds checks and populates validation_warnings.

        Never mutates or discards extracted IR data.
        """
        warnings: List[str] = []
        validity_score = 1.0

        if not ir.axes:
            warnings.append("Scientific Warning: Missing independent axis grid.")
            validity_score -= 0.3
        else:
            axis = ir.axes[0]
            # Grid Monotonicity Check
            values = axis.values
            if len(values) > 1:
                is_monotonic = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
                if not is_monotonic:
                    warnings.append(f"Scientific Warning: Independent axis '{axis.name}' values are not strictly monotonic.")
                    validity_score -= 0.2

        if ir.targets:
            target = ir.targets[0]
            if target.dense_matrix:
                all_vals = [val for row in target.dense_matrix for val in row]
                if all_vals:
                    max_val = max(all_vals)
                    min_val = min(all_vals)

                    # Physical bounds check for Quantum Efficiency
                    if target.name.lower() == "qe" and max_val > 105.0:
                        warnings.append(f"Physics Warning: QE max value {max_val:.1f}% exceeds physical limit 100%.")
                        validity_score -= 0.25

                    # Physical bounds check for Current Density
                    if target.name.lower() in ["current", "jtot"] and (max_val > 150.0 or min_val < -150.0):
                        warnings.append(f"Physics Warning: Current density values [{min_val:.1f}, {max_val:.1f}] exceed standard PV ranges.")
                        validity_score -= 0.20

        block_audit.validation_warnings = warnings
        block_audit.scientific_validity = max(0.0, validity_score)
        return block_audit.scientific_validity
