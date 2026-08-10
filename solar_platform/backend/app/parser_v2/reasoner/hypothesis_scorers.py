"""
parser_v2.reasoner.hypothesis_scorers
--------------------------------------
Pure evidence-based scoring functions for the three approved generic structural
hypotheses (NumericCurveGrid, ScalarSweepTable, CategoryTable) and scientific
label layering.

Score semantics
~~~~~~~~~~~~~~~
All scoring functions produce **evidence-support scores**, NOT probabilities or
percentages.  Scores are accumulated as a sum of positive evidence contributions
minus contradicting evidence penalties.

* A score of 0.00 for a hypothesis means no positive scoring rule contributed
  evidence for it.  It does NOT mean the hypothesis is deemed impossible.
* A score of 1.00 means the maximum evidence-contribution ceiling was reached.
* Scores are clamped to [0.0, 1.0] as an evidence-capacity bound, NOT as a
  probability normalisation.

Concept separation
~~~~~~~~~~~~~~~~~~
Each scorer evaluates ONE structural hypothesis (NumericCurveGrid,
ScalarSweepTable, CategoryTable, or Unknown) and then *layers* a scientific
interpretation (JV, QE, RESPONSIVITY, …) on top.  These two concepts are
carried separately in HypothesisCandidate and must never be conflated.

The AcceptanceState of the enclosing RegionInterpretation is determined *after*
all scorers run and is not computed here.

Rules:
  - All evidence items from EvidenceBag are evaluated together with positive and
    negative contributions.
  - No single if/elif rule determines the score alone.
  - Rule 7A: "No parameter found" does NOT penalize generic NumericCurveGrid.
  - Rule 7B: Single-trace NumericCurveGrid is fully valid.
  - Rule 7C: Plain numeric headers require contextual & structural support.
  - Rule 7D: Sheet name is weak evidence (+0.05 max; never overrides region evidence).
  - Rule 7E: Structural score and scientific semantics are derived separately.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.parser_v2.probes.evidence import CellRef, EvidenceBag, EvidencePolarity
from app.parser_v2.reasoner.hypothesis import (
    HypothesisCandidate,
    StructuralHypothesis,
)
from app.parser_v2.scanner.region import CandidateRegion


def _collect_provenance(bag: EvidenceBag) -> List[CellRef]:
    refs: List[CellRef] = []
    for item in bag.items:
        for cell_ref in item.source_cells:
            if cell_ref not in refs:
                refs.append(cell_ref)
                if len(refs) >= 15:
                    break
        if len(refs) >= 15:
            break
    return refs


def score_numeric_curve_grid(bag: EvidenceBag, region: CandidateRegion) -> HypothesisCandidate:
    """
    Score the generic NumericCurveGrid structural hypothesis and layer scientific interpretation.
    """
    score = 0.0
    supporting: List[str] = []
    contradicting: List[str] = []
    missing: List[str] = []

    # 1. Monotonic axis evidence (score primary x-column once)
    mono_items = bag.filter_by_name("axis_monotonicity")
    if mono_items:
        item = mono_items[0]
        if item.polarity == EvidencePolarity.POSITIVE:
            score += 0.35
            direction = item.value.get("direction", "monotonic")
            n_pts = item.value.get("n_points", 0)
            supporting.append(f"Primary column forms monotonic {direction} grid with {n_pts} points (+0.35).")
        elif item.polarity == EvidencePolarity.NEGATIVE:
            score -= 0.20
            contradicting.append("Primary axis column is numeric but not strictly monotonic (-0.20).")
    else:
        missing.append("No axis monotonicity evidence found.")

    # 2. String category column 0 contradiction
    cat_items = bag.filter_by_name("axis_non_numeric_category")
    if cat_items:
        score -= 0.60
        contradicting.append("Column 0 consists of string category labels, not a numeric grid (-0.60).")

    # 3. Numeric density
    num_dens_items = bag.filter_by_name("numeric_density_high")
    if any(item.polarity == EvidencePolarity.POSITIVE for item in num_dens_items):
        score += 0.25
        supporting.append(f"Region has high numeric density {region.numeric_density:.1%} (+0.25).")

    mixed_dens_items = bag.filter_by_name("numeric_density_low")
    if any(item.polarity == EvidencePolarity.NEGATIVE for item in mixed_dens_items):
        score -= 0.40
        contradicting.append(f"Region has low numeric density {region.numeric_density:.1%} (-0.40).")

    # 4. Trace repetition & trace count
    rep_items = bag.filter_by_name("repeated_trace_group")
    for item in rep_items:
        if item.polarity == EvidencePolarity.POSITIVE:
            t_cnt = item.value.get("trace_count", 0)
            if t_cnt > 1:
                score += 0.25
                supporting.append(f"Discovered group of {t_cnt} numeric trace columns (+0.25).")
            elif t_cnt == 1:
                # Rule 7B: Single-trace NumericCurveGrid is fully valid!
                score += 0.20
                supporting.append("Single-trace numeric grid detected — valid device characterization curve (+0.20).")
        elif item.polarity == EvidencePolarity.NEGATIVE:
            # Rule 7A: "No parameter / trace group" does NOT heavily penalize generic curve grid
            score -= 0.10
            contradicting.append("No repeated numeric trace columns discovered (-0.10).")

    # 5. Scalar metrics & Stacked curve blocks contradiction
    lbl_items = bag.filter_by_name("candidate_label_concepts")
    scalar_metric_count = 0
    param_concept_count = 0
    distinct_concepts: List[str] = []
    if lbl_items:
        for item in lbl_items:
            concepts = item.value.get("distinct_concepts", [])
            distinct_concepts.extend(concepts)
            for c in concepts:
                if c in ("Voc", "FF", "PCE", "V_MPP", "J_MPP", "Jsc"):
                    scalar_metric_count += 1
                if c in ("Thickness", "Temperature", "Defect_Density", "Acceptor_Density", "Donor_Density", "Series_Resistance", "Shunt_Resistance"):
                    param_concept_count += 1

    param_items = bag.filter_by_name("candidate_parameter_expressions")
    has_param_expr = any(item.polarity == EvidencePolarity.POSITIVE for item in param_items)
    has_monotonic_col = any(item.polarity == EvidencePolarity.POSITIVE for item in mono_items)

    stacked_items = bag.filter_by_name("stacked_axis_reset_sequence")
    has_stacked_blocks = any(item.polarity == EvidencePolarity.POSITIVE and item.value.get("n_blocks", 0) >= 2 for item in stacked_items)

    if has_stacked_blocks:
        score -= 0.45
        contradicting.append("Region consists of vertically stacked sub-blocks — better fit for StackedCurveBlocks (-0.45).")

    interleaved_items = bag.filter_by_name("interleaved_column_group")
    has_interleaved_group = any(item.polarity == EvidencePolarity.POSITIVE and item.value.get("n_curves", 0) >= 2 for item in interleaved_items)

    if has_interleaved_group:
        score -= 0.45
        contradicting.append("Region consists of horizontally interleaved curve pairs — better fit for InterleavedCurveGrid (-0.45).")

    # 2-column/3-column parameter vs scalar metric summary table (e.g. thickness/Rsh vs Jsc) -> penalize CurveGrid
    is_scalar_table_shape = (
        scalar_metric_count >= 1
        and (has_param_expr or param_concept_count >= 1 or has_monotonic_col)
        and region.n_cols <= 3
        and region.n_rows <= 15
    )
    if scalar_metric_count >= 3 or is_scalar_table_shape:
        score -= 0.45
        contradicting.append("Contains scalar metric headers with parameter sweep — likely a ScalarSweepTable (-0.45).")

    # 6. Sheet name weak contextual evidence (Rule 7D)
    sheet_name_lower = region.sheet_name.lower()
    if any(k in sheet_name_lower for k in ("etl", "pvk", "htl", "temperature", "defect", "interface", "rsh", "rs", "optimized")):
        score += 0.05
        supporting.append(f"Sheet name '{region.sheet_name}' weakly supports numeric curve grid (+0.05).")

    # Clamp to evidence-support bounds [0.0, 1.0].
    # 0.0 = lower bound (no positive evidence; does NOT mean impossible).
    # 1.0 = upper bound (maximum evidence-contribution ceiling; NOT a probability).
    final_score = max(0.0, min(1.0, score))

    # Layered scientific interpretation (Rule 3 & 7E)
    scientific_label: Optional[str] = None
    axis_concept: Optional[str] = None
    target_concept: Optional[str] = None

    if "Position" in distinct_concepts or any("x(um)" in t.lower() for t in region.raw_banner_texts + region.raw_header_texts):
        # Sheet1 / Device profile rule (Rule 7E & 9)
        axis_concept = "Position"
        scientific_label = "UNKNOWN"
        final_score = max(0.0, final_score - 0.65)
        contradicting.append("Position axis 'x(um)' detected — not a standard curve experiment (-0.65).")
    else:
        if "Voltage" in distinct_concepts:
            axis_concept = "Voltage"
        elif "Wavelength" in distinct_concepts:
            axis_concept = "Wavelength"

        # Detectivity takes precedence over generic Current/Jsc
        if "Detectivity" in distinct_concepts:
            target_concept = "Detectivity"
            scientific_label = "DETECTIVITY"
        elif "Responsivity" in distinct_concepts:
            target_concept = "Responsivity"
            scientific_label = "RESPONSIVITY"
        elif "QE" in distinct_concepts:
            target_concept = "QE"
            scientific_label = "QE"
        elif "Current" in distinct_concepts or "Jsc" in distinct_concepts:
            target_concept = "Current"
            scientific_label = "JV"

        if not scientific_label and final_score >= 0.50:
            if axis_concept == "Voltage":
                scientific_label = "JV"
            elif axis_concept == "Wavelength":
                scientific_label = "QE"
            else:
                scientific_label = "UNKNOWN"

    # Rationale: human-readable summary of structural evidence-support score
    # and the separately-derived scientific interpretation.
    rationale = (
        f"NumericCurveGrid evidence-support score: {final_score:.2f} "
        f"(Scientific interpretation: {scientific_label}, "
        f"Axis: {axis_concept}, Target: {target_concept}). "
        f"Score is not a probability."
    )

    return HypothesisCandidate(
        structure=StructuralHypothesis.NUMERIC_CURVE_GRID,
        scientific_label=scientific_label,
        axis_concept=axis_concept,
        target_concept=target_concept,
        score=final_score,
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        missing_evidence=missing,
        provenance_cells=_collect_provenance(bag),
        rationale=rationale,
    )


def score_scalar_sweep_table(bag: EvidenceBag, region: CandidateRegion) -> HypothesisCandidate:
    """
    Score the generic ScalarSweepTable structural hypothesis.
    """
    score = 0.0
    supporting: List[str] = []
    contradicting: List[str] = []
    missing: List[str] = []

    # 1. String category column 0 contradiction
    cat_items = bag.filter_by_name("axis_non_numeric_category")
    if cat_items:
        score -= 0.35
        contradicting.append("Column 0 consists of string category labels, not a numeric parameter sweep (-0.35).")

    # 2. Scalar PV metric headers & Parameter concept headers
    lbl_items = bag.filter_by_name("candidate_label_concepts")
    scalar_metrics: List[str] = []
    param_concepts: List[str] = []
    distinct_concepts: List[str] = []

    if lbl_items:
        for item in lbl_items:
            concepts = item.value.get("distinct_concepts", [])
            distinct_concepts.extend(concepts)
            for c in concepts:
                if c in ("Jsc", "Voc", "FF", "PCE", "V_MPP", "J_MPP", "Responsivity", "QE"):
                    scalar_metrics.append(c)
                if c in ("Thickness", "Temperature", "Defect_Density", "Acceptor_Density", "Donor_Density", "Series_Resistance", "Shunt_Resistance"):
                    param_concepts.append(c)

    mono_items = bag.filter_by_name("axis_monotonicity")
    has_monotonic_col = any(item.polarity == EvidencePolarity.POSITIVE for item in mono_items)

    param_items = bag.filter_by_name("candidate_parameter_expressions")
    has_param_expr = (
        any(item.polarity == EvidencePolarity.POSITIVE for item in param_items)
        or len(param_concepts) >= 1
        or has_monotonic_col
    )

    unique_scalar_metrics = list(set(scalar_metrics))
    if len(unique_scalar_metrics) >= 2 or (len(unique_scalar_metrics) >= 1 and has_param_expr):
        score += 0.50
        supporting.append(f"Headers match {len(unique_scalar_metrics)} scalar metrics + parameter sweep (+0.50).")
    elif len(unique_scalar_metrics) == 1:
        score += 0.25
        supporting.append(f"Headers match 1 scalar metric: {unique_scalar_metrics[0]} (+0.25).")
    else:
        score -= 0.40
        contradicting.append("No scalar PV metric concepts detected in headers (-0.40).")
        missing.append("Scalar metric headers (Voc, Jsc, FF, PCE).")

    # 3. Col 0 parameter axis
    if has_param_expr:
        score += 0.25
        supporting.append("Col 0 / banner headers contain parameter sweep expressions/concepts (+0.25).")

    if has_monotonic_col:
        # Monotonic parameter column
        score += 0.20
        supporting.append("Parameter column exhibits monotonic values (+0.20).")

    # 4. Contradiction: Spectral/Voltage curve grid (e.g. 15+ data points per column with voltage/wavelength axis)
    has_curve_axis = ("Voltage" in distinct_concepts or "Wavelength" in distinct_concepts or
                      any(k in t.lower() for t in region.raw_header_texts + region.raw_banner_texts for k in ("v(v)", "v (v)", "lambda", "nm")))

    if region.n_rows >= 15 and has_curve_axis:
        score -= 0.55
        contradicting.append(f"Region has {region.n_rows} rows with spectral/voltage grid headers — likely a NumericCurveGrid (-0.55).")

    # 5. Sheet name weak evidence (Rule 7D)
    sheet_name_lower = region.sheet_name.lower()
    if any(k in sheet_name_lower for k in ("na", "t (k)", "temperature", "rs", "rsh", "defect")):
        score += 0.05
        supporting.append(f"Sheet name '{region.sheet_name}' weakly supports scalar sweep table (+0.05).")

    # Clamp to evidence-support bounds [0.0, 1.0].
    final_score = max(0.0, min(1.0, score))
    scientific_label = "SCALAR_SWEEP" if final_score >= 0.40 else None

    # Rationale: human-readable summary of structural evidence-support score.
    rationale = (
        f"ScalarSweepTable evidence-support score: {final_score:.2f} "
        f"(Scalar metrics found: {unique_scalar_metrics}). Score is not a probability."
    )

    return HypothesisCandidate(
        structure=StructuralHypothesis.SCALAR_SWEEP_TABLE,
        scientific_label=scientific_label,
        axis_concept="Parameter",
        target_concept="Scalar_Metrics",
        score=final_score,
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        missing_evidence=missing,
        provenance_cells=_collect_provenance(bag),
        rationale=rationale,
    )


def score_category_table(bag: EvidenceBag, region: CandidateRegion) -> HypothesisCandidate:
    """
    Score the generic CategoryTable structural hypothesis.
    """
    score = 0.0
    supporting: List[str] = []
    contradicting: List[str] = []
    missing: List[str] = []

    # 1. Non-numeric category column 0
    cat_items = bag.filter_by_name("axis_non_numeric_category")
    if cat_items:
        score += 0.65
        hdr = cat_items[0].value.get("header_text", "")
        supporting.append(f"Column 0 consists of string category labels (e.g. '{hdr}') (+0.65).")
    else:
        # Contradiction: Col 0 is monotonic numeric
        mono_items = bag.filter_by_name("axis_monotonicity")
        if any(item.polarity == EvidencePolarity.POSITIVE for item in mono_items):
            score -= 0.60
            contradicting.append("Column 0 is a strictly monotonic numeric grid, not string categories (-0.60).")

    # 2. Interface / Layer comparison tokens in banners
    lbl_items = bag.filter_by_name("candidate_label_concepts")
    if lbl_items:
        for item in lbl_items:
            concepts = item.value.get("distinct_concepts", [])
            if "Interface_Density" in concepts:
                score += 0.20
                supporting.append("Header contains interface comparison tokens (+0.20).")

    # 3. Numeric curve contradiction
    if region.numeric_density >= 0.85 and region.n_rows >= 20:
        score -= 0.40
        contradicting.append("Region is a dense numeric grid with >=20 rows (-0.40).")

    # Clamp to evidence-support bounds [0.0, 1.0].
    final_score = max(0.0, min(1.0, score))
    scientific_label = "CATEGORY" if final_score >= 0.40 else None

    # Rationale: human-readable summary of structural evidence-support score.
    rationale = (
        f"CategoryTable evidence-support score: {final_score:.2f}. "
        f"Score is not a probability."
    )

    return HypothesisCandidate(
        structure=StructuralHypothesis.CATEGORY_TABLE,
        scientific_label=scientific_label,
        axis_concept="Category",
        target_concept="Comparative_Values",
        score=final_score,
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        missing_evidence=missing,
        provenance_cells=_collect_provenance(bag),
        rationale=rationale,
    )


def score_stacked_curve_blocks(bag: EvidenceBag, region: CandidateRegion) -> HypothesisCandidate:
    """
    Pure evidence-based scoring function for the generic StackedCurveBlocks hypothesis.

    Evaluates:
      - stacked_axis_reset_sequence evidence from AxisProbe (n_blocks >= 2)
      - structural uniformity of sub-block lengths
      - inter-block banner parameters (helpful but not mandatory)
      - layered scientific interpretation (QE, JV, RESPONSIVITY, etc.)
    """
    score = 0.0
    supporting: List[str] = []
    contradicting: List[str] = []
    missing: List[str] = []

    stacked_items = bag.filter_by_name("stacked_axis_reset_sequence")
    has_stacked = False
    n_blocks = 0

    for item in stacked_items:
        if item.polarity == EvidencePolarity.POSITIVE:
            has_stacked = True
            n_blocks = item.value.get("n_blocks", 0)
            score += 0.55
            supporting.append(f"AxisProbe detected {n_blocks} stacked monotonic x-axis blocks (+0.55).")
            if n_blocks >= 5:
                score += 0.20
                supporting.append(f"Large stacked series of {n_blocks} uniform blocks (+0.20).")

    if not has_stacked or n_blocks < 2:
        score -= 0.60
        contradicting.append("No repeated monotonic x-axis reset sequence detected (-0.60).")

    # Parameter / banner evidence (helpful but NOT mandatory)
    param_items = bag.filter_by_name("candidate_parameter_expressions")
    if any(item.polarity == EvidencePolarity.POSITIVE for item in param_items):
        score += 0.10
        supporting.append("Inter-block banner parameters detected (+0.10).")

    # Layered scientific interpretation
    lbl_items = bag.filter_by_name("candidate_label_concepts")
    axis_concept = "Wavelength"
    target_concept = "QE"
    sci_label = "UNKNOWN"

    if lbl_items:
        for item in lbl_items:
            concepts = item.value.get("distinct_concepts", [])
            for c in concepts:
                if c in ("QE", "Current", "Responsivity", "Detectivity", "Absorption"):
                    sci_label = c.upper() if c != "Current" else "JV"
                    score += 0.10
                    supporting.append(f"Headers match scientific concept {c} (+0.10).")

    final_score = max(0.0, min(1.0, score))

    rationale = (
        f"StackedCurveBlocks evidence-support score: {final_score:.2f} ({n_blocks} stacked blocks). "
        f"Score is not a probability."
    )

    return HypothesisCandidate(
        structure=StructuralHypothesis.STACKED_CURVE_BLOCKS,
        scientific_label=sci_label if final_score >= 0.40 else "UNKNOWN",
        axis_concept=axis_concept,
        target_concept=target_concept,
        score=final_score,
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        missing_evidence=missing,
        provenance_cells=_collect_provenance(bag),
        rationale=rationale,
    )


def score_interleaved_curve_grid(bag: EvidenceBag, region: CandidateRegion) -> HypothesisCandidate:
    """
    Pure evidence-based scoring function for the generic InterleavedCurveGrid hypothesis.

    Evaluates:
      - interleaved_column_group evidence from RepetitionProbe (n_curves >= 2)
      - multiple independent monotonic x-axis columns from AxisProbe
      - header parameter tokens/expressions
      - layered scientific interpretation (JV, QE, RESPONSIVITY, etc.)
    """
    score = 0.0
    supporting: List[str] = []
    contradicting: List[str] = []
    missing: List[str] = []

    interleaved_items = bag.filter_by_name("interleaved_column_group")
    has_interleaved = False
    n_curves = 0

    for item in interleaved_items:
        if item.polarity == EvidencePolarity.POSITIVE:
            has_interleaved = True
            n_curves = item.value.get("n_curves", 0)
            score += 0.55
            supporting.append(f"RepetitionProbe detected {n_curves} repeated independent (axis, target) column pairs (+0.55).")
            if n_curves >= 3:
                score += 0.20
                supporting.append(f"Multi-curve family of {n_curves} interleaved pairs (+0.20).")

    if not has_interleaved or n_curves < 2:
        score -= 0.60
        contradicting.append("No repeated independent (axis, target) column pairs detected (-0.60).")

    # Header parameter evidence (helpful but NOT mandatory)
    param_items = bag.filter_by_name("candidate_parameter_expressions")
    if any(item.polarity == EvidencePolarity.POSITIVE for item in param_items):
        score += 0.10
        supporting.append("Header parameter expressions detected for interleaved curves (+0.10).")

    # Layered scientific interpretation
    lbl_items = bag.filter_by_name("candidate_label_concepts")
    axis_concept = "Voltage"
    target_concept = "Current"
    sci_label = "UNKNOWN"

    if lbl_items:
        for item in lbl_items:
            concepts = item.value.get("distinct_concepts", [])
            for c in concepts:
                if c in ("Voltage", "Current", "Wavelength", "QE", "Responsivity", "Detectivity"):
                    if c in ("Voltage", "Current"):
                        sci_label = "JV"
                    elif c in ("Wavelength", "QE"):
                        sci_label = "QE"
                    elif c == "Responsivity":
                        sci_label = "RESPONSIVITY"
                    elif c == "Detectivity":
                        sci_label = "DETECTIVITY"
                    score += 0.10
                    supporting.append(f"Headers match scientific concept {c} (+0.10).")

    final_score = max(0.0, min(1.0, score))

    rationale = (
        f"InterleavedCurveGrid evidence-support score: {final_score:.2f} ({n_curves} interleaved curves). "
        f"Score is not a probability."
    )

    return HypothesisCandidate(
        structure=StructuralHypothesis.INTERLEAVED_CURVE_GRID,
        scientific_label=sci_label if final_score >= 0.40 else "UNKNOWN",
        axis_concept=axis_concept,
        target_concept=target_concept,
        score=final_score,
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        missing_evidence=missing,
        provenance_cells=_collect_provenance(bag),
        rationale=rationale,
    )


def score_unknown(bag: EvidenceBag, region: CandidateRegion) -> HypothesisCandidate:
    """
    Fallback Unknown structural hypothesis.  Always evaluated alongside the three
    named structural hypotheses so the ranked candidate set is always complete.

    The baseline evidence-support score of 0.20 is a minimum floor to prevent a
    zero-score winner from being automatically selected.  It is NOT a 20%
    probability that the region is truly unknown.
    """
    return HypothesisCandidate(
        structure=StructuralHypothesis.UNKNOWN,
        scientific_label="UNKNOWN",
        axis_concept=None,
        target_concept=None,
        # Minimum baseline evidence-support score (not a probability).
        score=0.20,
        supporting_evidence=[
            "Fallback baseline: no specific structural evidence applies. "
            "Score=0.20 is a minimum floor, not a 20% probability."
        ],
        contradicting_evidence=[],
        missing_evidence=[],
        provenance_cells=_collect_provenance(bag),
        rationale=(
            "Unknown fallback structural hypothesis with minimum baseline "
            "evidence-support score of 0.20 (not a probability)."
        ),
    )
