"""Pattern-level tests: each regex tuple fires on representative academic prose."""

from __future__ import annotations

import re

from scicite_claims import ClaimType
from scicite_claims._patterns import (
    CAUSAL_PATTERNS,
    COMPARATIVE_PATTERNS,
    METHODOLOGICAL_PATTERNS,
    STATISTICAL_PATTERNS,
    THEORETICAL_PATTERNS,
    confidence_for,
    iter_pattern_matches,
)


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def test_statistical_patterns_match_percent_change() -> None:
    assert _matches_any(STATISTICAL_PATTERNS, "a 23% increase in accuracy")
    assert _matches_any(STATISTICAL_PATTERNS, "improved by 47%")


def test_statistical_patterns_match_p_value() -> None:
    assert _matches_any(STATISTICAL_PATTERNS, "p < 0.001")
    assert _matches_any(STATISTICAL_PATTERNS, "P = .05")


def test_statistical_patterns_match_correlation() -> None:
    assert _matches_any(STATISTICAL_PATTERNS, "correlation of r = 0.85")
    assert _matches_any(STATISTICAL_PATTERNS, "strong correlation")


def test_statistical_patterns_match_sample_size() -> None:
    assert _matches_any(STATISTICAL_PATTERNS, "n = 512")
    # The pattern is intentionally `sample (size|of) <num>`; it catches
    # "sample size 100" or "sample of 100" but does NOT catch the longer
    # "sample size of 100" phrasing. Captured here so future tightening
    # of the pattern can decide whether to broaden coverage.
    assert _matches_any(STATISTICAL_PATTERNS, "sample size 100")
    assert _matches_any(STATISTICAL_PATTERNS, "sample of 100")


def test_methodological_patterns_match_used_employed() -> None:
    assert _matches_any(METHODOLOGICAL_PATTERNS, "We used gradient boosting")
    assert _matches_any(METHODOLOGICAL_PATTERNS, "this study employed an RNN")


def test_methodological_patterns_match_novel_approach() -> None:
    assert _matches_any(METHODOLOGICAL_PATTERNS, "a novel method for classification")
    assert _matches_any(METHODOLOGICAL_PATTERNS, "the proposed algorithm")


def test_comparative_patterns_match_outperforms() -> None:
    assert _matches_any(COMPARATIVE_PATTERNS, "outperforms BERT")
    assert _matches_any(COMPARATIVE_PATTERNS, "exceeds the baseline")


def test_comparative_patterns_match_better_than() -> None:
    assert _matches_any(COMPARATIVE_PATTERNS, "more accurate than the baseline")
    assert _matches_any(COMPARATIVE_PATTERNS, "better than existing methods")


def test_theoretical_patterns_match_suggests_that() -> None:
    assert _matches_any(THEORETICAL_PATTERNS, "These results suggest that")
    assert _matches_any(THEORETICAL_PATTERNS, "evidence for the hypothesis")


def test_causal_patterns_match_leads_to() -> None:
    assert _matches_any(CAUSAL_PATTERNS, "Sleep deprivation leads to fatigue")
    assert _matches_any(CAUSAL_PATTERNS, "causes a reduction in")
    assert _matches_any(CAUSAL_PATTERNS, "results in higher error")


def test_iter_pattern_matches_emits_type_match_pairs() -> None:
    """The dispatcher walks every type and yields one entry per hit."""
    text = "Our model outperforms BERT (p < 0.001)."
    matches = list(iter_pattern_matches(text))
    types_hit = {claim_type for claim_type, _ in matches}
    assert ClaimType.STATISTICAL in types_hit
    assert ClaimType.COMPARATIVE in types_hit


def test_iter_pattern_matches_returns_empty_for_no_triggers() -> None:
    text = "The sky was blue and the birds sang sweetly."
    assert list(iter_pattern_matches(text)) == []


def test_confidence_for_returns_per_type_floor() -> None:
    assert confidence_for(ClaimType.STATISTICAL) == 0.9
    assert confidence_for(ClaimType.METHODOLOGICAL) == 0.85
    assert confidence_for(ClaimType.COMPARATIVE) == 0.85
    assert confidence_for(ClaimType.THEORETICAL) == 0.8
    assert confidence_for(ClaimType.CAUSAL) == 0.8


def test_confidence_for_returns_default_for_unmapped_type() -> None:
    """FACTUAL / EVALUATIVE aren't in the regex map; the fallback is 0.7."""
    assert confidence_for(ClaimType.FACTUAL) == 0.7
    assert confidence_for(ClaimType.EVALUATIVE) == 0.7


def test_improved_by_percent_hits_only_the_directional_rule() -> None:
    # The README and the _dedup_claims docstring once said this phrase hit two
    # rules. It hits one: the percent-change rule needs the number first.
    hits = [p for p in STATISTICAL_PATTERNS if re.search(p, "improved by 23%", re.IGNORECASE)]
    assert hits == [STATISTICAL_PATTERNS[1]]
    assert [m.group() for _, m in iter_pattern_matches("Accuracy improved by 23%.")] == [
        "improved by 23%"
    ]


def test_no_pattern_emits_factual_or_evaluative() -> None:
    """The README table lists only the five types that have patterns."""
    emitted = {claim_type for claim_type, _ in iter_pattern_matches(
        "A major advance: BERT is a language model defined by Devlin et al."
    )}
    assert ClaimType.FACTUAL not in emitted
    assert ClaimType.EVALUATIVE not in emitted
