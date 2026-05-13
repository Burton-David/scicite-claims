"""End-to-end :class:`ClaimExtractor` tests using a real spaCy pipeline."""

from __future__ import annotations

from scicite_claims import ClaimExtractor, ClaimType, extract_claims

# ---- coverage per claim type ----


def test_extracts_percent_change_as_statistical(extractor: ClaimExtractor) -> None:
    text = "The new method achieved a 12% increase in accuracy."
    claims = list(extractor.extract(text))
    stat = [c for c in claims if c.type == ClaimType.STATISTICAL]
    assert stat, f"expected STATISTICAL; got {[c.type for c in claims]}"
    assert any("12%" in c.text for c in stat)


def test_extracts_p_value_as_statistical(extractor: ClaimExtractor) -> None:
    text = "The treatment effect was statistically significant (p < 0.001)."
    claims = list(extractor.extract(text))
    assert any(c.type == ClaimType.STATISTICAL for c in claims)


def test_extracts_correlation_as_statistical(extractor: ClaimExtractor) -> None:
    text = "We found a strong correlation of r = 0.85 between dosage and response."
    claims = list(extractor.extract(text))
    assert any(c.type == ClaimType.STATISTICAL for c in claims)


def test_extracts_methodological_claim(extractor: ClaimExtractor) -> None:
    text = "We employed a transformer architecture with multi-head attention."
    claims = list(extractor.extract(text))
    assert any(c.type == ClaimType.METHODOLOGICAL for c in claims)


def test_extracts_comparative_outperforms(extractor: ClaimExtractor) -> None:
    text = "Our model outperforms BERT on three benchmarks."
    claims = list(extractor.extract(text))
    assert any(c.type == ClaimType.COMPARATIVE for c in claims)


def test_extracts_comparative_better_than(extractor: ClaimExtractor) -> None:
    text = "The proposed approach is more accurate than existing baselines."
    claims = list(extractor.extract(text))
    assert any(c.type == ClaimType.COMPARATIVE for c in claims)


def test_extracts_causal_leads_to(extractor: ClaimExtractor) -> None:
    text = "Sleep deprivation leads to reduced cognitive performance."
    claims = list(extractor.extract(text))
    assert any(c.type == ClaimType.CAUSAL for c in claims)


def test_extracts_theoretical_suggests_that(extractor: ClaimExtractor) -> None:
    text = "These results suggest that attention mechanisms are universal."
    claims = list(extractor.extract(text))
    assert any(c.type == ClaimType.THEORETICAL for c in claims)


# ---- ordering, dedup, shape invariants ----


def test_claims_returned_in_document_order(extractor: ClaimExtractor) -> None:
    text = (
        "We propose a novel attention mechanism. "
        "It outperforms LSTMs by 23%. "
        "These findings suggest that self-attention is universally effective."
    )
    claims = list(extractor.extract(text))
    starts = [c.start_char for c in claims]
    assert starts == sorted(starts)


def test_overlapping_matches_are_deduplicated(extractor: ClaimExtractor) -> None:
    """Multiple regexes can fire on the same span — return one Claim per span."""
    from itertools import pairwise

    text = "Accuracy improved by 30%."
    claims = list(extractor.extract(text))
    stat_spans = [
        (c.start_char, c.end_char)
        for c in claims
        if c.type == ClaimType.STATISTICAL
    ]
    for (a_start, a_end), (b_start, b_end) in pairwise(stat_spans):
        assert a_end <= b_start, (
            f"overlap: ({a_start},{a_end}) and ({b_start},{b_end})"
        )


def test_extract_returns_empty_for_empty_text(extractor: ClaimExtractor) -> None:
    assert extractor.extract("") == ()
    assert extractor.extract("   \t\n  ") == ()


def test_extract_returns_empty_for_descriptive_prose(extractor: ClaimExtractor) -> None:
    """Sentences without claim-trigger patterns should yield zero claims."""
    text = "The sky was blue. The trees swayed gently in the breeze. A bird flew by."
    assert extractor.extract(text) == ()


def test_each_claim_has_non_empty_context(extractor: ClaimExtractor) -> None:
    text = "The treatment achieved a 47% improvement in remission rates."
    claims = list(extractor.extract(text))
    assert claims
    for c in claims:
        assert c.context
        assert len(c.context) >= len(c.text)


def test_each_claim_has_search_terms_when_nouns_present(
    extractor: ClaimExtractor,
) -> None:
    """Search terms drive downstream queries — must be non-empty for non-trivial claims."""
    text = "Multi-head attention outperforms LSTMs on machine translation tasks."
    claims = list(extractor.extract(text))
    assert claims
    assert any(c.suggested_search_terms for c in claims)


def test_start_end_chars_round_trip_to_original_text(
    extractor: ClaimExtractor,
) -> None:
    """start_char/end_char must index into the *original* text so callers
    can highlight the claim in place."""
    text = "We achieved a 12% improvement using gradient boosting."
    for c in extractor.extract(text):
        assert text[c.start_char : c.end_char] == c.text


def test_confidence_is_in_unit_interval(extractor: ClaimExtractor) -> None:
    text = (
        "The drug significantly reduced symptoms (p < 0.01) and outperformed placebo."
    )
    for c in extractor.extract(text):
        assert 0.0 <= c.confidence <= 1.0


def test_search_terms_capped_at_five(extractor: ClaimExtractor) -> None:
    """A claim with many surrounding nouns should still get at most 5 search terms."""
    text = (
        "Using gradient boosting and random forests and support vector machines and "
        "neural networks and decision trees, we improved accuracy by 23% on the test set."
    )
    for c in extractor.extract(text):
        assert len(c.suggested_search_terms) <= 5


def test_search_terms_filter_stop_pronouns(extractor: ClaimExtractor) -> None:
    """'we', 'they', 'it' etc. shouldn't appear as search terms — they aren't useful queries."""
    text = "We outperform their model significantly on every benchmark we tested."
    for c in extractor.extract(text):
        terms_lower = {t.lower() for t in c.suggested_search_terms}
        assert "we" not in terms_lower
        assert "they" not in terms_lower
        assert "it" not in terms_lower


def test_methodological_extracts_proper_noun_method_names(
    extractor: ClaimExtractor,
) -> None:
    """Method names like 'BERT' should surface as search terms for methodological claims."""
    text = "We employed BERT and Adam to optimize the loss."
    claims = [c for c in extractor.extract(text) if c.type == ClaimType.METHODOLOGICAL]
    if claims:
        terms = {t for c in claims for t in c.suggested_search_terms}
        assert "BERT" in terms or "Adam" in terms


# ---- API surface ----


def test_extractor_has_a_name(extractor: ClaimExtractor) -> None:
    assert extractor.name == "scicite-claims"


def test_extractor_accepts_custom_context_chars(nlp) -> None:  # type: ignore[no-untyped-def]
    """context_chars controls the window from which search terms are pulled."""
    narrow = ClaimExtractor(nlp=nlp, context_chars=10)
    wide = ClaimExtractor(nlp=nlp, context_chars=200)
    text = (
        "Gradient boosting yielded a 23% improvement on the held-out test split, "
        "decisively beating the random-forest baseline."
    )
    narrow_terms = {t for c in narrow.extract(text) for t in c.suggested_search_terms}
    wide_terms = {t for c in wide.extract(text) for t in c.suggested_search_terms}
    # Wider window should not strictly shrink the term pool.
    assert len(wide_terms) >= len(narrow_terms)


# ---- module-level extract_claims ----


def test_module_level_extract_claims_works() -> None:
    """The convenience function lazily creates a default extractor."""
    text = "Sleep deprivation leads to a 23% drop in reaction time."
    claims = extract_claims(text)
    assert claims
    types = {c.type for c in claims}
    assert ClaimType.CAUSAL in types or ClaimType.STATISTICAL in types


def test_module_level_extract_claims_returns_empty_for_no_triggers() -> None:
    assert extract_claims("The cat sat on the mat.") == ()


def test_module_level_extract_claims_caches_extractor() -> None:
    """Repeated calls reuse the cached extractor — no model-load each time."""
    import scicite_claims._extractor as ext_mod

    # Trigger first call to populate the cache.
    extract_claims("We achieved 12% gains.")
    first = ext_mod._default_extractor
    extract_claims("Sleep leads to fatigue.")
    second = ext_mod._default_extractor
    assert first is second
