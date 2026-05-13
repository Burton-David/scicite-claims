"""Public dataclasses: ``Claim`` and ``ClaimType``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class ClaimType(StrEnum):
    """The kind of evidence a claim asserts.

    Different types want different citations: a STATISTICAL claim cites
    a study; a METHODOLOGICAL claim cites the paper that introduced the
    method; a COMPARATIVE claim cites both compared baselines. Downstream
    citation finders can rank candidates differently per type.
    """

    STATISTICAL = "statistical"
    """Numeric claims: percentages, correlations, p-values, sample sizes."""

    METHODOLOGICAL = "methodological"
    """Claims about techniques, algorithms, or experimental approaches."""

    COMPARATIVE = "comparative"
    """X outperforms / is better than / matches Y."""

    THEORETICAL = "theoretical"
    """Claims about mechanisms, hypotheses, or theoretical implications."""

    CAUSAL = "causal"
    """X causes / leads to / results in Y."""

    FACTUAL = "factual"
    """Empirical claims that don't fit the above (named entities,
    historical facts, definitions)."""

    EVALUATIVE = "evaluative"
    """Quality / importance claims ('a major advance', 'state of the
    art'). Hardest to cite well."""


@dataclass(frozen=True, slots=True)
class Claim:
    """A claim extracted from text that may need a citation.

    Attributes:
        text: The verbatim claim as it appears in the source text.
        type: The ClaimType classification.
        confidence: Pattern strength, in ``[0, 1]``.
        context: Surrounding text window for disambiguation. Always a
            superset of ``text``.
        suggested_search_terms: Noun-chunk- and proper-noun-derived
            search terms from the context window. Empty for trivial
            claims with no surrounding nouns.
        start_char: Character offset of ``text`` in the original source.
        end_char: End character offset (exclusive).
        metadata: Optional pass-through metadata. Empty by default.
    """

    text: str
    type: ClaimType
    confidence: float
    context: str
    suggested_search_terms: tuple[str, ...] = ()
    start_char: int = 0
    end_char: int = 0
    metadata: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
