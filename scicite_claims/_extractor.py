"""``ClaimExtractor``: sync pattern + spaCy pipeline.

Pipeline per :meth:`ClaimExtractor.extract`:

1. Run ``text`` through spaCy once for tokenization + POS + sentence
   segmentation + NER. Cheap on the small ``en_core_web_sm`` model
   (~5ms / KB).
2. Apply the regex patterns from :mod:`._patterns` and emit a
   :class:`Claim` per match.
3. For each claim, derive ``suggested_search_terms`` from the spaCy
   noun chunks within the claim's *context window*. A one-word claim
   like "outperforms" has nothing to search, but its surrounding noun
   chunks do.
4. Drop spans that overlap an earlier kept span. Same-start ties go to
   the higher-confidence claim.
5. Sort by ``start_char`` so callers see claims in document order.

Async callers can wrap with ``asyncio.to_thread(extractor.extract, text)``.
spaCy is written in Cython and releases the GIL in parts of its pipeline,
so the thread offload keeps the event loop responsive.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Final

from scicite_claims._patterns import confidence_for, iter_pattern_matches
from scicite_claims._types import Claim, ClaimType

if TYPE_CHECKING:  # pragma: no cover
    from spacy.language import Language
    from spacy.tokens import Doc

_DEFAULT_MODEL: Final = "en_core_web_sm"
_DEFAULT_CONTEXT_CHARS: Final = 100
_MAX_SEARCH_TERMS: Final = 5
_HOW_TO_INSTALL: Final = (
    "Install spaCy and the small English model:\n"
    '  pip install "scicite-claims @ git+https://github.com/Burton-David/scicite-claims@v0.1.0"\n'
    "  python -m spacy download en_core_web_sm"
)


_LEADING_DETERMINERS: frozenset[str] = frozenset(
    {"a", "an", "the", "this", "that", "these", "those", "our", "their", "its"}
)
_STOP_CHUNKS: frozenset[str] = frozenset(
    {"we", "i", "they", "it", "this", "that", "these", "those", "you"}
)


_default_extractor: ClaimExtractor | None = None


class ClaimExtractor:
    """Sync claim extractor combining regex patterns and a spaCy pipeline.

    Args:
        model_name: spaCy model to load. Defaults to ``en_core_web_sm``.
            Install with ``python -m spacy download en_core_web_sm``.
        nlp: Optional pre-loaded :class:`spacy.language.Language`. Pass
            this to share one pipeline across many extractors and avoid
            paying the model-load cost more than once per process.
        context_chars: Character window around each pattern match used
            to scope noun-chunk search-term extraction. Defaults to 100.
    """

    name: str = "scicite-claims"

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL,
        nlp: Language | None = None,
        context_chars: int = _DEFAULT_CONTEXT_CHARS,
    ) -> None:
        if nlp is None:
            nlp = _load_spacy(model_name)
        self._nlp = nlp
        self._context = context_chars

    def extract(self, text: str) -> tuple[Claim, ...]:
        """Extract claims from ``text``.

        Args:
            text: Source text to analyze. Empty or whitespace-only input
                returns an empty tuple.

        Returns:
            Claims in document order. When pattern matches overlap, the
            one that starts first is kept; same-start ties go to the
            higher-confidence match.
        """
        if not text or not text.strip():
            return ()
        doc = self._nlp(text)
        raw_claims: list[Claim] = []
        for claim_type, match in iter_pattern_matches(text):
            start, end = match.start(), match.end()
            claim = Claim(
                text=match.group(),
                type=claim_type,
                confidence=confidence_for(claim_type),
                context=_make_context(text, start, end, self._context),
                start_char=start,
                end_char=end,
                suggested_search_terms=_search_terms_for(
                    doc, claim_type, start, end, self._context
                ),
            )
            raw_claims.append(claim)
        deduped = _dedup_claims(raw_claims)
        return tuple(sorted(deduped, key=lambda c: c.start_char))


def extract_claims(text: str) -> tuple[Claim, ...]:
    """One-shot convenience: load a default :class:`ClaimExtractor` lazily and extract.

    The default extractor is cached at module scope so repeated calls
    don't re-pay the spaCy model-load cost. Pass your own
    :class:`ClaimExtractor` instance if you want to customize the
    model or context window.

    Raises:
        RuntimeError: spaCy isn't installed or the model isn't downloaded.
            See ``python -m spacy download en_core_web_sm``.
    """
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = ClaimExtractor()
    return _default_extractor.extract(text)


def _load_spacy(model_name: str) -> Language:
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover, environment-specific
        raise RuntimeError(
            f"spaCy is required for scicite-claims. {_HOW_TO_INSTALL}"
        ) from exc
    try:
        return spacy.load(model_name)
    except OSError as exc:  # pragma: no cover
        raise RuntimeError(
            f"spaCy model {model_name!r} is not installed. {_HOW_TO_INSTALL}"
        ) from exc


def _make_context(text: str, start: int, end: int, window: int) -> str:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return text[lo:hi]


def _search_terms_for(
    doc: Doc, claim_type: ClaimType, start: int, end: int, window: int
) -> tuple[str, ...]:
    """Pull up to ``_MAX_SEARCH_TERMS`` noun-chunk-derived search terms.

    For METHODOLOGICAL claims, proper-noun tokens from the context
    window are also surfaced, since method names like "Adam" or "BERT"
    make distinctive query refinements.
    """
    lo = max(0, start - window)
    hi = min(len(doc.text), end + window)
    seen: dict[str, None] = {}

    for chunk in doc.noun_chunks:
        if chunk.start_char >= hi or chunk.end_char <= lo:
            continue
        clean = _clean_chunk(chunk.text)
        if clean and clean.lower() not in _STOP_CHUNKS:
            seen.setdefault(clean, None)

    if claim_type == ClaimType.METHODOLOGICAL:
        for token in doc:
            if token.idx < lo or token.idx + len(token.text) > hi:
                continue
            if token.pos_ == "PROPN" and token.is_alpha:
                seen.setdefault(token.text, None)

    return tuple(list(seen.keys())[:_MAX_SEARCH_TERMS])


def _clean_chunk(text: str) -> str:
    """Trim leading determiners ('the dominant model' → 'dominant model')."""
    parts = text.split()
    while parts and parts[0].lower() in _LEADING_DETERMINERS:
        parts.pop(0)
    return " ".join(parts).strip()


def _dedup_claims(claims: Iterable[Claim]) -> list[Claim]:
    """Sort by (start_char, -confidence) and drop spans that overlap a kept one.

    Two patterns can fire on overlapping text: "statistically significant
    correlation" hits both "statistically significant" and "significant
    correlation". ("improved by 23%" hits only the directional-change
    rule; the percent-change rule needs the number first.) Without dedup,
    one statement surfaces as two Claims. The earlier span wins an
    overlap. The ``-confidence`` key only breaks ties between spans that
    start at the same offset.
    """
    ordered = sorted(claims, key=lambda c: (c.start_char, -c.confidence))
    kept: list[Claim] = []
    last_end = -1
    for claim in ordered:
        if claim.start_char < last_end:
            continue
        kept.append(claim)
        last_end = claim.end_char
    return kept
