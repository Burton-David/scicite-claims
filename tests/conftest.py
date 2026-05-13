"""Shared fixtures: spaCy pipeline + ClaimExtractor instances."""

from __future__ import annotations

import pytest

from scicite_claims import ClaimExtractor


@pytest.fixture(scope="session")
def nlp():  # type: ignore[no-untyped-def]
    """Session-scoped spaCy pipeline so we only pay the load cost once."""
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:  # pragma: no cover — environment-specific
        pytest.skip(f"en_core_web_sm not installed: {exc}")


@pytest.fixture
def extractor(nlp) -> ClaimExtractor:  # type: ignore[no-untyped-def]
    """ClaimExtractor wired to the session-scoped spaCy pipeline."""
    return ClaimExtractor(nlp=nlp)
