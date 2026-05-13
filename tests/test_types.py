"""Shape tests for :class:`Claim` and :class:`ClaimType`."""

from __future__ import annotations

import pytest

from scicite_claims import Claim, ClaimType


def test_claim_is_frozen() -> None:
    """Claim is a frozen dataclass — mutation must raise."""
    c = Claim(text="x", type=ClaimType.STATISTICAL, confidence=0.5, context="x")
    with pytest.raises((AttributeError, TypeError)):
        c.text = "y"  # type: ignore[misc]


def test_claim_defaults() -> None:
    """Optional fields take sensible defaults so the dataclass is callable
    with just the required positional fields."""
    c = Claim(text="x", type=ClaimType.STATISTICAL, confidence=0.5, context="x")
    assert c.start_char == 0
    assert c.end_char == 0
    assert c.suggested_search_terms == ()
    assert dict(c.metadata) == {}


def test_claim_metadata_is_read_only() -> None:
    c = Claim(text="x", type=ClaimType.STATISTICAL, confidence=0.5, context="x")
    with pytest.raises(TypeError):
        c.metadata["k"] = "v"  # type: ignore[index]


def test_claim_type_string_values() -> None:
    """ClaimType is a StrEnum so .value gives the string form."""
    assert ClaimType.STATISTICAL.value == "statistical"
    assert ClaimType.METHODOLOGICAL.value == "methodological"
    assert ClaimType.COMPARATIVE.value == "comparative"
    assert ClaimType.THEORETICAL.value == "theoretical"
    assert ClaimType.CAUSAL.value == "causal"
    assert ClaimType.FACTUAL.value == "factual"
    assert ClaimType.EVALUATIVE.value == "evaluative"


def test_claim_type_equals_string() -> None:
    """StrEnum: enum members compare equal to their string values."""
    assert ClaimType.STATISTICAL == "statistical"


def test_claim_type_can_be_constructed_from_string() -> None:
    assert ClaimType("statistical") == ClaimType.STATISTICAL
    with pytest.raises(ValueError):
        ClaimType("not-a-claim-type")
