"""scicite-claims — citation-aware claim extraction for scientific text.

Quick start::

    from scicite_claims import extract_claims

    text = (
        "Our model outperforms BERT on three benchmarks. "
        "We achieved a 23% improvement in F1 using gradient boosting."
    )
    for claim in extract_claims(text):
        print(claim.type.value, "→", claim.text)

Requires the spaCy ``en_core_web_sm`` model. After installing the package,
run::

    python -m spacy download en_core_web_sm
"""

from scicite_claims._extractor import ClaimExtractor, extract_claims
from scicite_claims._types import Claim, ClaimType

__all__ = [
    "Claim",
    "ClaimExtractor",
    "ClaimType",
    "extract_claims",
]

__version__ = "0.1.0"
