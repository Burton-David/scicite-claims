# scicite-claims

[![CI](https://github.com/Burton-David/scicite-claims/actions/workflows/ci.yml/badge.svg)](https://github.com/Burton-David/scicite-claims/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Citation-aware claim extraction for scientific text. Regex patterns + spaCy noun-chunk analysis. No LLM, no API calls, no key required.

```python
from scicite_claims import extract_claims

text = (
    "Our model outperforms BERT on three benchmarks. "
    "We achieved a 23% improvement in F1 using gradient boosting."
)
for claim in extract_claims(text):
    print(f"[{claim.type.value:>15}] {claim.text!r}  search={claim.suggested_search_terms}")
```

```
[    comparative] 'outperforms'  search=('Our model', 'BERT', 'three benchmarks')
[   statistical] 'improved by 23%'  search=('23% improvement', 'F1', 'gradient boosting')
[ methodological] 'We achieved'  search=('23% improvement', 'F1', 'gradient boosting')
```

## What it extracts

Seven `ClaimType` categories, each tuned to a different citation pattern:

| Type | Example | Citation strategy |
|---|---|---|
| `STATISTICAL` | "p < 0.001", "23% improvement", "n=512" | Cite a study reporting the number. |
| `METHODOLOGICAL` | "we employed gradient boosting" | Cite the paper that introduced the method. |
| `COMPARATIVE` | "outperforms BERT" | Cite both compared baselines. |
| `THEORETICAL` | "results suggest that X is universal" | Cite supporting / contradicting theory. |
| `CAUSAL` | "sleep deprivation leads to fatigue" | Cite the original causal study. |
| `FACTUAL` | Named entities, definitions | Cite reference materials. |
| `EVALUATIVE` | "a major advance" | Hardest — cite reviews or position pieces. |

## Install

```bash
pip install scicite-claims
python -m spacy download en_core_web_sm
```

Python 3.10+. Runtime dependency: [spaCy](https://spacy.io/) 3.7+. The `en_core_web_sm` model download is a one-time post-install step (~12MB) — spaCy doesn't bundle models with the package, so the install isn't strictly declarative.

## Usage

### One-shot

```python
from scicite_claims import extract_claims
claims = extract_claims("Sleep deprivation leads to a 23% drop in reaction time.")
```

`extract_claims` caches a default extractor at module scope so repeated calls don't reload spaCy.

### Reusable extractor

```python
from scicite_claims import ClaimExtractor

extractor = ClaimExtractor()
for chunk in chunks_of_text:
    for claim in extractor.extract(chunk):
        ...
```

One pre-loaded spaCy pipeline, reused across calls. Use this in batch jobs.

### Inject your own spaCy pipeline

```python
import spacy
from scicite_claims import ClaimExtractor

nlp = spacy.load("en_core_web_lg")  # bigger model with vectors
extractor = ClaimExtractor(nlp=nlp)
```

Or share one `nlp` across multiple extractor configurations.

### Async callers

The pipeline is synchronous and CPU-bound. spaCy releases the GIL during its rust-backed work, so async callers can offload:

```python
import asyncio
from scicite_claims import ClaimExtractor

extractor = ClaimExtractor()

async def extract_async(text: str):
    return await asyncio.to_thread(extractor.extract, text)
```

## Output shape

`Claim` is a frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class Claim:
    text: str                                # the matched span verbatim
    type: ClaimType                          # one of STATISTICAL, METHODOLOGICAL, ...
    confidence: float                        # in [0, 1]; per-type floor
    context: str                             # ±100 chars around the match
    suggested_search_terms: tuple[str, ...]  # noun chunks from the context window
    start_char: int                          # offset into the original text
    end_char: int
    metadata: Mapping[str, str]              # empty by default; pass-through
```

`start_char` / `end_char` index into the original text — verify with `text[c.start_char:c.end_char] == c.text` for any claim.

## Patterns

Regexes live in `scicite_claims/_patterns.py`. They're conservative on purpose: false positives dilute downstream citation suggestions, and a missed claim is recoverable while a wrong-typed claim leads citation finders to bad candidates.

Two patterns can fire on the same span ("improved by 23%" matches both percent-change and directional-change rules). The extractor deduplicates by overlapping spans, keeping the higher-confidence one.

## Development

```bash
git clone https://github.com/Burton-David/scicite-claims
cd scicite-claims
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
pytest
ruff check scicite_claims tests
mypy scicite_claims
```

## Credits

Extracted from [research-mcp](https://github.com/Burton-David/ResearchAssistantMCP)'s spaCy claim extractor. The pattern library and dedup invariants were settled there first.

## License

MIT — see `LICENSE`.
