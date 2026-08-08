"""
Pure retrieval-quality metrics — no I/O, no LLM, no heavy deps.

Every function takes:
  - retrieved: an ORDERED list of doc ids (best-ranked first), possibly with
    duplicates (same doc, different chunks). Order matters for rank metrics.
  - relevant:  a set of doc ids that SHOULD have been retrieved.

These isolate "did retrieval find the right documents?" from "did the LLM write
a good answer?" — the two things a RAG eval must never conflate. They are pure
functions so they can be unit-tested exhaustively without a vector store or an
API key.

Definitions (all standard IR metrics):
  hit@k       1 if at least one relevant doc appears in the top k, else 0.
  recall@k    fraction of the relevant set that appears in the top k.
  precision@k fraction of the top k that is relevant (dedup-aware).
  MRR         reciprocal rank of the FIRST relevant doc (1/rank); 0 if none.
"""
from typing import Iterable, List, Sequence, Set


def _dedup_preserve_order(items: Sequence[str]) -> List[str]:
    """Collapse repeated doc ids (many chunks -> one doc) keeping first-seen order."""
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if any relevant doc is in the top-k of retrieved, else 0.0."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top_k = _dedup_preserve_order(retrieved)[:k]
    return 1.0 if any(d in relevant for d in top_k) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant set found within the top-k retrieved docs."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top_k = set(_dedup_preserve_order(retrieved)[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved docs that are relevant."""
    relevant = set(relevant)
    top_k = _dedup_preserve_order(retrieved)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for d in top_k if d in relevant)
    return hits / len(top_k)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1/rank of the first relevant doc (rank is 1-based). 0.0 if none present."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    for i, doc in enumerate(_dedup_preserve_order(retrieved), start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def mean(values: Sequence[float]) -> float:
    """Plain arithmetic mean; 0.0 for an empty sequence (so aggregates never crash)."""
    values = list(values)
    return sum(values) / len(values) if values else 0.0
