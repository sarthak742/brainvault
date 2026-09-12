"""
RetrievalGrader — the "did we even find the right pages?" critic.

This is the CRAG (Corrective Retrieval-Augmented Generation) idea: before
paying for a generation call, ask a cheap LLM pass whether the retrieved
chunks actually contain information relevant to the question. If none are
relevant, the caller can rewrite the query and search again instead of
feeding irrelevant context to the model (which is how RAG systems produce
confident-but-wrong answers).

Design choices:
- We grade all candidate chunks in ONE LLM call (a list of indices), not one
  call per chunk, to keep latency and cost bounded.
- We FAIL OPEN: if the grader errors or returns garbage, we treat every
  chunk as relevant and let the base pipeline proceed. A broken critic must
  never be worse than no critic.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from reflection._json import extract_json

# Same structural type the rest of the system passes around.
ChunkRecord = Dict[str, Any]
ScoredChunk = Tuple[float, ChunkRecord]

logger = logging.getLogger(__name__)


@dataclass
class GradeResult:
    """
    Outcome of grading a set of retrieved chunks.

    verdict:
        "sufficient"   -> at least one chunk is relevant; proceed.
        "insufficient" -> nothing relevant found; caller should rewrite+retry.
    relevant:
        The subset of input chunks the grader judged relevant, preserving the
        original (score, chunk) tuples and their order.
    reasoning:
        Short human-readable note (useful for logs, the audit trail, and the
        defense doc). Never shown verbatim to end users.
    graded:
        False when we fell open due to an error, so callers/logs can tell a
        real "sufficient" from a "we couldn't grade, assuming sufficient".
    """
    verdict: str
    relevant: List[ScoredChunk] = field(default_factory=list)
    reasoning: str = ""
    graded: bool = True


class RetrievalGrader:
    """Grades retrieved chunks for relevance to the query using the LLM."""

    SYSTEM_PROMPT = (
        "You are a strict relevance grader for a retrieval system.\n"
        "You are given a user QUESTION and a numbered list of CONTEXT blocks.\n"
        "Decide which blocks contain information that helps answer the "
        "question.\n"
        "Return ONLY a JSON object of the form:\n"
        '{\"relevant\": [1, 3], \"reasoning\": \"one short sentence\"}\n'
        "Rules:\n"
        "- 'relevant' is the list of block numbers (1-based) that are on-topic "
        "and useful.\n"
        "- If NO block is relevant, return an empty list: {\"relevant\": []}.\n"
        "- Judge relevance to the question, not writing quality.\n"
        "- Do not add any text outside the JSON object.\n"
    )

    def __init__(self, client: Any, max_block_chars: int = 500) -> None:
        """
        Args:
            client: Anything with .generate(prompt: str) -> str (the same
                    OpenRouterClient the answer engine uses).
            max_block_chars: Truncate each chunk to this many characters when
                    grading. The grader only needs the gist, and short blocks
                    keep the grading call cheap.
        """
        self.client = client
        self.max_block_chars = max_block_chars

    def grade(self, query: str, scored_chunks: List[ScoredChunk]) -> GradeResult:
        """Grade the candidate chunks; never raises."""
        if not scored_chunks:
            return GradeResult(verdict="insufficient", relevant=[], reasoning="No candidates.")

        prompt = self._build_prompt(query, scored_chunks)

        try:
            raw = self.client.generate(prompt)
        except Exception as e:  # noqa: BLE001 - fail open on ANY client error
            logger.warning(f"RetrievalGrader: LLM call failed ({e}). Failing open.")
            return GradeResult(
                verdict="sufficient",
                relevant=list(scored_chunks),
                reasoning="Grader unavailable; assuming relevant.",
                graded=False,
            )

        parsed = extract_json(raw)
        if not parsed or "relevant" not in parsed:
            logger.warning("RetrievalGrader: unparseable response. Failing open.")
            return GradeResult(
                verdict="sufficient",
                relevant=list(scored_chunks),
                reasoning="Grader response unparseable; assuming relevant.",
                graded=False,
            )

        indices = self._clean_indices(parsed.get("relevant"), len(scored_chunks))
        reasoning = str(parsed.get("reasoning", ""))[:300]

        if not indices:
            return GradeResult(verdict="insufficient", relevant=[], reasoning=reasoning)

        relevant = [scored_chunks[i - 1] for i in indices]
        return GradeResult(verdict="sufficient", relevant=relevant, reasoning=reasoning)

    # ---------------- internals ----------------

    def _build_prompt(self, query: str, scored_chunks: List[ScoredChunk]) -> str:
        blocks = []
        for i, (_, chunk) in enumerate(scored_chunks):
            text = str(chunk.get("text", "")).replace("\n", " ")[: self.max_block_chars]
            blocks.append(f"[{i + 1}] {text}")
        context = "\n\n".join(blocks)
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"QUESTION:\n{query}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "JSON:"
        )

    @staticmethod
    def _clean_indices(raw_indices: Any, n: int) -> List[int]:
        """Coerce the model's list into unique, in-range, ordered ints."""
        if not isinstance(raw_indices, list):
            return []
        seen = set()
        out: List[int] = []
        for item in raw_indices:
            try:
                idx = int(item)
            except (ValueError, TypeError):
                continue
            if 1 <= idx <= n and idx not in seen:
                seen.add(idx)
                out.append(idx)
        return out
