"""
AnswerCritic — the "is this answer actually backed by the sources?" critic.

The base engine validates that citations like [1] are PRESENT and in range.
That catches a lazy model (forgot to cite) but not a sneaky one (cited [2] on
a sentence that chunk 2 never supports). This critic closes that gap: it runs
a second LLM pass that reads the context and the answer together and judges
faithfulness.

verdict:
    "supported"   -> every claim is grounded in the context.
    "partial"     -> some claims are grounded, some are not.
    "unsupported" -> the answer makes claims the context does not back.

Fails open to "supported" (with graded=False) so an unavailable critic never
blocks a legitimate answer.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from reflection._json import extract_json

ChunkRecord = Dict[str, Any]
ScoredChunk = Tuple[float, ChunkRecord]

logger = logging.getLogger(__name__)

_VALID_VERDICTS = {"supported", "partial", "unsupported"}


@dataclass
class CritiqueResult:
    verdict: str          # supported | partial | unsupported
    reason: str = ""
    graded: bool = True   # False => we failed open, treat with caution

    @property
    def is_grounded(self) -> bool:
        """True only when the critic actively judged the answer supported."""
        return self.verdict == "supported"

    @property
    def needs_retry(self) -> bool:
        """A genuine (non-fail-open) unsupported verdict warrants regeneration."""
        return self.graded and self.verdict == "unsupported"


class AnswerCritic:
    """Judges whether an answer is faithful to its retrieved context."""

    SYSTEM_PROMPT = (
        "You are a strict faithfulness checker for a question-answering "
        "system.\n"
        "You are given the CONTEXT that was provided to an assistant and the "
        "ANSWER it produced.\n"
        "Decide whether every factual claim in the ANSWER is supported by the "
        "CONTEXT.\n"
        "Return ONLY a JSON object of the form:\n"
        '{\"verdict\": \"supported\", \"reason\": \"one short sentence\"}\n'
        "verdict must be exactly one of: supported, partial, unsupported.\n"
        "- supported: all claims are backed by the context.\n"
        "- partial: at least one claim is backed, but some are not.\n"
        "- unsupported: the answer asserts things the context does not "
        "contain.\n"
        "Refusals such as 'I could not find relevant information' count as "
        "supported.\n"
        "Do not add any text outside the JSON object.\n"
    )

    def __init__(self, client: Any, max_context_chars: int = 6000) -> None:
        self.client = client
        self.max_context_chars = max_context_chars

    def critique(
        self,
        query: str,
        answer: str,
        scored_chunks: List[ScoredChunk],
    ) -> CritiqueResult:
        """Judge faithfulness of `answer` against `scored_chunks`. Never raises."""
        if not answer or not answer.strip():
            return CritiqueResult(verdict="unsupported", reason="Empty answer.")

        prompt = self._build_prompt(query, answer, scored_chunks)

        try:
            raw = self.client.generate(prompt)
        except Exception as e:  # noqa: BLE001 - fail open
            logger.warning(f"AnswerCritic: LLM call failed ({e}). Failing open.")
            return CritiqueResult(verdict="supported", reason="Critic unavailable.", graded=False)

        parsed = extract_json(raw)
        if not parsed or "verdict" not in parsed:
            logger.warning("AnswerCritic: unparseable response. Failing open.")
            return CritiqueResult(verdict="supported", reason="Critic response unparseable.", graded=False)

        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in _VALID_VERDICTS:
            logger.warning(f"AnswerCritic: unknown verdict '{verdict}'. Failing open.")
            return CritiqueResult(verdict="supported", reason="Unknown verdict.", graded=False)

        reason = str(parsed.get("reason", ""))[:300]
        return CritiqueResult(verdict=verdict, reason=reason)

    # ---------------- internals ----------------

    def _build_prompt(self, query: str, answer: str, scored_chunks: List[ScoredChunk]) -> str:
        blocks = []
        running = 0
        for i, (_, chunk) in enumerate(scored_chunks):
            text = str(chunk.get("text", "")).replace("\n", " ")
            block = f"[{i + 1}] {text}"
            if running + len(block) > self.max_context_chars:
                break
            blocks.append(block)
            running += len(block)
        context = "\n\n".join(blocks)
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"QUESTION:\n{query}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "JSON:"
        )
