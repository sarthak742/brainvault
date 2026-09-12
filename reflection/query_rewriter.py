"""
QueryRewriter — turns a weak question into a stronger search query.

When the grader reports that retrieval found nothing relevant, the problem is
often the *query*, not the corpus: it may be vague, use different vocabulary
than the documents, or bundle several questions together. This rewrites the
question into a keyword-richer, more explicit search query and lets the caller
retrieve a second time.

Fails open: on any error it returns the original query unchanged, so a broken
rewriter simply means "no second chance", never a crash.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrites a user question into a better retrieval query via the LLM."""

    SYSTEM_PROMPT = (
        "You rewrite search queries for a document retrieval system.\n"
        "Given a user QUESTION, produce a single improved search query that:\n"
        "- keeps the original intent,\n"
        "- makes implicit terms explicit,\n"
        "- adds likely synonyms or domain keywords,\n"
        "- removes conversational filler.\n"
        "Return ONLY the rewritten query as plain text. No quotes, no "
        "explanation, no preamble.\n"
    )

    def __init__(self, client: Any, max_len: int = 300) -> None:
        self.client = client
        self.max_len = max_len

    def rewrite(self, query: str) -> str:
        """Return an improved query, or the original on any failure."""
        if not query or not query.strip():
            return query

        prompt = f"{self.SYSTEM_PROMPT}\n\nQUESTION:\n{query}\n\nRewritten query:"

        try:
            raw = self.client.generate(prompt)
        except Exception as e:  # noqa: BLE001 - fail open
            logger.warning(f"QueryRewriter: LLM call failed ({e}). Using original query.")
            return query

        cleaned = self._clean(raw)
        if not cleaned:
            return query

        # Guard against the model "rewriting" into an essay.
        if len(cleaned) > self.max_len:
            cleaned = cleaned[: self.max_len].strip()

        logger.info(f"QueryRewriter: '{query}' -> '{cleaned}'")
        return cleaned

    @staticmethod
    def _clean(text: str) -> str:
        if not text:
            return ""
        line = text.strip().splitlines()[0].strip() if text.strip() else ""
        # Strip wrapping quotes the model sometimes adds.
        return line.strip('"').strip("'").strip()
