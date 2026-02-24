import logging
import re
from typing import List, Tuple, Optional, TypedDict

from retrieval.retriever import Retriever
from llm.client import OpenRouterClient
from chunking.chunker import ChunkRecord

logger = logging.getLogger(__name__)


class AnswerResult(TypedDict):
    """
    Structured response from the Answer Engine.

    Attributes:
        answer: The generated text from the LLM.
        citations: List of (score, ChunkRecord) used as context.
        grounded: False if retrieval failed, scores were low, or validation failed.
    """
    answer: str
    citations: List[Tuple[float, ChunkRecord]]
    grounded: bool


class AnswerEngine:
    """
    Orchestrates the RAG pipeline with safety checks, structured output,
    and inline citation enforcement.
    """

    # STRICT, MODEL-AGNOSTIC PROMPT (Gemini / GLM compliant)
    DEFAULT_SYSTEM_PROMPT = (
        "You are an automated question-answering system.\n"
        "You MUST follow these rules:\n"
        "1. Use ONLY the information from the provided context blocks.\n"
        "2. Every factual sentence MUST end with a citation like [1], [2], etc.\n"
        "3. Do NOT add explanations, introductions, or conclusions.\n"
        "4. Do NOT mention the words 'context', 'documents', or 'sources'.\n"
        "5. If the answer is not present, reply exactly with:\n"
        "   I could not find relevant information in the provided documents.\n"
    )

    def __init__(
        self,
        retriever: Retriever,
        client: OpenRouterClient,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.retriever = retriever
        self.client = client
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def generate_answer(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.25,
        max_context_chars: int = 10_000,
    ) -> AnswerResult:
        """
        End-to-end RAG generation with safety guards and citation validation.
        """

        # 1. Retrieve
        logger.info(f"Retrieving context for: {query}")
        raw_results = self.retriever.retrieve(query, k=k)

        # 2. Filter by score
        valid_results = [
            (score, chunk) for score, chunk in raw_results if score >= score_threshold
        ]

        # 3. Fail fast if nothing usable
        if not valid_results:
            logger.warning("No chunks met the score threshold. Refusing to answer.")
            return {
                "answer": (
                    "I could not find any relevant information in your documents "
                    "to answer this question."
                ),
                "citations": [],
                "grounded": False,
            }

        # 4. Build context (with truncation guard)
        context_text, num_used_chunks = self._build_context(
            valid_results, max_context_chars
        )

        # 5. Build final prompt
        final_prompt = self._build_prompt(query, context_text)

        # 6. Call LLM
        try:
            answer_text = self.client.generate(final_prompt)

            # 7. Citation validation
            is_valid, validated_answer = self._validate_citations(
                answer_text, num_used_chunks
            )

            return {
                "answer": validated_answer,
                "citations": valid_results[:num_used_chunks] if is_valid else [],
                "grounded": is_valid,
            }

        except RuntimeError as e:
            logger.error(f"LLM Generation failed: {e}")
            return {
                "answer": "I'm sorry, I encountered an error while communicating with the AI model.",
                "citations": valid_results[:num_used_chunks],
                "grounded": False,
            }

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------

    def _validate_citations(self, answer_text: str, num_chunks: int) -> Tuple[bool, str]:
        """
        Check if answer contains valid citations [1]..[N].
        Returns (is_valid, possibly_modified_text).
        """

        matches = re.findall(r"\[(\d+)\]", answer_text)
        found_indices = [int(m) for m in matches]

        if not found_indices:
            warning = "⚠️ Answer may be unreliable due to missing citations.\n"
            return False, warning + answer_text

        valid_range = range(1, num_chunks + 1)
        if any(idx not in valid_range for idx in found_indices):
            warning = "⚠️ Answer may be unreliable due to invalid citations.\n"
            return False, warning + answer_text

        return True, answer_text

    def _build_context(
        self,
        scored_chunks: List[Tuple[float, ChunkRecord]],
        max_chars: int,
    ) -> Tuple[str, int]:
        """
        Format retrieved chunks into a single string.
        Returns (formatted_text, count_of_chunks_used).
        """

        context_parts: List[str] = []
        current_len = 0
        chunks_used = 0

        for i, (_, chunk) in enumerate(scored_chunks):
            text = chunk.get("text", "").replace("\n", " ")

            # Minimal, model-friendly format
            block = f"[{i+1}] {text}"
            block_len = len(block)

            if current_len + block_len > max_chars:
                # Always include at least one chunk
                if chunks_used == 0:
                    context_parts.append(block)
                    chunks_used = 1
                logger.info(f"Context truncated. Used {chunks_used} chunks.")
                break

            context_parts.append(block)
            current_len += block_len + 2
            chunks_used += 1

        return "\n\n".join(context_parts), chunks_used

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Construct the final prompt sent to the LLM.
        """

        return (
            f"{self.system_prompt}\n\n"
            "Answer the question using only the information below.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Answer:"
        )
