import logging
import re
from typing import List, Tuple, Optional, TypedDict, Any

from retrieval.retriever import Retriever
from llm.client import OpenRouterClient
from chunking.chunker import ChunkRecord

logger = logging.getLogger(__name__)


class AnswerResult(TypedDict, total=False):
    """
    Structured response from the Answer Engine.

    Attributes:
        answer: The generated text from the LLM.
        citations: List of (score, ChunkRecord) used as context.
        grounded: False if retrieval failed, scores were low, citation
            validation failed, OR the self-critique judged the answer
            unsupported by its sources.
        reflection: Optional audit trail of the self-reflective loop
            (retrieval grading, query rewrite, answer critique). Only present
            when reflection components are wired in. Safe to ignore.
    """
    answer: str
    citations: List[Tuple[float, ChunkRecord]]
    grounded: bool
    reflection: dict


class AnswerEngine:
    """
    Orchestrates the RAG pipeline with safety checks, structured output,
    and inline citation enforcement.

    Optionally runs a self-reflective loop:
      1. RETRIEVAL GRADING (CRAG): grade retrieved chunks for relevance before
         generating. If nothing is relevant, rewrite the query and retrieve
         once more.
      2. ANSWER CRITIQUE: after generating, check the answer is actually
         supported by the chunks. If not, regenerate once under a stricter
         instruction.

    All reflection components are optional. When none are supplied, behaviour
    is identical to the original retrieve -> generate -> validate pipeline.
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

    # Extra instruction injected when regenerating after a failed critique.
    RETRY_SYSTEM_SUFFIX = (
        "\nIMPORTANT: A previous attempt asserted claims NOT supported by the "
        "context. Answer again using ONLY facts present in the context blocks. "
        "If the context does not contain the answer, reply exactly with:\n"
        "I could not find relevant information in the provided documents.\n"
    )

    def __init__(
        self,
        retriever: Retriever,
        client: OpenRouterClient,
        system_prompt: Optional[str] = None,
        grader: Any = None,
        query_rewriter: Any = None,
        critic: Any = None,
        max_retrieval_attempts: int = 2,
        allow_regeneration: bool = True,
    ) -> None:
        """
        Args:
            retriever, client, system_prompt: as before.
            grader: optional RetrievalGrader (relevance grading + CRAG).
            query_rewriter: optional QueryRewriter (used only if grader present).
            critic: optional AnswerCritic (faithfulness self-critique).
            max_retrieval_attempts: total retrieval tries, including the first.
                2 => one original attempt + one rewrite-and-retry.
            allow_regeneration: if True, regenerate once when the critic judges
                the first answer unsupported.
        """
        self.retriever = retriever
        self.client = client
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        self.grader = grader
        self.query_rewriter = query_rewriter
        self.critic = critic
        self.max_retrieval_attempts = max(1, max_retrieval_attempts)
        self.allow_regeneration = allow_regeneration

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
        End-to-end RAG generation with optional self-reflection, safety guards,
        and citation validation.
        """
        reflection: dict = {"enabled": bool(self.grader or self.critic)}

        # 1. Retrieve (with optional grading + query-rewrite retry).
        working_results, retrieval_trail = self._retrieve_with_reflection(query, k)
        reflection["retrieval"] = retrieval_trail

        # 2. Filter by score.
        valid_results = [
            (score, chunk) for score, chunk in working_results if score >= score_threshold
        ]

        # 3. Fail fast if nothing usable.
        if not valid_results:
            logger.warning("No chunks met the score threshold. Refusing to answer.")
            return {
                "answer": (
                    "I could not find any relevant information in your documents "
                    "to answer this question."
                ),
                "citations": [],
                "grounded": False,
                "reflection": reflection,
            }

        # 4. Build context (with truncation guard).
        context_text, num_used_chunks = self._build_context(valid_results, max_context_chars)

        # 5-7. Generate, validate citations, and self-critique (with one retry).
        result = self._generate_and_critique(
            query=query,
            context_text=context_text,
            valid_results=valid_results,
            num_used_chunks=num_used_chunks,
            reflection=reflection,
        )
        return result

    # ---------------------------------------------------------
    # REFLECTION: RETRIEVAL SIDE (CRAG)
    # ---------------------------------------------------------

    def _retrieve_with_reflection(
        self, query: str, k: int
    ) -> Tuple[List[Tuple[float, ChunkRecord]], dict]:
        """
        Retrieve candidates. If a grader is wired in, grade relevance; if the
        first attempt is insufficient and a rewriter is available, rewrite the
        query and retrieve once more. Returns (chunks_to_use, audit_trail).
        """
        trail: dict = {"attempts": [], "rewritten_query": None, "graded": bool(self.grader)}

        raw = self.retriever.retrieve(query, k=k)

        if not self.grader:
            # Base behaviour: no grading, use raw candidates.
            trail["attempts"].append({"query": query, "candidates": len(raw), "verdict": "ungraded"})
            return raw, trail

        grade = self.grader.grade(query, raw)
        trail["attempts"].append(
            {"query": query, "candidates": len(raw),
             "verdict": grade.verdict, "relevant": len(grade.relevant),
             "reasoning": grade.reasoning}
        )

        if grade.verdict == "sufficient":
            return grade.relevant, trail

        # Insufficient: try a single rewrite-and-retry if possible.
        if self.query_rewriter and self.max_retrieval_attempts >= 2:
            new_query = self.query_rewriter.rewrite(query)
            if new_query and new_query != query:
                trail["rewritten_query"] = new_query
                raw2 = self.retriever.retrieve(new_query, k=k)
                grade2 = self.grader.grade(new_query, raw2)
                trail["attempts"].append(
                    {"query": new_query, "candidates": len(raw2),
                     "verdict": grade2.verdict, "relevant": len(grade2.relevant),
                     "reasoning": grade2.reasoning}
                )
                if grade2.verdict == "sufficient":
                    return grade2.relevant, trail
                # Second attempt also weak: fall back to best raw candidates
                # rather than dropping everything (let the threshold decide).
                return raw2 or raw, trail

        # No rewriter or retries exhausted: fall back to raw candidates.
        return raw, trail

    # ---------------------------------------------------------
    # REFLECTION: ANSWER SIDE (critique + optional regeneration)
    # ---------------------------------------------------------

    def _generate_and_critique(
        self,
        query: str,
        context_text: str,
        valid_results: List[Tuple[float, ChunkRecord]],
        num_used_chunks: int,
        reflection: dict,
    ) -> AnswerResult:
        """Generate an answer, validate citations, and optionally self-critique."""
        used_chunks = valid_results[:num_used_chunks]

        # First generation.
        final_prompt = self._build_prompt(query, context_text, self.system_prompt)
        try:
            answer_text = self.client.generate(final_prompt)
        except RuntimeError as e:
            logger.error(f"LLM Generation failed: {e}")
            return {
                "answer": "I'm sorry, I encountered an error while communicating with the AI model.",
                "citations": used_chunks,
                "grounded": False,
                "reflection": reflection,
            }

        is_valid, validated_answer = self._validate_citations(answer_text, num_used_chunks)

        # Answer critique (faithfulness). Optional.
        critique_trail: Optional[dict] = None
        if self.critic:
            critique = self.critic.critique(query, answer_text, used_chunks)
            critique_trail = {
                "verdict": critique.verdict,
                "reason": critique.reason,
                "graded": critique.graded,
                "regenerated": False,
            }

            # Regenerate once if the critic actively judged it unsupported.
            if critique.needs_retry and self.allow_regeneration:
                logger.info("AnswerCritic flagged unsupported answer. Regenerating once.")
                retry_prompt = self._build_prompt(
                    query, context_text, self.system_prompt + self.RETRY_SYSTEM_SUFFIX
                )
                try:
                    answer_text = self.client.generate(retry_prompt)
                    is_valid, validated_answer = self._validate_citations(
                        answer_text, num_used_chunks
                    )
                    critique = self.critic.critique(query, answer_text, used_chunks)
                    critique_trail = {
                        "verdict": critique.verdict,
                        "reason": critique.reason,
                        "graded": critique.graded,
                        "regenerated": True,
                    }
                except RuntimeError as e:
                    logger.error(f"Regeneration failed: {e}")

            reflection["critique"] = critique_trail

            # An answer is grounded only if citations are valid AND the critic
            # did not (confidently) judge it unsupported.
            grounded = is_valid and not critique.needs_retry
            citations = used_chunks if grounded else []
            return {
                "answer": validated_answer,
                "citations": citations,
                "grounded": grounded,
                "reflection": reflection,
            }

        # No critic: original behaviour (grounded == citation validity).
        return {
            "answer": validated_answer,
            "citations": used_chunks if is_valid else [],
            "grounded": is_valid,
            "reflection": reflection,
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
            block = f"[{i+1}] {text}"
            block_len = len(block)

            if current_len + block_len > max_chars:
                if chunks_used == 0:
                    context_parts.append(block)
                    chunks_used = 1
                logger.info(f"Context truncated. Used {chunks_used} chunks.")
                break

            context_parts.append(block)
            current_len += block_len + 2
            chunks_used += 1

        return "\n\n".join(context_parts), chunks_used

    def _build_prompt(self, query: str, context: str, system_prompt: str) -> str:
        """Construct the final prompt sent to the LLM."""
        return (
            f"{system_prompt}\n\n"
            "Answer the question using only the information below.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Answer:"
        )
