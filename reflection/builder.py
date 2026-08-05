"""
Factory that assembles an AnswerEngine with the self-reflective layer wired in
according to config.yaml. Keeps app.py / main.py construction sites to a single
call and preserves backward compatibility (reflection off => plain engine).
"""
import logging
from typing import Any

from llm.answer_engine import AnswerEngine
from reflection.grader import RetrievalGrader
from reflection.query_rewriter import QueryRewriter
from reflection.critic import AnswerCritic
from config import (
    is_reflection_enabled,
    is_reflection_grade_retrieval,
    is_reflection_rewrite_query,
    is_reflection_critique_answer,
    get_reflection_max_retrieval_attempts,
    is_reflection_allow_regeneration,
)

logger = logging.getLogger(__name__)


def build_answer_engine(retriever: Any, client: Any) -> AnswerEngine:
    """
    Construct an AnswerEngine, attaching reflection components per config.

    The grader/rewriter/critic all reuse the SAME llm client instance the
    answer engine uses, so no extra credentials or config are required.
    """
    if not is_reflection_enabled():
        logger.info("Reflection disabled. Building plain AnswerEngine.")
        return AnswerEngine(retriever, client)

    grader = RetrievalGrader(client) if is_reflection_grade_retrieval() else None
    rewriter = (
        QueryRewriter(client)
        if (is_reflection_grade_retrieval() and is_reflection_rewrite_query())
        else None
    )
    critic = AnswerCritic(client) if is_reflection_critique_answer() else None

    logger.info(
        "Reflection enabled | grader=%s rewriter=%s critic=%s",
        bool(grader), bool(rewriter), bool(critic),
    )

    return AnswerEngine(
        retriever,
        client,
        grader=grader,
        query_rewriter=rewriter,
        critic=critic,
        max_retrieval_attempts=get_reflection_max_retrieval_attempts(),
        allow_regeneration=is_reflection_allow_regeneration(),
    )
