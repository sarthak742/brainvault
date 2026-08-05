"""
Self-reflective RAG layer.

Adds three cooperating critics on top of the base retrieve->generate pipeline:

- RetrievalGrader : judges whether retrieved chunks are actually relevant
                    to the question BEFORE we spend a generation call (CRAG).
- QueryRewriter   : rewrites a weak/vague question into a stronger search
                    query so a second retrieval attempt can recover.
- AnswerCritic    : judges whether the generated answer is actually supported
                    by the retrieved chunks (faithfulness), catching the
                    "cited but fabricated" failure the citation regex cannot.

All three are pure add-ons: if they are not wired in, AnswerEngine behaves
exactly as before. Every LLM call is wrapped so that a failure of the
reflection layer degrades gracefully to the base behaviour instead of
breaking the user's request (fail-open).
"""

from reflection.grader import RetrievalGrader, GradeResult
from reflection.query_rewriter import QueryRewriter
from reflection.critic import AnswerCritic, CritiqueResult

__all__ = [
    "RetrievalGrader",
    "GradeResult",
    "QueryRewriter",
    "AnswerCritic",
    "CritiqueResult",
]
