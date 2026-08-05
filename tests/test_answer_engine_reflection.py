"""
Integration tests for AnswerEngine's self-reflective loop, using fake
retriever + scripted client (no faiss / no network).
"""
import pytest
from llm.answer_engine import AnswerEngine


class FakeRetriever:
    """Returns a preset list of (score, chunk) per call; records queries."""
    def __init__(self, *result_batches):
        self._batches = list(result_batches)
        self.queries = []

    def retrieve(self, query, k=5):
        self.queries.append(query)
        if len(self._batches) == 1:
            return self._batches[0]
        return self._batches.pop(0) if self._batches else []


class ScriptedClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self._responses.pop(0)


def chunk(text, score=0.9):
    return (score, {"text": text, "source": "doc.pdf"})


def test_baseline_no_reflection_matches_old_behaviour():
    retr = FakeRetriever([chunk("Paris is the capital of France.")])
    client = ScriptedClient(["The capital is Paris [1]."])
    engine = AnswerEngine(retr, client)  # no reflection components
    res = engine.generate_answer("capital of France?", score_threshold=0.25)
    assert res["grounded"] is True
    assert "Paris" in res["answer"]
    # Exactly one LLM call when no reflection.
    assert len(client.prompts) == 1


def test_crag_rewrites_query_when_first_retrieval_irrelevant():
    from reflection.grader import RetrievalGrader
    from reflection.query_rewriter import QueryRewriter

    # First retrieval irrelevant, second (after rewrite) relevant.
    retr = FakeRetriever(
        [chunk("Unrelated text about cooking.")],
        [chunk("The Eiffel Tower is 330 metres tall.")],
    )
    client = ScriptedClient([
        '{"relevant": []}',                       # grade attempt 1 -> insufficient
        "height of eiffel tower metres",          # rewrite
        '{"relevant": [1]}',                       # grade attempt 2 -> sufficient
        "It is 330 metres tall [1].",             # generation
        '{"verdict": "supported", "reason": "ok"}',  # critique
    ])
    grader = RetrievalGrader(client)
    rewriter = QueryRewriter(client)
    from reflection.critic import AnswerCritic
    critic = AnswerCritic(client)
    engine = AnswerEngine(retr, client, grader=grader, query_rewriter=rewriter, critic=critic)

    res = engine.generate_answer("how tall?", score_threshold=0.25)
    assert res["grounded"] is True
    assert res["reflection"]["retrieval"]["rewritten_query"] == "height of eiffel tower metres"
    assert len(retr.queries) == 2  # retrieved twice


def test_critic_triggers_one_regeneration_on_unsupported():
    from reflection.critic import AnswerCritic

    retr = FakeRetriever([chunk("Water boils at 100C at sea level.")])
    client = ScriptedClient([
        "Water boils at 90C [1].",                     # first (wrong) generation
        '{"verdict": "unsupported", "reason": "90 wrong"}',  # critique -> retry
        "Water boils at 100C [1].",                    # regeneration
        '{"verdict": "supported", "reason": "fixed"}',  # re-critique
    ])
    critic = AnswerCritic(client)
    engine = AnswerEngine(retr, client, critic=critic)

    res = engine.generate_answer("boiling point?", score_threshold=0.25)
    assert res["grounded"] is True
    assert res["reflection"]["critique"]["regenerated"] is True
    assert "100C" in res["answer"]


def test_unsupported_without_regeneration_marks_ungrounded():
    from reflection.critic import AnswerCritic

    retr = FakeRetriever([chunk("Water boils at 100C.")])
    client = ScriptedClient([
        "Water boils at 90C [1].",
        '{"verdict": "unsupported", "reason": "wrong"}',
    ])
    critic = AnswerCritic(client)
    engine = AnswerEngine(retr, client, critic=critic, allow_regeneration=False)

    res = engine.generate_answer("boiling point?", score_threshold=0.25)
    assert res["grounded"] is False
    assert res["citations"] == []
