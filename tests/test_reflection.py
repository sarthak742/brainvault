"""
Unit tests for the self-reflective RAG layer.

These use a scripted fake LLM client (no network, no heavy deps) so the
grader/rewriter/critic and the AnswerEngine loop can be tested deterministically.
"""
import pytest

from reflection.grader import RetrievalGrader, GradeResult
from reflection.query_rewriter import QueryRewriter
from reflection.critic import AnswerCritic, CritiqueResult
from reflection._json import extract_json


class ScriptedClient:
    """Returns queued responses in order; records prompts it received."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("ScriptedClient ran out of responses")
        return self._responses.pop(0)


class BoomClient:
    """Always raises, to exercise fail-open paths."""
    def generate(self, prompt):
        raise RuntimeError("simulated LLM outage")


def _chunks(*texts):
    return [(1.0, {"text": t, "source": "doc.pdf"}) for t in texts]


# ---------------- _json ----------------

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}

def test_extract_json_wrapped_in_prose():
    assert extract_json('Sure! {"verdict": "supported"} done') == {"verdict": "supported"}

def test_extract_json_garbage_returns_none():
    assert extract_json("no json here") is None


# ---------------- RetrievalGrader ----------------

def test_grader_keeps_relevant_subset():
    client = ScriptedClient(['{"relevant": [1, 3], "reasoning": "ok"}'])
    g = RetrievalGrader(client)
    res = g.grade("q", _chunks("a", "b", "c"))
    assert res.verdict == "sufficient"
    assert [c[1]["text"] for c in res.relevant] == ["a", "c"]

def test_grader_none_relevant_is_insufficient():
    client = ScriptedClient(['{"relevant": []}'])
    res = RetrievalGrader(client).grade("q", _chunks("a", "b"))
    assert res.verdict == "insufficient"
    assert res.relevant == []

def test_grader_out_of_range_indices_ignored():
    client = ScriptedClient(['{"relevant": [1, 9, 0, "x"]}'])
    res = RetrievalGrader(client).grade("q", _chunks("a", "b"))
    assert [c[1]["text"] for c in res.relevant] == ["a"]

def test_grader_fails_open_on_client_error():
    res = RetrievalGrader(BoomClient()).grade("q", _chunks("a", "b"))
    assert res.verdict == "sufficient"
    assert res.graded is False
    assert len(res.relevant) == 2

def test_grader_fails_open_on_bad_json():
    res = RetrievalGrader(ScriptedClient(["not json"])).grade("q", _chunks("a"))
    assert res.verdict == "sufficient"
    assert res.graded is False


# ---------------- QueryRewriter ----------------

def test_rewriter_returns_cleaned_query():
    client = ScriptedClient(['"machine learning model training cost"'])
    out = QueryRewriter(client).rewrite("how much to train")
    assert out == "machine learning model training cost"

def test_rewriter_fails_open_to_original():
    out = QueryRewriter(BoomClient()).rewrite("original q")
    assert out == "original q"

def test_rewriter_empty_stays_empty():
    assert QueryRewriter(ScriptedClient([])).rewrite("   ") == "   "


# ---------------- AnswerCritic ----------------

def test_critic_supported():
    c = AnswerCritic(ScriptedClient(['{"verdict": "supported", "reason": "ok"}']))
    res = c.critique("q", "answer [1]", _chunks("a"))
    assert res.is_grounded is True
    assert res.needs_retry is False

def test_critic_unsupported_needs_retry():
    c = AnswerCritic(ScriptedClient(['{"verdict": "unsupported", "reason": "made up"}']))
    res = c.critique("q", "answer", _chunks("a"))
    assert res.verdict == "unsupported"
    assert res.needs_retry is True

def test_critic_fails_open_to_supported():
    res = AnswerCritic(BoomClient()).critique("q", "answer", _chunks("a"))
    assert res.verdict == "supported"
    assert res.graded is False
    assert res.needs_retry is False  # fail-open must NOT trigger a retry

def test_critic_empty_answer_unsupported():
    res = AnswerCritic(ScriptedClient([])).critique("q", "", _chunks("a"))
    assert res.verdict == "unsupported"
