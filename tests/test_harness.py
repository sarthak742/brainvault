"""Offline integration tests for the evaluation harness (fakes, no faiss/keys)."""
from evaluation.harness import (
    evaluate_retrieval, evaluate_answers, compare_engines, retrieved_doc_order,
)


class FakeRetriever:
    """Maps a query to a preset ranked list of (score, chunk)."""
    def __init__(self, mapping):
        self._mapping = mapping  # query -> list[(score, {"source": name})]

    def retrieve(self, query, k=5):
        return self._mapping.get(query, [])[:k]


def _chunk(name):
    return {"source": f"/data/{name}", "text": "..."}


def test_retrieved_doc_order_extracts_basenames():
    scored = [(0.9, _chunk("a.pdf")), (0.5, _chunk("b.pdf"))]
    assert retrieved_doc_order(scored) == ["a.pdf", "b.pdf"]


def test_evaluate_retrieval_scores_recall_and_mrr():
    questions = [
        {"id": "q1", "question": "Q1", "relevant_doc_ids": ["a.pdf"]},
        {"id": "q2", "question": "Q2", "relevant_doc_ids": ["z.pdf"]},  # missed
        {"id": "q3", "question": "Q3", "relevant_doc_ids": []},          # skipped
    ]
    retr = FakeRetriever({
        "Q1": [(0.9, _chunk("a.pdf")), (0.4, _chunk("b.pdf"))],  # hit at rank 1
        "Q2": [(0.9, _chunk("b.pdf")), (0.4, _chunk("c.pdf"))],  # miss
    })
    rep = evaluate_retrieval(retr, questions, k=5)
    assert rep["scored_questions"] == 2          # q3 skipped
    assert rep["aggregate"]["recall@5"] == 0.5   # 1 hit, 1 miss
    assert rep["aggregate"]["mrr"] == 0.5        # (1.0 + 0.0) / 2


class FakeEngine:
    """Returns a preset {answer, grounded} per query."""
    def __init__(self, responses):
        self._responses = responses

    def generate_answer(self, query, **kw):
        return self._responses[query]


def _questions():
    return [
        {"id": "q1", "question": "real", "expect_grounded": True,
         "relevant_doc_ids": ["a.pdf"], "expected_keywords": ["paris"]},
        {"id": "q2", "question": "unanswerable", "expect_grounded": False,
         "relevant_doc_ids": [], "expected_keywords": []},
    ]


def test_evaluate_answers_classifies_tp_and_tn():
    engine = FakeEngine({
        "real": {"answer": "It is Paris [1].", "grounded": True},
        "unanswerable": {"answer": "I could not find relevant information.", "grounded": False},
    })
    rep = evaluate_answers(engine, _questions())
    assert rep["counts"]["TP"] == 1
    assert rep["counts"]["TN"] == 1
    assert rep["aggregate"]["grounding_accuracy"] == 1.0
    assert rep["aggregate"]["keyword_accuracy"] == 1.0  # "paris" present


def test_ab_comparison_detects_reflection_fixing_a_hallucination():
    # Baseline hallucinates on the unanswerable question (grounded=True => FP).
    baseline = FakeEngine({
        "real": {"answer": "It is Paris [1].", "grounded": True},
        "unanswerable": {"answer": "The King is Zog [1].", "grounded": True},  # FP!
    })
    # Reflection catches it and refuses (grounded=False => TN).
    reflection = FakeEngine({
        "real": {"answer": "It is Paris [1].", "grounded": True},
        "unanswerable": {"answer": "I could not find relevant information.", "grounded": False},
    })
    cmp = compare_engines({"baseline": baseline, "reflection": reflection}, _questions())
    # Reflection should improve refusal_accuracy and cut hallucination_rate.
    d = cmp["deltas_vs_baseline"]["reflection"]
    assert d["refusal_accuracy"] == 1.0        # 0.0 -> 1.0
    assert d["hallucination_rate"] == -0.5     # 0.5 -> 0.0
