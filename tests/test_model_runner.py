import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# -------------------------
# Fake AnswerEngine outputs
# -------------------------

FAKE_RESULT_GROUNDED = {
    "answer": "This is the answer [1].",
    "citations": [(0.9, {"source": "doc1.txt", "text": "answer text"})],
    "grounded": True,
}

FAKE_RESULT_UNGROUNDED = {
    "answer": "I could not find relevant information.",
    "citations": [],
    "grounded": False,
}


# -------------------------
# Tests
# -------------------------

def test_load_questions_creates_sample(tmp_path):
    from evaluation.model_runner import load_questions

    path = tmp_path / "eval_questions.json"

    data = load_questions(path)

    assert path.exists()
    assert isinstance(data, list)
    assert "question" in data[0]


def test_load_questions_valid_file(tmp_path):
    from evaluation.model_runner import load_questions

    path = tmp_path / "eval_questions.json"
    sample = [{"id": "q1", "question": "Test?", "expect_grounded": True}]
    path.write_text(json.dumps(sample))

    data = load_questions(path)

    assert data == sample


@patch("evaluation.model_runner.time.sleep", lambda x: None)
@patch("evaluation.model_runner.VectorStore.load")
@patch("evaluation.model_runner.Embedder")
@patch("evaluation.model_runner.Retriever")
@patch("evaluation.model_runner.OpenRouterClient")
@patch("evaluation.model_runner.AnswerEngine")
def test_run_benchmark_writes_results(
    mock_engine_cls,
    mock_client_cls,
    mock_retriever_cls,
    mock_embedder_cls,
    mock_vs_load,
    tmp_path,
    monkeypatch,
):
    """
    Full pipeline control-flow test:
    - no real API
    - no FAISS
    - verifies results.json structure
    """

    # ------------------
    # Fake env + paths
    # ------------------

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    root = tmp_path
    (root / "vectorstore_data").mkdir()
    (root / "vectorstore_data" / "index.faiss").write_text("x")
    (root / "vectorstore_data" / "metadata.json").write_text("{}")

    (root / "evaluation").mkdir()

    questions = [
        {
            "id": "q1",
            "question": "Known question",
            "expect_grounded": True,
            "relevant_doc_ids": ["doc1.txt"],
            "expected_keywords": ["answer"],
        },
        {
            "id": "q2",
            "question": "Unknown question",
            "expect_grounded": False,
            "relevant_doc_ids": [],
            "expected_keywords": [],
        },
    ]

    (root / "evaluation" / "eval_questions.json").write_text(json.dumps(questions))

    # ------------------
    # Mock pipeline
    # ------------------

    mock_vs_load.return_value = Mock()

    fake_engine = Mock()
    fake_engine.generate_answer.side_effect = [
        FAKE_RESULT_GROUNDED,
        FAKE_RESULT_UNGROUNDED,
    ]
    mock_engine_cls.return_value = fake_engine

    # ------------------
    # Patch module paths
    # ------------------

    import evaluation.model_runner as mr

    monkeypatch.setattr(mr, "__file__", str(root / "evaluation" / "model_runner.py"))
    monkeypatch.setattr(mr, "TARGET_MODELS", ["test-model"])  # CRITICAL FIX

    # ------------------
    # Run
    # ------------------

    mr.run_benchmark()

    # ------------------
    # Validate results
    # ------------------

    results_path = root / "evaluation" / "results.json"
    assert results_path.exists()

    results = json.loads(results_path.read_text())

    assert isinstance(results, dict)
    assert len(results) == 1

    for model, data in results.items():
        assert "summary" in data
        assert "details" in data

        summary = data["summary"]
        assert "metrics" in summary
        assert "counts" in summary

        counts = summary["counts"]
        for k in ["TP_strong", "TP_weak", "TN", "FP", "FN", "errors"]:
            assert k in counts

        details = data["details"]
        assert len(details) == 2

        assert "question" in details[0]
        assert "status" in details[0]
