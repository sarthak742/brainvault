"""
A/B benchmark: baseline (hybrid, no reflection) vs reflection-enabled engine,
over the same eval set and the same retriever/index.

This is the script that produces the "faithfulness/refusal went from X to Y"
number. It needs a built index + OPENROUTER_API_KEY (same as model_runner.py),
so run it locally, not in CI.

Usage:
    python -m evaluation.compare_reflection
    python -m evaluation.compare_reflection evaluation/eval_questions.json
"""
import json
import logging
import os
import sys
from pathlib import Path

from embeddings.embeddings import Embedder
from vectorstore.index import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retrieval import BM25Retriever
from llm.client import OpenRouterClient
from llm.answer_engine import AnswerEngine
from reflection.grader import RetrievalGrader
from reflection.query_rewriter import QueryRewriter
from reflection.critic import AnswerCritic
from evaluation.harness import (
    evaluate_retrieval,
    compare_engines,
    format_answer_comparison,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(message)s")
logger = logging.getLogger("AB_EVAL")
logger.setLevel(logging.INFO)


def main() -> None:
    root = Path(__file__).parent.parent
    data_dir = root / "vectorstore_data"
    q_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "evaluation" / "eval_questions.json"

    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY missing.")
        return
    if not (data_dir / "index.faiss").exists():
        logger.error("Index not found. Run build_index.py first.")
        return
    if not q_path.exists():
        logger.error(f"Eval set not found at {q_path}. See eval_questions.sample.json.")
        return

    questions = json.loads(q_path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(questions)} questions from {q_path.name}")

    # Shared retriever/index (hybrid: dense + BM25).
    store = VectorStore.load(str(data_dir / "index.faiss"), str(data_dir / "metadata.json"))
    chunks = list(store.metadata.values())
    embedder = Embedder()
    bm25 = BM25Retriever(chunks) if chunks else None
    retriever = Retriever(embedder, store, sparse_retriever=bm25)
    client = OpenRouterClient()

    # 1. Retrieval quality (LLM-free, cheap).
    logger.info("Scoring retrieval quality...")
    retr_report = evaluate_retrieval(retriever, questions, k=5)
    print("\n=== RETRIEVAL QUALITY (hybrid) ===")
    print(json.dumps(retr_report["aggregate"], indent=2))

    # 2. Answer A/B: baseline vs reflection.
    baseline = AnswerEngine(retriever, client)  # no reflection
    reflection = AnswerEngine(
        retriever,
        client,
        grader=RetrievalGrader(client),
        query_rewriter=QueryRewriter(client),
        critic=AnswerCritic(client),
    )

    logger.info("Running A/B answer evaluation (this makes LLM calls)...")
    comparison = compare_engines(
        {"baseline": baseline, "reflection": reflection}, questions
    )

    print("\n=== ANSWER QUALITY: baseline vs reflection ===")
    print(format_answer_comparison(comparison))
    print("\n=== DELTAS (reflection - baseline) ===")
    print(json.dumps(comparison["deltas_vs_baseline"], indent=2))

    out = root / "evaluation" / "ab_results.json"
    out.write_text(json.dumps(
        {"retrieval": retr_report, "answers": comparison}, indent=2
    ), encoding="utf-8")
    print(f"\nSaved full report to {out}")


if __name__ == "__main__":
    main()
