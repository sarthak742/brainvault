"""
Evaluation harness for BrainVault.

Two independent evaluations, deliberately kept separate:

  1. evaluate_retrieval(retriever, questions, k)
     Isolates RETRIEVAL quality. Calls the retriever directly and scores the
     ranked doc list against each question's relevant_doc_ids using the pure
     metrics in metrics.py (recall@k, hit@k, MRR). No LLM involved, so this is
     cheap and deterministic.

  2. evaluate_answers(engine, questions)
     Isolates ANSWER quality. Calls the full engine and classifies each answer
     as a true/false positive/negative against expect_grounded, plus a keyword
     check. This is where the reflection layer shows up.

  3. compare_engines({name: engine}, questions)
     Runs evaluate_answers for several engine configs (e.g. "baseline" vs
     "reflection") and reports per-metric deltas — the A/B that proves whether
     the reflection layer actually helped.

Everything takes injected `retriever` / `engine` objects (duck-typed:
retriever needs .retrieve(query, k); engine needs .generate_answer(query)),
so the harness runs in tests with fakes and in production with the real thing.

Question schema (same as the existing eval_questions.json):
  {
    "id": str,
    "question": str,
    "expect_grounded": bool,
    "relevant_doc_ids": [str, ...],   # filenames
    "expected_keywords": [str, ...]
  }
"""
import logging
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict, List, Sequence

from evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank, mean

logger = logging.getLogger("EVAL_HARNESS")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _basename(path: str) -> str:
    """Filename from a path, tolerant of both / and \\ separators."""
    if not path:
        return ""
    # Handle whichever separator is present, cross-platform.
    name = PureWindowsPath(path).name if "\\" in path else PurePosixPath(path).name
    return name


def retrieved_doc_order(scored_chunks: Sequence) -> List[str]:
    """
    Turn a retriever's [(score, chunk), ...] output into an ordered list of doc
    filenames (best first). Chunks keep their retrieval order; docs may repeat.
    """
    order: List[str] = []
    for _, chunk in scored_chunks:
        name = _basename(str(chunk.get("source", "")))
        if name:
            order.append(name)
    return order


# ----------------------------------------------------------------------------
# 1. retrieval evaluation
# ----------------------------------------------------------------------------

def evaluate_retrieval(retriever: Any, questions: List[Dict], k: int = 5) -> Dict:
    """
    Score retrieval quality over questions that declare relevant_doc_ids.
    Questions with no relevant_doc_ids are skipped (nothing to score against).
    """
    per_question = []
    recalls, hits, rrs = [], [], []

    for item in questions:
        relevant = set(item.get("relevant_doc_ids", []))
        if not relevant:
            continue  # e.g. "should refuse" questions have no gold docs

        scored = retriever.retrieve(item["question"], k=k)
        order = retrieved_doc_order(scored)

        r = recall_at_k(order, relevant, k)
        h = hit_at_k(order, relevant, k)
        rr = reciprocal_rank(order, relevant)
        recalls.append(r); hits.append(h); rrs.append(rr)

        per_question.append({
            "id": item.get("id", "?"),
            "recall_at_k": round(r, 3),
            "hit_at_k": h,
            "reciprocal_rank": round(rr, 3),
            "retrieved": order[:k],
            "relevant": sorted(relevant),
        })

    return {
        "k": k,
        "scored_questions": len(recalls),
        "aggregate": {
            f"recall@{k}": round(mean(recalls), 3),
            f"hit@{k}": round(mean(hits), 3),
            "mrr": round(mean(rrs), 3),
        },
        "per_question": per_question,
    }


# ----------------------------------------------------------------------------
# 2. answer evaluation
# ----------------------------------------------------------------------------

def _keywords_met(answer: str, keywords: Sequence[str]) -> bool:
    lower = answer.lower()
    return all(kw.lower() in lower for kw in keywords)


def evaluate_answers(engine: Any, questions: List[Dict]) -> Dict:
    """
    Run the full engine and classify each answer.

    Confusion classes (grounding decision vs expectation):
      TP: expected grounded  AND grounded        (answered when it should)
      FN: expected grounded  AND not grounded    (refused/failed wrongly)
      TN: expected ungrounded AND not grounded   (correctly refused)
      FP: expected ungrounded AND grounded       (hallucinated / over-answered)
    """
    stats = {"TP": 0, "FN": 0, "TN": 0, "FP": 0,
             "keyword_pass": 0, "keyword_total": 0, "errors": 0}
    per_question = []

    for item in questions:
        q = item["question"]
        expect_grounded = item.get("expect_grounded", True)
        keywords = item.get("expected_keywords", [])

        try:
            result = engine.generate_answer(q)
        except Exception as e:  # noqa: BLE001 - one bad question shouldn't abort the run
            stats["errors"] += 1
            per_question.append({"id": item.get("id", "?"), "error": str(e)})
            continue

        grounded = bool(result.get("grounded"))
        answer = result.get("answer", "")

        if expect_grounded and grounded:
            cls = "TP"; stats["TP"] += 1
        elif expect_grounded and not grounded:
            cls = "FN"; stats["FN"] += 1
        elif not expect_grounded and not grounded:
            cls = "TN"; stats["TN"] += 1
        else:
            cls = "FP"; stats["FP"] += 1

        kw = None
        if expect_grounded and keywords:
            stats["keyword_total"] += 1
            kw = _keywords_met(answer, keywords)
            if kw:
                stats["keyword_pass"] += 1

        per_question.append({
            "id": item.get("id", "?"),
            "class": cls,
            "grounded": grounded,
            "keyword_match": kw,
            "answer_snippet": answer[:160],
        })

    tp, fn, tn, fp = stats["TP"], stats["FN"], stats["TN"], stats["FP"]
    total = tp + fn + tn + fp
    aggregate = {
        # Of questions that HAVE an answer, did we correctly decide to answer?
        "grounding_accuracy": round((tp + tn) / total, 3) if total else 0.0,
        # Of questions that should be answered, how many were?
        "answer_recall": round(tp / (tp + fn), 3) if (tp + fn) else 0.0,
        # Of questions that should be refused, how many were?
        "refusal_accuracy": round(tn / (tn + fp), 3) if (tn + fp) else 0.0,
        # How often did we over-answer (hallucination proxy)?
        "hallucination_rate": round(fp / total, 3) if total else 0.0,
        # Of answered+keyworded questions, keyword hit rate.
        "keyword_accuracy": round(stats["keyword_pass"] / stats["keyword_total"], 3)
        if stats["keyword_total"] else None,
    }
    return {"aggregate": aggregate, "counts": stats, "per_question": per_question}


# ----------------------------------------------------------------------------
# 3. A/B comparison
# ----------------------------------------------------------------------------

def compare_engines(named_engines: Dict[str, Any], questions: List[Dict]) -> Dict:
    """
    Run evaluate_answers for each named engine and compute deltas relative to
    the FIRST engine (treated as the baseline).
    """
    if not named_engines:
        raise ValueError("compare_engines needs at least one engine")

    reports = {name: evaluate_answers(eng, questions) for name, eng in named_engines.items()}
    names = list(named_engines.keys())
    baseline = names[0]
    base_agg = reports[baseline]["aggregate"]

    deltas = {}
    for name in names[1:]:
        agg = reports[name]["aggregate"]
        deltas[name] = {
            metric: (None if (agg.get(metric) is None or base_agg.get(metric) is None)
                     else round(agg[metric] - base_agg[metric], 3))
            for metric in base_agg
        }

    return {"baseline": baseline, "reports": reports, "deltas_vs_baseline": deltas}


def format_answer_comparison(comparison: Dict) -> str:
    """Render compare_engines() output as a plain-text table."""
    reports = comparison["reports"]
    names = list(reports.keys())
    metrics = list(reports[names[0]]["aggregate"].keys())

    col_w = 22
    header = "metric".ljust(col_w) + "".join(n.ljust(14) for n in names)
    lines = [header, "-" * len(header)]
    for m in metrics:
        row = m.ljust(col_w)
        for n in names:
            v = reports[n]["aggregate"].get(m)
            row += ("n/a" if v is None else f"{v}").ljust(14)
        lines.append(row)
    return "\n".join(lines)
