"""
Retrieval evaluation harness for BrainVault.

Measures retrieval quality WITHOUT calling the LLM. This is deliberate:
retrieval is the part you can tune, LLM calls cost money and add noise, and
if the right chunk is never retrieved the answer cannot be correct no matter
how good the model is.

Usage:
    python eval/run_eval.py            # evaluate at the config alpha
    python eval/run_eval.py --sweep    # baselines + alpha sweep 0.0 -> 1.0
    python eval/run_eval.py --k 10     # change top-k
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_vectorstore_dir, get_hybrid_alpha, get_grounding_threshold
from embeddings.embeddings import Embedder
from vectorstore.index import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retrieval import BM25Retriever

EVAL_FILE = PROJECT_ROOT / "eval" / "eval_questions.json"
RESULTS_FILE = PROJECT_ROOT / "eval" / "results.json"
GROUNDING_THRESHOLD = get_grounding_threshold()


# ------------------------------------------------------------ load once

class Corpus:
    """
    Loads the index and embedding model ONCE and hands out retrievers.

    Without this, an 11-point alpha sweep reloads the sentence-transformer
    model 11 times, which dominates the runtime and measures nothing useful.
    """

    def __init__(self):
        vs_dir = get_vectorstore_dir()
        index_path = vs_dir / "index.faiss"
        meta_path = vs_dir / "metadata.json"

        if not index_path.exists():
            sys.exit(f"No index at {index_path}. Run `python build_index.py` first.")

        self.store = VectorStore.load(str(index_path), str(meta_path))
        self.chunks = list(self.store.metadata.values())
        self.embedder = Embedder()
        self.bm25 = BM25Retriever(self.chunks)
        print(f"Loaded {len(self.chunks)} chunks from index.")

    def retriever(self, mode="hybrid", alpha=None):
        """mode: 'hybrid' | 'dense' | 'sparse'"""
        if mode == "sparse":
            return self.bm25
        if mode == "dense":
            return Retriever(self.embedder, self.store, sparse_retriever=None)
        return Retriever(self.embedder, self.store,
                         sparse_retriever=self.bm25, alpha=alpha)


# ------------------------------------------------------------ matching

def chunk_matches(chunk, expected_source, expected_pages):
    """
    A chunk is a hit if it is from the expected document AND overlaps at least
    one expected page.

    Overlap rather than equality: chunks span page ranges, so a chunk covering
    pages 3-4 should count for an answer that lives on page 4.
    """
    source = Path(str(chunk.get("source", ""))).name.lower()
    if expected_source.lower() not in source:
        return False

    if not expected_pages:
        return True  # document-level match only

    start = chunk.get("start_page")
    if start is None:
        return True  # no page metadata - don't penalise

    end = chunk.get("end_page")
    end = start if end is None else end
    return any(start <= p <= end for p in expected_pages)


# ------------------------------------------------------------ metrics

def evaluate(retriever, questions, k=5):
    """
    hit_rate -- fraction of questions where a correct chunk is in the top-k.
                The headline number.
    mrr      -- mean reciprocal rank. Rewards ranking the right chunk FIRST.
                hit_rate 0.90 with mrr 0.35 means correct chunks are sitting
                at position 4-5, where the LLM may well ignore them.
    """
    per_type = defaultdict(lambda: {"n": 0, "hits": 0, "rr": 0.0})
    failures = []

    for q in questions:
        qtype = q.get("type", "untyped")
        if "FILL IN" in q["question"]:
            continue  # skip unfinished placeholders

        results = retriever.retrieve(q["question"], k=k)

        # 'absent' is inverted: correct behaviour is refusing to answer.
        # Judged on RAW cosine similarity, not the fused score -- the fused
        # score is min-max normalized and is always 1.0 for the top hit.
        if qtype == "absent":
            per_type[qtype]["n"] += 1
            best_dense = max((c.get("_dense_score", -1.0) for _s, c in results),
                             default=-1.0)
            limit = q.get("max_acceptable_score", GROUNDING_THRESHOLD)
            if best_dense < limit:
                per_type[qtype]["hits"] += 1
                per_type[qtype]["rr"] += 1.0
            else:
                failures.append({"id": q["id"], "question": q["question"],
                                 "type": qtype,
                                 "reason": f"dense sim {best_dense:.3f} >= {limit:.3f}"})
            continue

        rank = None
        for i, (_score, chunk) in enumerate(results, start=1):
            if chunk_matches(chunk, q["source_doc"], q.get("expected_pages", [])):
                rank = i
                break

        per_type[qtype]["n"] += 1
        if rank:
            per_type[qtype]["hits"] += 1
            per_type[qtype]["rr"] += 1.0 / rank
        else:
            failures.append({
                "id": q["id"],
                "question": q["question"],
                "type": qtype,
                "expected": f"{q['source_doc']} p{q.get('expected_pages')}",
                "got": [f"{Path(str(c.get('source',''))).name} p{c.get('start_page')}"
                        for _s, c in results[:3]],
            })

    n = sum(v["n"] for v in per_type.values())
    hits = sum(v["hits"] for v in per_type.values())
    rr = sum(v["rr"] for v in per_type.values())

    return {
        "k": k, "n": n,
        "hit_rate": hits / n if n else 0.0,
        "mrr": rr / n if n else 0.0,
        "by_type": {t: {"n": v["n"],
                        "hit_rate": v["hits"] / v["n"] if v["n"] else 0.0,
                        "mrr": v["rr"] / v["n"] if v["n"] else 0.0}
                    for t, v in sorted(per_type.items())},
        "failures": failures,
    }


# ------------------------------------------------------------ reporting

def print_report(label, m):
    print(f"\n{label}")
    print(f"  hit@{m['k']}: {m['hit_rate']:.1%}   MRR: {m['mrr']:.3f}   (n={m['n']})")
    for t, v in m["by_type"].items():
        print(f"    {t:<16}{v['hit_rate']:>7.1%}  mrr {v['mrr']:.3f}  (n={v['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    if not EVAL_FILE.exists():
        sys.exit(f"No eval set at {EVAL_FILE}")

    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    live = [q for q in questions if "FILL IN" not in q["question"]]
    print(f"{len(live)} usable questions ({len(questions) - len(live)} placeholders skipped).")

    corpus = Corpus()
    output = {}

    if args.sweep:
        # Baselines first. Without these, a hybrid number means nothing --
        # you cannot claim hybrid helped if you never measured the alternatives.
        print_report("DENSE ONLY (FAISS)",
                     evaluate(corpus.retriever("dense"), questions, args.k))
        print_report("SPARSE ONLY (BM25)",
                     evaluate(corpus.retriever("sparse"), questions, args.k))

        print("\nALPHA SWEEP  (alpha = weight on dense; 0.0 = pure BM25)")
        print(f"  {'alpha':>6} {'hit@'+str(args.k):>8} {'MRR':>8}")
        sweep = []
        for a in [round(x * 0.1, 1) for x in range(11)]:
            m = evaluate(corpus.retriever("hybrid", alpha=a), questions, args.k)
            sweep.append({"alpha": a, "hit_rate": m["hit_rate"], "mrr": m["mrr"]})
            print(f"  {a:>6.1f} {m['hit_rate']:>7.1%} {m['mrr']:>8.3f}")

        best = max(sweep, key=lambda r: (r["hit_rate"], r["mrr"]))
        print(f"\nBest alpha: {best['alpha']}  "
              f"(hit {best['hit_rate']:.1%}, mrr {best['mrr']:.3f})")
        print(f"config.yaml currently sets: {get_hybrid_alpha()}")
        output = {"sweep": sweep, "best": best}
    else:
        m = evaluate(corpus.retriever("hybrid"), questions, args.k)
        print_report(f"HYBRID (alpha={get_hybrid_alpha()})", m)
        if m["failures"]:
            print(f"\n{len(m['failures'])} failures - first 5:")
            for f in m["failures"][:5]:
                print(f"  [{f['id']}] {f['question'][:65]}")
                print(f"       expected: {f.get('expected', f.get('reason'))}")
                if "got" in f:
                    print(f"       got:      {f['got']}")
        output = {"run": m}

    RESULTS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
