import json
import time
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Set

# Standard imports assuming python -m evaluation.model_runner
from embeddings.embeddings import Embedder
from vectorstore.index import VectorStore
from retrieval.retriever import Retriever
from llm.client import OpenRouterClient
from llm.answer_engine import AnswerEngine

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("EVAL_RUNNER")

TARGET_MODELS = [
    "deepseek/deepseek-r1",
    "openai/gpt-4o-mini",
    "z-ai/glm-4.6" 
]

def load_questions(path: Path) -> List[Dict[str, Any]]:
    """
    Load evaluation dataset.
    Schema: [{
        "id": str, 
        "question": str, 
        "expect_grounded": bool,
        "relevant_doc_ids": List[str] (filenames),
        "expected_keywords": List[str]
    }]
    """
    if not path.exists():
        logger.warning(f"⚠️ {path} not found. Creating sample dataset with extended schema.")
        sample_data = [
            {
                "id": "q1", 
                "question": "What is the main topic of the documents?", 
                "expect_grounded": True,
                "relevant_doc_ids": ["doc1.txt"],
                "expected_keywords": ["topic", "summary"]
            },
            {
                "id": "q2", 
                "question": "Who is the King of Mars?", 
                "expect_grounded": False,
                "relevant_doc_ids": [],
                "expected_keywords": []
            }
        ]
        with open(path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        return sample_data

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list) or not all("question" in item for item in data):
        raise ValueError("Invalid format: eval_questions.json must be a list of objects with 'question' key.")
    
    return data

def run_benchmark():
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "vectorstore_data"
    QUESTIONS_PATH = ROOT_DIR / "evaluation" / "eval_questions.json"
    RESULTS_PATH = ROOT_DIR / "evaluation" / "results.json"

    # 1. Pre-flight Checks
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("❌ OPENROUTER_API_KEY missing.")
        return

    if not (DATA_DIR / "index.faiss").exists():
        logger.error("❌ Index not found. Run build_index.py first.")
        return

    questions = load_questions(QUESTIONS_PATH)
    logger.info(f"🧪 Loaded {len(questions)} test cases.")

    # 2. Load Shared Components
    logger.info("🧠 Loading VectorStore...")
    try:
        store = VectorStore.load(str(DATA_DIR / "index.faiss"), str(DATA_DIR / "metadata.json"))
        embedder = Embedder()
        retriever = Retriever(embedder, store)
    except Exception as e:
        logger.critical(f"❌ Setup failed: {e}")
        return

    final_results = {}

    # 3. Model Loop
    for model_name in TARGET_MODELS:
        logger.info(f"\n🚀 Benchmarking: {model_name}")
        
        # Determine Rate Limit Strategy
        sleep_duration = 1.2 if "glm-4.6" in model_name else 0.5
        
        try:
            client = OpenRouterClient(model=model_name)
            engine = AnswerEngine(retriever, client)
            
            # Metrics Containers
            details = []
            stats = {
                "TP_strong": 0, # Grounded + Keywords Matched
                "TP_weak": 0,   # Grounded + Keywords Failed
                "TN": 0,        # Correctly Refused
                "FP": 0,        # Cited when shouldn't (Hallucination)
                "FN": 0,        # Failed to cite when expected
                "retrieval_hits": 0,
                "retrieval_misses": 0,
                "retrieval_expected": 0,
                "citation_format_errors": 0, # New Counter
                "errors": 0
            }
            latencies = []

            print(f"   Progress: ", end="", flush=True)

            for item in questions:
                q_text = item["question"]
                q_id = item.get("id", "unknown")
                expect_grounded = item.get("expect_grounded", True)
                relevant_docs = set(item.get("relevant_doc_ids", []))
                expected_keywords = item.get("expected_keywords", [])
                
                start_time = time.perf_counter()
                
                try:
                    result = engine.generate_answer(q_text)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    latencies.append(elapsed_ms)
                    
                    # --- Analysis ---
                    is_grounded = result["grounded"]
                    answer_text = result["answer"]
                    retrieved_citations = result.get("citations", [])
                    
                    # New: Check for formatting failure
                    format_failed = answer_text.startswith("⚠️")
                    if format_failed:
                        stats["citation_format_errors"] += 1

                    # 1. Retrieval Quality
                    retrieved_filenames = set()
                    for _, chunk in retrieved_citations:
                        src = chunk.get("source", "")
                        if src:
                            retrieved_filenames.add(Path(src).name)
                    
                    # Hit if ANY relevant doc was found (Intersection)
                    if relevant_docs:
                        stats["retrieval_expected"] += 1
                        if not retrieved_filenames.isdisjoint(relevant_docs):
                            stats["retrieval_hits"] += 1
                            is_retrieval_hit = True
                        else:
                            stats["retrieval_misses"] += 1
                            is_retrieval_hit = False
                    else:
                        is_retrieval_hit = None # N/A

                    # 2. Answer Correctness (Keyword Match)
                    if expected_keywords:
                        keywords_met = all(kw.lower() in answer_text.lower() for kw in expected_keywords)
                    else:
                        keywords_met = None # N/A

                    # 3. Compliance & Grounding Classification
                    status = "ERR"
                    if expect_grounded:
                        if is_grounded:
                            # True Positive (Compliance)
                            if keywords_met is False: # Explicitly failed keywords
                                status = "TP_weak"
                                stats["TP_weak"] += 1
                            else:
                                # Matched keywords OR no keywords defined -> Strong
                                status = "TP_strong"
                                stats["TP_strong"] += 1
                        else:
                            # False Negative (Refusal OR Format Error)
                            status = "FN"
                            stats["FN"] += 1
                    else:
                        if not is_grounded:
                            # True Negative (Refusal)
                            status = "TN"
                            stats["TN"] += 1
                        else:
                            # False Positive (Hallucination)
                            status = "FP"
                            stats["FP"] += 1

                    # Log Detail
                    top_score = retrieved_citations[0][0] if retrieved_citations else 0.0
                    
                    details.append({
                        "id": q_id,
                        "question": q_text,
                        "latency_ms": round(elapsed_ms, 2),
                        "status": status,
                        "format_failed": format_failed, # Specific flag
                        "metrics": {
                            "grounded_actual": is_grounded,
                            "retrieval_hit": is_retrieval_hit,
                            "keyword_match": keywords_met
                        },
                        "retrieval_debug": {
                            "expected_docs": list(relevant_docs),
                            "retrieved_docs": list(retrieved_filenames),
                            "top_score": round(top_score, 4),
                            "num_chunks": len(retrieved_citations)
                        },
                        "answer_snippet": answer_text[:200]
                    })

                    # Visual Feedback
                    symbol_map = {
                        "TP_strong": "✅", "TP_weak": "⚠️", 
                        "TN": "🛡️", "FP": "🚨", "FN": "❌"
                    }
                    if format_failed:
                        print("📝", end="", flush=True) # Special icon for format fail
                    else:
                        print(symbol_map.get(status, "?"), end="", flush=True)
                    
                    # Rate Limiting
                    time.sleep(sleep_duration)

                except Exception as e:
                    print("💥", end="", flush=True)
                    stats["errors"] += 1
                    details.append({"id": q_id, "error": str(e)})

            print(" Done.")

            # 4. Aggregation
            total_qs = len(questions)
            total_valid = len(latencies)
            
            # Helper sums
            tp_total = stats["TP_strong"] + stats["TP_weak"]
            retrieval_denominator = stats["retrieval_expected"]
            
            summary = {
                "total_questions": total_qs,
                "config": {
                    "rate_limit_sleep_s": sleep_duration
                },
                "avg_latency_ms": round(sum(latencies) / total_valid, 2) if total_valid else 0,
                "metrics": {
                    # Retrieval Quality
                    "retrieval_recall": round(stats["retrieval_hits"] / retrieval_denominator, 2) if retrieval_denominator > 0 else 0,
                    
                    # Answer Quality
                    "keyword_accuracy": round(stats["TP_strong"] / tp_total, 2) if tp_total > 0 else 0,
                    
                    # Safety / Compliance
                    "citation_compliance_recall": round(tp_total / (tp_total + stats["FN"]), 2) if (tp_total + stats["FN"]) > 0 else 0,
                    "citation_format_error_rate": round(stats["citation_format_errors"] / total_valid, 2) if total_valid else 0, # New Metric
                    "citation_false_positive_rate": round(stats["FP"] / total_valid, 2) if total_valid else 0,
                    "refusal_accuracy": round(stats["TN"] / (stats["TN"] + stats["FP"]), 2) if (stats["TN"] + stats["FP"]) > 0 else 0
                },
                "counts": stats
            }
            
            final_results[model_name] = {
                "summary": summary,
                "details": details
            }

            logger.info(f"   -> Recall: {summary['metrics']['retrieval_recall']} | Format Errs: {stats['citation_format_errors']}")

        except Exception as e:
            logger.error(f"   ❌ Failed model {model_name}: {e}")
            final_results[model_name] = {"error": str(e)}

    # 5. Save Report
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"\n📊 Detailed Report saved to: {RESULTS_PATH}")

if __name__ == "__main__":
    run_benchmark()