# 🧠 BrainVault

A local, citation-grounded **RAG (Retrieval-Augmented Generation)** system: upload your documents, ask questions, and get answers drawn **only** from your files — with source-and-page citations, hybrid search, a self-correcting reflection loop, an async indexing queue, live metrics, and a built-in evaluation harness.

Unlike pasting text into a chatbot, BrainVault searches across a whole library of documents, cites exactly where each answer came from, and **refuses to answer** when it can't find support — so it doesn't hallucinate.

---

## ✨ Features

- **Hybrid retrieval** — combines dense semantic search (FAISS + MiniLM embeddings) with sparse keyword search (BM25), so it catches both *meaning* ("young dog" → "puppy") and *exact terms* (error codes, names, IDs).
- **Self-reflective loop** — grades retrieved chunks before answering (rewriting the query if they're weak) and critiques the answer's faithfulness afterward (regenerating if it's unsupported). Every critic **fails open** so it can never break a request.
- **Async indexing queue** — uploads return immediately with a `task_id`; a separate Celery + Redis worker does the embedding. The web server never blocks, and you can poll job progress from the UI.
- **Incremental indexing** — a new upload embeds and adds only *its* chunks (via explicit FAISS IDs), instead of rebuilding the whole index. Deletes remove just that document's chunks.
- **Observability** — every query is timed and logged; `/api/metrics` aggregates grounded rate, refusal rate, cache-hit rate, and latency (avg / p50 / p95).
- **Semantic cache** — repeat or near-identical questions return a stored answer without an LLM call, and the cache clears itself whenever the document set changes.
- **Multi-tenant capable** — every chunk carries a `user_id` tag and retrieval, cache, and document folders are all scoped to it, so one deployment can isolate multiple users' libraries. Runs single-user by default (`user_id` defaults to `"default"`).
- **Grounded answers with citations** — every answer cites `[1] [2]` mapping to a real source file and page range.
- **OCR fallback** — fully-scanned PDFs are sent to Sarvam's document-intelligence API when normal text extraction finds nothing.

---

## 🏗️ Architecture

Two processes share state through the filesystem (the FAISS index) and Redis (job status) — never through Python memory:

```mermaid
flowchart TD
    subgraph UPLOAD["Upload (web server)"]
        U[POST /api/upload] --> SAVE[Save file to data/raw_docs/&lt;user&gt;/]
        SAVE --> ENQ[Enqueue index job in Redis] --> TID[Return task_id]
    end

    subgraph WORKER["Worker process (Celery)"]
        JOB[Pick up job] --> ING[Ingest to source, page, text]
        ING --> CHK[Chunk ~1000 chars, 200 overlap]
        CHK --> EMB[Embed - MiniLM, 384-dim, tag user_id]
        EMB --> ADD[Add only new chunks to index] --> IDX[(FAISS index + metadata.json)]
    end

    subgraph QUERY["Query time (every question)"]
        Q[POST /api/chat] --> CACHE{Semantic cache hit?}
        CACHE -- yes --> OUT[Answer + sources]
        CACHE -- no --> R[Embed query - same model]
        R --> S[Hybrid retrieve: dense + BM25, scoped to user_id, fuse]
        S --> G{Grade chunks relevant?}
        G -- no --> RW[Rewrite query + retry] --> S
        G -- yes --> T{Pass grounding threshold?}
        T -- no --> X[Refuse: no LLM call]
        T -- yes --> CTX[Build numbered context] --> LLM[LLM - temp 0]
        LLM --> V[Validate citations]
        V --> CR{Answer faithful?}
        CR -- no --> RG[Regenerate once] --> V
        CR -- yes --> OUT
        OUT --> MET[Record metrics + cache answer]
    end

    IDX -.reloaded on change.-> S
```

**Two models, never confused:** the *embedding model* (MiniLM) makes vectors; the *LLM* writes answers. **Two processes, never confused:** the *web server* serves requests; the *worker* does the indexing.

---

## 🔁 How a query flows (in plain terms)

1. **Ask** — the question hits the engine.
2. **Cache** — if a semantically-equivalent question was already answered, return that instantly (no LLM call).
3. **Find** — hybrid search retrieves candidate chunks (semantic + keyword, fused), scoped to the asking user.
4. **Check the chunks** *(before the LLM)* — a score threshold refuses junk; the reflection grader rewrites-and-retries if chunks are weak.
5. **Answer** — good chunks + a "use only these" prompt go to the LLM.
6. **Check the answer** *(after the LLM)* — citations are validated and the critic verifies faithfulness (regenerating once if needed).
7. **Show & record** — the answer renders with its cited sources, and one metrics line is logged.

---

## 📊 Evaluation

The harness scores **retrieval** and **answers** separately on a labelled set (6 documents, 8 questions), and A/B-tests the reflection layer.

**Retrieval quality** (hybrid dense + BM25, no API key needed):

| Metric | Score |
|---|---|
| recall@5 | 1.00 |
| hit@5 | 1.00 |
| MRR | 1.00 |

Every topical question retrieved its correct document at rank 1. (Note: a small, topic-distinct set, so retrieval is an easy case — a larger, more ambiguous set would be a harder test.)

**Answer quality — baseline pipeline** on an adversarial set (3 real + 5 "trap" questions designed to tempt hallucination), Llama-3.1-8B:

| Metric | Baseline |
|---|---|
| grounding accuracy | 0.875 |
| refusal accuracy | 0.80 |
| hallucination rate | 0.125 |

The base pipeline correctly refused 4 of 5 trap questions on its own — the strict "answer only from context, else say you couldn't find it" prompt does real work.

**Reflection layer — honest note.** In A/B tests, the reflection layer did **not** beat baseline. The cause is diagnosable: the critic ran on the *same-size* model as the generator, so self-critique shared the generator's blind spots (it missed a hallucination the generator made, and over-flagged a correct answer). The correct configuration uses a **judge model stronger than the generator**; validating that at eval volume needs paid API throughput, so a rigorous reflection A/B is **future work**. The base retrieval + citation + threshold guards already deliver the strong numbers above.

**Reproduce:**
```bash
python -m pytest tests/test_metrics.py tests/test_harness.py -q   # offline, no key
python -m evaluation.compare_reflection evaluation/eval_questions.json   # needs a built index + API key
```

---

## 🛠️ Tech stack

- **Retrieval:** FAISS (`IndexFlatIP` wrapped in `IndexIDMap`), `sentence-transformers` (`all-MiniLM-L6-v2`), `rank-bm25`
- **LLM:** OpenAI-compatible chat API (OpenRouter / NVIDIA NIM / etc.), model set by config
- **Async:** Celery + Redis (job queue + task status)
- **OCR:** Sarvam AI document-intelligence (fallback for scanned PDFs)
- **API/UI:** FastAPI + a vanilla HTML/JS front end
- **Config:** single `config.yaml` (all tunable knobs)

---

## 🚀 Getting started

```bash
git clone https://github.com/sarthak742/brainvault.git
cd brainvault
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # then add your API key (OPENROUTER_API_KEY; SARVAM_API_KEY for OCR)
```

You also need a **Redis** server running (the job queue + task-status store):

```bash
docker run -d -p 6379:6379 redis      # or install redis and run: redis-server
```

Then run **three processes** — this split is the point of the async design:

```bash
# 1. Redis — see above

# 2. The worker (does the indexing, on its own)
celery -A tasks.celery_app worker --loglevel=info --concurrency=1

# 3. The web server
python app.py            # open http://localhost:8000
```

`--concurrency=1` makes the worker index one job at a time, so two jobs never write the FAISS index file at once. Drop documents through the UI (or place them under `data/raw_docs/` and let the worker index them); the CLI (`python main.py`) still works for offline use.

---

## 🧭 Known limitations & roadmap

- **Reflection needs a stronger judge model** than the generator (see Evaluation) — the current self-critique shares the generator's blind spots.
- **Brute-force search** (`IndexFlatIP`) is exact but O(n); at millions of chunks, move to an approximate index (IVF/HNSW), measured against the eval harness because it can trade off recall.
- **Semantic cache is in-process** — it lives in the web server's memory (cleared on restart). Sharing it across processes or surviving restarts means moving it to Redis.
- **OCR triggers per-document, not per-page** — a mostly-text PDF with one scanned page won't OCR that page.
- **No reranker yet** — a cross-encoder over the top-k would sharpen results.
- **Horizontal scaling** is deployment, not code — run N copies of the app behind a load balancer and more workers off the same Redis.

---

## 📁 Project structure

```
ingestion/    read files + OCR fallback to PageRecords
chunking/     paragraph-aware chunker
embeddings/   MiniLM embedder
vectorstore/  FAISS index + metadata (save/load, per-chunk user_id)
retrieval/    dense, BM25, hybrid fusion (all user-scoped)
llm/          answer engine + OpenAI-compatible client
reflection/   retrieval grader, query rewriter, answer critic
evaluation/   metrics + A/B harness
indexing.py   shared indexing core (web server + worker)
tasks.py      Celery worker (Redis-backed job queue)
observability.py  per-query metrics -> logs/metrics.jsonl, aggregated by /api/metrics
cache.py      semantic cache (skip the LLM for similar questions)
app.py        FastAPI web app   .   main.py  CLI
```
