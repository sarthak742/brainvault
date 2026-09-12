# 🧠 BrainVault

A local, citation-grounded **RAG (Retrieval-Augmented Generation)** system: upload your documents, ask questions, and get answers drawn **only** from your files — with source-and-page citations, hybrid search, a self-correcting reflection loop, and a built-in evaluation harness.

Unlike pasting text into a chatbot, BrainVault searches across a whole library of documents, cites exactly where each answer came from, and **refuses to answer** when it can't find support — so it doesn't hallucinate.

---

## ✨ Features

- **Hybrid retrieval** — combines dense semantic search (FAISS + MiniLM embeddings) with sparse keyword search (BM25), so it catches both *meaning* ("young dog" → "puppy") and *exact terms* (error codes, names, IDs).
- **Self-reflective loop** — grades retrieved chunks before answering (rewriting the query if they're weak) and critiques the answer's faithfulness afterward (regenerating if it's unsupported). Every critic **fails open** so it can never break a request.
- **Evaluation harness** — measures retrieval quality (recall@k, hit@k, MRR) and answer quality (grounding, refusal, hallucination), and A/B-tests baseline vs. reflection.
- **Grounded answers with citations** — every answer cites `[1] [2]` mapping to a real source file and page range.
- **OCR fallback** — fully-scanned PDFs are sent to Sarvam's document-intelligence API when normal text extraction finds nothing.
- **Background indexing** — uploads reindex on a background thread so the UI never freezes.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph BUILD["Build time (on upload)"]
        A[Files: PDF / TXT / MD] --> B[Ingest to source, page, text]
        B --> C[Chunk ~1000 chars, 200 overlap]
        C --> D[Embed - MiniLM, 384-dim]
        D --> E[(FAISS index + metadata.json)]
    end

    subgraph QUERY["Query time (every question)"]
        Q[User question] --> R[Embed query - same model]
        R --> S[Hybrid retrieve: dense + BM25, normalize, fuse]
        S --> G{Grade chunks relevant?}
        G -- no --> RW[Rewrite query + retry] --> S
        G -- yes --> T{Pass 0.25 threshold?}
        T -- no --> X[Refuse: no LLM call]
        T -- yes --> CTX[Build numbered context] --> LLM[LLM - temp 0]
        LLM --> V[Validate citations]
        V --> CR{Answer faithful?}
        CR -- no --> RG[Regenerate once] --> V
        CR -- yes --> OUT[Answer + sources]
    end

    E -.-> S
```

**Two models, never confused:** the *embedding model* (MiniLM) makes vectors; the *LLM* writes answers.

---

## 🔁 How a query flows (in plain terms)

1. **Ask** — the question hits the engine.
2. **Find** — hybrid search retrieves candidate chunks (semantic + keyword, fused).
3. **Check the chunks** *(before the LLM)* — a cheap score threshold refuses junk; the reflection grader rewrites-and-retries if chunks are weak.
4. **Answer** — good chunks + a "use only these" prompt go to the LLM.
5. **Check the answer** *(after the LLM)* — citations are validated and the critic verifies faithfulness (regenerating once if needed).
6. **Show** — the answer renders with its cited sources.

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

- **Retrieval:** FAISS (`IndexFlatIP`), `sentence-transformers` (`all-MiniLM-L6-v2`), `rank-bm25`
- **LLM:** OpenAI-compatible chat API (OpenRouter / NVIDIA NIM / etc.), model set by config
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
python build_index.py     # index documents placed under data/raw_docs
python app.py             # open http://localhost:8000
```

---

## 🧭 Known limitations & roadmap

- **Reflection needs a stronger judge model** than the generator (see Evaluation) — the current self-critique shares the generator's blind spots.
- **Full rebuild on every upload** — should be *incremental* (only embed new chunks).
- **Brute-force search** (`IndexFlatIP`) is exact but O(n); at millions of chunks, move to an approximate index (IVF/HNSW).
- **Single-process, in-memory** — for real concurrency, externalize the index to a vector DB.
- **OCR triggers per-document, not per-page** — a mostly-text PDF with one scanned page won't OCR that page.
- **No reranker yet** — a cross-encoder over the top-k would sharpen results.

---

## 📁 Project structure

```
ingestion/    read files + OCR fallback to PageRecords
chunking/     paragraph-aware chunker
embeddings/   MiniLM embedder
vectorstore/  FAISS index + metadata (save/load)
retrieval/    dense, BM25, hybrid fusion
llm/          answer engine + OpenAI-compatible client
reflection/   retrieval grader, query rewriter, answer critic
evaluation/   metrics + A/B harness
app.py        FastAPI web app   .   main.py  CLI
```
