🧠 BrainVault
A powerful RAG (Retrieval-Augmented Generation) system that lets you upload documents and chat with them using AI. Supports both digital PDFs and handwritten/scanned documents.
✨ Features

📄 Document Upload — Upload PDF, TXT, and Markdown files via drag-and-drop
🔍 Hybrid Retrieval — Combines dense (FAISS) and sparse (BM25) search for accurate results
🤖 AI-Powered Answers — Uses DeepSeek R1 via OpenRouter for intelligent responses
📸 OCR Support — Sarvam AI Vision reads handwritten and scanned PDFs
💬 Web Chat Interface — Clean, modern UI with source citations
⚡ Background Indexing — Index rebuilds in the background without freezing the UI
🗑️ Document Management — View and delete uploaded documents
 Getting Started
Prerequisites

Python 3.10+
Tesseract OCR (optional, for basic OCR)
Poppler (for PDF to image conversion)

*Installation*

Clone the repository

git clone https://github.com/sarthak742/brainvault.git
cd brainvault

Create a virtual environment

python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

Install dependencies

pip install -r requirements.txt

Set up environment variables

Create a .env file in the project root:
envOPENROUTER_API_KEY=your_openrouter_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here

Get OpenRouter API key at: https://openrouter.ai
Get Sarvam API key at: https://dashboard.sarvam.ai


Build the initial index (optional, if you have existing docs)

python build_index.py

Run the web app

python app.py
Open your browser at http://localhost:8000



Usage
Web Interface

Open http://localhost:8000
Upload documents using the sidebar (drag-and-drop or click)
Wait for the index to rebuild (status indicator in bottom left)
Ask questions in the chat interface
Get answers with source citations

## 🪞 Self-Reflection Layer (CRAG + Answer Critique)

BrainVault does not blindly trust its first retrieval or its first answer. A
self-reflective loop wraps the base `retrieve → generate → validate` pipeline
with three cooperating LLM critics. All of it is config-gated and
backward-compatible: turn it off and the pipeline behaves exactly as before.

**1. Retrieval grading (Corrective RAG).** Before spending a generation call,
a `RetrievalGrader` asks the LLM which retrieved chunks are actually relevant
to the question. If none are, the system does not push irrelevant context into
the model (the classic source of confident-but-wrong RAG answers).

**2. Query rewrite + retry.** When grading comes back *insufficient*, a
`QueryRewriter` rewrites the question into a stronger search query and retrieval
runs a second time. This recovers vague or vocabulary-mismatched questions.

**3. Answer critique (faithfulness).** The base citation check only proves a
`[1]` marker is *present and in range* — it cannot catch an answer that cites
`[2]` for a fact chunk 2 never stated. An `AnswerCritic` runs a second pass that
reads the context and the answer together and judges whether every claim is
supported. If it judges the answer *unsupported*, the engine regenerates once
under a stricter instruction, then re-checks.

Every critic **fails open**: if a reflection LLM call errors or returns
unparseable output, the system degrades to the base behaviour rather than
breaking the user's request. Each answer carries a `reflection` audit trail
(grading verdicts, any rewritten query, the critique result) for transparency.

### Reflection config (`config.yaml`)

| Key | Default | Effect |
|-----|---------|--------|
| `reflection_enabled` | `true` | Master switch. Off = original pipeline. |
| `reflection_grade_retrieval` | `true` | Grade chunk relevance before generating. |
| `reflection_rewrite_query` | `true` | Rewrite + retry when grading is insufficient. |
| `reflection_critique_answer` | `true` | Faithfulness critique after generating. |
| `reflection_max_retrieval_attempts` | `2` | Total retrieval tries (incl. the first). |
| `reflection_allow_regeneration` | `true` | Regenerate once on an unsupported verdict. |

**Cost/latency tradeoff:** with everything on, a hard question can cost up to
~6 LLM calls (grade → rewrite → grade → generate → critique → regenerate).
Each stage is individually toggleable so you can dial reliability against cost.
