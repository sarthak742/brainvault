"""
FastAPI Web Application for RAG Second Brain — full upgrade.

Layered on top of Phase 1 (incremental indexing), this adds:
  - Phase 2  ASYNC QUEUE     : uploads are indexed by a separate Celery+Redis
                               worker, not a thread in this process.
  - OBSERVABILITY            : every query is timed and logged to logs/metrics.jsonl;
                               /api/metrics aggregates it (grounded/refusal/cache/latency).
  - SEMANTIC CACHE           : repeat/similar questions skip the LLM entirely.

Cross-process rule that shows up throughout: the worker and the web server are
different processes, so they share state through the filesystem (the index) and
Redis (job status), never through Python memory.
"""
import logging
import sys
import threading
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from celery.result import AsyncResult

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from indexing import DATA_DIR, INDEX_PATH, META_PATH, build_components
from tasks import celery_app, index_document_task, full_rebuild_task
import observability
from cache import SemanticCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RAG_WEB_APP")

app = FastAPI(title="RAG Second Brain API")

rag_components = {"retriever": None, "engine": None}
_last_index_mtime = None
components_lock = threading.Lock()
_cache: Optional[SemanticCache] = None   # semantic cache, created on startup

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    grounded: bool

class DocumentInfo(BaseModel):
    filename: str
    size: int
    date: str

# --- Helpers ---
def format_citation(chunk: dict) -> str:
    source = Path(chunk.get("source", "Unknown")).name
    start = chunk.get("start_page")
    end = chunk.get("end_page")
    if start is None:
        return f"{source}"
    if end is None or start == end:
        return f"{source} (Page {start})"
    return f"{source} (Pages {start}-{end})"


def get_engine():
    """Return the engine, reloading from disk when the worker changed the index.
    When the index changes, the semantic cache is cleared so we never serve an
    answer from before a newly-added document."""
    global _last_index_mtime
    if not (INDEX_PATH.exists() and META_PATH.exists()):
        return None
    mtime = INDEX_PATH.stat().st_mtime
    with components_lock:
        if rag_components["engine"] is None or mtime != _last_index_mtime:
            logger.info("Index changed on disk — reloading retriever/engine.")
            retriever, engine = build_components()
            rag_components["retriever"] = retriever
            rag_components["engine"] = engine
            _last_index_mtime = mtime
            if _cache is not None:
                _cache.clear()   # stale-answer guard
        return rag_components["engine"]

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_file = project_root / "static" / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return HTMLResponse(content="<html><body><h1>Frontend not found</h1></body></html>", status_code=404)

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Save the file and QUEUE indexing. Returns immediately with a task_id."""
    allowed = {".pdf", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")
    file_path = DATA_DIR / file.filename
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved file: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    task = index_document_task.delay(file.filename)
    return {"message": f"File '{file.filename}' uploaded — indexing queued", "task_id": task.id}

@app.get("/api/index-status/{task_id}")
async def index_status(task_id: str):
    """Poll a queued indexing job (drives the progress bar)."""
    res = AsyncResult(task_id, app=celery_app)
    info = res.info if isinstance(res.info, dict) else {}
    done = res.state in ("SUCCESS", "FAILURE")
    payload = {
        "task_id": task_id,
        "state": res.state,
        "done": done,
        "message": info.get("message", ""),
        "progress": {"done": info.get("done", 0), "total": info.get("total", 0)},
    }
    if res.state == "FAILURE":
        payload["message"] = f"Indexing failed: {res.info}"
    return payload

@app.get("/api/metrics")
async def metrics():
    """Aggregated observability numbers — the dashboard endpoint."""
    return observability.summary()

@app.get("/api/documents", response_model=List[DocumentInfo])
async def list_documents():
    documents = []
    if not DATA_DIR.exists():
        return documents
    for file_path in DATA_DIR.iterdir():
        if file_path.is_file():
            stat = file_path.stat()
            documents.append(DocumentInfo(
                filename=file_path.name, size=stat.st_size,
                date=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            ))
    documents.sort(key=lambda x: x.date, reverse=True)
    return documents

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    file_path = DATA_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
    try:
        file_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
    task = full_rebuild_task.delay()
    return {"message": f"Document '{filename}' deleted — rebuild queued", "task_id": task.id}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=400, detail="RAG system not initialized. Please upload documents first.")

    # 1. Semantic cache: a similar question already answered? Skip the LLM.
    if _cache is not None:
        cached = _cache.get(request.question)
        if cached is not None:
            observability.record({
                "event": "query", "question": request.question,
                "grounded": cached.grounded, "num_sources": len(cached.sources),
                "cache_hit": True, "latency_ms": 0.0,
            })
            return cached

    # 2. Cache miss: run the pipeline, timing it (this is the "trace").
    try:
        with observability.Timer() as t:
            result = engine.generate_answer(request.question)

        sources = []
        if result.get("grounded") and result.get("citations"):
            for score, chunk in result["citations"]:
                text = chunk.get("text", "")
                sources.append({
                    "score": float(score),
                    "source": format_citation(chunk),
                    "text": text[:200] + "..." if len(text) > 200 else text,
                })
        response = ChatResponse(
            answer=result.get("answer", "No answer generated"),
            sources=sources,
            grounded=result.get("grounded", False),
        )

        # 3. Observability: one metrics line per query.
        observability.record({
            "event": "query", "question": request.question,
            "grounded": response.grounded, "num_sources": len(sources),
            "cache_hit": False, "latency_ms": round(t.ms, 1),
        })

        # 4. Cache the fresh answer for next time.
        if _cache is not None:
            _cache.put(request.question, response)

        return response
    except Exception as e:
        logger.exception(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.on_event("startup")
async def startup_event():
    global _cache
    logger.info("Initializing RAG Second Brain API...")
    _cache = SemanticCache(threshold=0.95, max_size=256)   # loads the embedder once
    get_engine()  # warm the engine if an index already exists

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
