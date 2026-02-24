"""
FastAPI Web Application for RAG Second Brain
Provides REST API for document upload, management, and chat interface.
"""
import logging
import sys
import os
import threading
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import (
    get_project_root,
    get_data_dir,
    get_vectorstore_dir,
    get_chunk_size,
    get_chunk_overlap,
    get_default_k,
)
from ingestion.ingest import (
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_markdown,
)
from chunking.chunker import chunk_documents
from embeddings.embeddings import Embedder
from vectorstore.index import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retrieval import BM25Retriever
from llm.client import OpenRouterClient
from llm.answer_engine import AnswerEngine

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RAG_WEB_APP")

# Global state
app = FastAPI(title="RAG Second Brain API")
index_rebuild_status = {"rebuilding": False, "message": ""}
rag_components = {"retriever": None, "engine": None}
components_lock = threading.Lock()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Paths ---
DATA_DIR = get_project_root() / "data" / "raw_docs"
VECTORSTORE_DIR = get_vectorstore_dir()
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
META_PATH = VECTORSTORE_DIR / "metadata.json"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


# --- Pydantic Models ---
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


# --- Helper Functions ---
def format_citation(chunk: dict) -> str:
    """Format source citation from chunk record."""
    source = Path(chunk.get("source", "Unknown")).name
    start = chunk.get("start_page")
    end = chunk.get("end_page")

    if start is None:
        return f"{source}"
    if end is None or start == end:
        return f"{source} (Page {start})"
    return f"{source} (Pages {start}-{end})"


def rebuild_index_background():
    """Rebuild the vector store index in a background thread."""
    global index_rebuild_status, rag_components

    logger.info("Starting index rebuild in background...")

    try:
        # Check for API key
        if not os.getenv("OPENROUTER_API_KEY"):
            index_rebuild_status = {"rebuilding": False, "message": "Error: OPENROUTER_API_KEY not set"}
            return

        # Import and run the indexing logic
        from ingestion.ingest import load_documents, PageRecord

        # Load documents
        files = load_documents(DATA_DIR)
        if not files:
            index_rebuild_status = {"rebuilding": False, "message": "No documents found"}
            return

        all_records: List[PageRecord] = []
        for file_path in files:
            suffix = file_path.suffix.lower()
            records = []

            try:
                if suffix == ".pdf":
                    records = extract_text_from_pdf(file_path)
                elif suffix == ".txt":
                    records = extract_text_from_txt(file_path)
                elif suffix == ".md":
                    records = extract_text_from_markdown(file_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                continue

            if records:
                all_records.extend(records)
                logger.info(f"Processed {file_path.name}: {len(records)} records")

        if not all_records:
            index_rebuild_status = {"rebuilding": False, "message": "No text extracted"}
            return

        # Chunk documents
        chunk_size = get_chunk_size()
        chunk_overlap = get_chunk_overlap()
        chunks = chunk_documents(all_records, chunk_size=chunk_size, overlap=chunk_overlap)
        logger.info(f"Generated {len(chunks)} chunks")

        # Generate embeddings
        embedder = Embedder()
        chunk_texts = [c["text"] for c in chunks]
        vectors = embedder.embed_texts(chunk_texts)
        logger.info(f"Created vectors with shape: {vectors.shape}")

        # Create and save vector store
        store = VectorStore(dim=vectors.shape[1])
        store.add(chunks, vectors)
        store.save(str(INDEX_PATH), str(META_PATH))
        logger.info(f"Index saved to {VECTORSTORE_DIR}")

        # Reinitialize RAG components
        chunks_for_bm25 = list(store.metadata.values())
        bm25_retriever = BM25Retriever(chunks_for_bm25) if chunks_for_bm25 else None
        retriever = Retriever(embedder, store, sparse_retriever=bm25_retriever)
        client = OpenRouterClient()
        engine = AnswerEngine(retriever, client)

        with components_lock:
            rag_components["retriever"] = retriever
            rag_components["engine"] = engine

        index_rebuild_status = {"rebuilding": False, "message": "Index rebuilt successfully"}
        logger.info("Index rebuild complete!")

    except Exception as e:
        logger.exception(f"Index rebuild failed: {e}")
        index_rebuild_status = {"rebuilding": False, "message": f"Error: {str(e)}"}


def init_rag_components():
    """Initialize RAG components if index exists."""
    global rag_components

    if not INDEX_PATH.exists() or not META_PATH.exists():
        logger.warning("Index not found. Please upload documents first.")
        return False

    try:
        store = VectorStore.load(str(INDEX_PATH), str(META_PATH))
        chunks = list(store.metadata.values())

        embedder = Embedder()
        bm25_retriever = BM25Retriever(chunks) if chunks else None
        retriever = Retriever(embedder, store, sparse_retriever=bm25_retriever)
        client = OpenRouterClient()
        engine = AnswerEngine(retriever, client)

        with components_lock:
            rag_components["retriever"] = retriever
            rag_components["engine"] = engine

        logger.info("RAG components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        return False


def trigger_index_rebuild():
    """Trigger index rebuild in background thread."""
    global index_rebuild_status

    if index_rebuild_status["rebuilding"]:
        return False

    index_rebuild_status = {"rebuilding": True, "message": "Rebuilding index..."}
    rebuild_thread = threading.Thread(target=rebuild_index_background, daemon=True)
    rebuild_thread.start()
    return True


# --- API Routes ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML file."""
    static_dir = project_root / "static"
    index_file = static_dir / "index.html"

    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    else:
        return HTMLResponse(
            content="<html><body><h1>Frontend not found</h1><p>Please create static/index.html</p></body></html>",
            status_code=404
        )


@app.get("/api/rebuild-status")
async def get_rebuild_status():
    """Get the current status of index rebuild."""
    return index_rebuild_status


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, TXT, or MD file and trigger index rebuild."""
    # Validate file extension
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )

    # Save file
    file_path = DATA_DIR / file.filename

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved file: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Trigger index rebuild in background
    trigger_index_rebuild()

    return {
        "message": f"File '{file.filename}' uploaded successfully",
        "rebuilding": True
    }


@app.get("/api/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents."""
    documents = []

    if not DATA_DIR.exists():
        return documents

    for file_path in DATA_DIR.iterdir():
        if file_path.is_file():
            stat = file_path.stat()
            documents.append(DocumentInfo(
                filename=file_path.name,
                size=stat.st_size,
                date=datetime.fromtimestamp(stat.st_mtime).isoformat()
            ))

    # Sort by date, newest first
    documents.sort(key=lambda x: x.date, reverse=True)
    return documents


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document and trigger index rebuild."""
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")

    try:
        file_path.unlink()
        logger.info(f"Deleted file: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

    # Trigger index rebuild in background
    trigger_index_rebuild()

    return {
        "message": f"Document '{filename}' deleted successfully",
        "rebuilding": True
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question and get an answer with sources."""
    # Check if components are initialized
    with components_lock:
        engine = rag_components.get("engine")

    if engine is None:
        # Try to initialize components
        if not init_rag_components():
            raise HTTPException(
                status_code=400,
                detail="RAG system not initialized. Please upload documents first."
            )
        with components_lock:
            engine = rag_components.get("engine")

    if engine is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize RAG engine"
        )

    try:
        result = engine.generate_answer(request.question)

        # Format sources
        sources = []
        if result.get("grounded") and result.get("citations"):
            for score, chunk in result["citations"]:
                sources.append({
                    "score": float(score),
                    "source": format_citation(chunk),
                    "text": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", "")
                })

        return ChatResponse(
            answer=result.get("answer", "No answer generated"),
            sources=sources,
            grounded=result.get("grounded", False)
        )

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup."""
    logger.info("Initializing RAG Second Brain API...")
    init_rag_components()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
