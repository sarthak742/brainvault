"""
Core indexing logic for BrainVault — shared by the web app AND the Celery worker.

There is NO FastAPI and NO Celery in this file on purpose. Both `app.py` (the web
server) and `tasks.py` (the worker) import from here, so the actual work lives in
one place and neither imports the other (that would be a circular import).
"""
import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from config import (
    get_project_root,
    get_vectorstore_dir,
    get_chunk_size,
    get_chunk_overlap,
)
from ingestion.ingest import (
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_markdown,
    load_documents,
)
from chunking.chunker import chunk_documents
from embeddings.embeddings import Embedder
from vectorstore.index import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retrieval import BM25Retriever
from llm.client import OpenRouterClient
from reflection.builder import build_answer_engine

logger = logging.getLogger("brainvault.indexing")

# --- Paths (same as before) ---
DATA_DIR = get_project_root() / "data" / "raw_docs"
VECTORSTORE_DIR = get_vectorstore_dir()
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
META_PATH = VECTORSTORE_DIR / "metadata.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# A progress callback: report(step_done, step_total, message)
ProgressCb = Optional[Callable[[int, int, str], None]]


def _extract_records(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix == ".txt":
        return extract_text_from_txt(file_path)
    if suffix == ".md":
        return extract_text_from_markdown(file_path)
    return []


def _load_or_new_store(dim: int) -> VectorStore:
    if INDEX_PATH.exists() and META_PATH.exists():
        return VectorStore.load(str(INDEX_PATH), str(META_PATH))
    return VectorStore(dim=dim)


def _remove_existing_chunks(store: VectorStore, filename: str, user_id: str = "default") -> None:
    """Idempotency guard, scoped to a single tenant.

    A queued job can run more than once (a retry after a crash, or the user
    re-uploading the same filename). Because our indexing APPENDS, running it
    twice would add the document's chunks twice. So before adding, we drop any
    chunks already stored for THIS user's copy of this filename. Matching on
    user_id too means user A re-uploading "notes.pdf" never touches user B's.
    """
    ids = [
        i for i, chunk in store.metadata.items()
        if Path(chunk.get("source", "")).name == filename
        and chunk.get("user_id", "default") == user_id
    ]
    if ids:
        store.index.remove_ids(np.array(ids, dtype=np.int64))
        for i in ids:
            del store.metadata[i]
        logger.info("Removed %d existing chunks for '%s' (re-index).", len(ids), filename)


def index_single_document(filename: str, user_id: str = "default", progress_cb: ProgressCb = None) -> dict:
    """Incrementally index ONE document for ONE user. Idempotent per (user, file)."""

    def report(done: int, total: int, msg: str) -> None:
        logger.info("[index %s/%s] %s (%d/%d)", user_id, filename, msg, done, total)
        if progress_cb:
            progress_cb(done, total, msg)

    report(0, 4, f"Reading {filename}")
    file_path = DATA_DIR / user_id / filename
    records = _extract_records(file_path)
    if not records:
        report(4, 4, f"No text extracted from {filename}")
        return {"filename": filename, "chunks": 0, "message": "No text extracted"}

    report(1, 4, "Chunking")
    chunks = chunk_documents(
        records, chunk_size=get_chunk_size(), overlap=get_chunk_overlap()
    )
    # Tag every chunk with its owner so retrieval can filter by tenant.
    for c in chunks:
        c["user_id"] = user_id

    report(2, 4, f"Embedding {len(chunks)} chunks")
    embedder = Embedder()
    vectors = embedder.embed_texts([c["text"] for c in chunks])

    report(3, 4, "Updating index")
    store = _load_or_new_store(dim=vectors.shape[1])
    _remove_existing_chunks(store, filename, user_id)   # idempotent re-index
    store.add(chunks, vectors)                          # append only the new chunks
    store.save(str(INDEX_PATH), str(META_PATH))

    report(4, 4, f"Indexed {filename}")
    return {"filename": filename, "chunks": len(chunks), "message": f"Indexed '{filename}'"}


def full_rebuild_index(progress_cb: ProgressCb = None) -> dict:
    """Rebuild the whole multi-tenant index from every user's documents on disk.

    Documents live under DATA_DIR/<user_id>/<file>, so each chunk is tagged with
    the user_id taken from its subdirectory — isolation survives a full rebuild.
    Any files sitting directly in DATA_DIR (pre-multi-tenant) are treated as the
    'default' tenant.
    """

    def report(done: int, total: int, msg: str) -> None:
        logger.info("[rebuild] %s (%d/%d)", msg, done, total)
        if progress_cb:
            progress_cb(done, total, msg)

    # Collect (user_id, file_path) pairs across all tenants.
    jobs = []
    if DATA_DIR.exists():
        for entry in DATA_DIR.iterdir():
            if entry.is_dir():
                for f in entry.iterdir():
                    if f.is_file():
                        jobs.append((entry.name, f))
            elif entry.is_file():
                jobs.append(("default", entry))

    if not jobs:
        # Nothing left (e.g. the last doc was deleted) — clear the index.
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        if META_PATH.exists():
            META_PATH.unlink()
        report(1, 1, "No documents — index cleared")
        return {"message": "No documents — index cleared"}

    total = len(jobs) + 1
    all_chunks: List = []
    for idx, (uid, file_path) in enumerate(jobs, start=1):
        report(idx, total, f"Reading {file_path.name}")
        try:
            records = _extract_records(file_path)
        except Exception as e:
            logger.error("Failed to process %s: %s", file_path.name, e)
            continue
        if not records:
            continue
        chunks = chunk_documents(
            records, chunk_size=get_chunk_size(), overlap=get_chunk_overlap()
        )
        for c in chunks:
            c["user_id"] = uid          # tag by the tenant that owns the folder
        all_chunks.extend(chunks)

    if not all_chunks:
        report(total, total, "No text extracted")
        return {"message": "No text extracted"}

    embedder = Embedder()
    vectors = embedder.embed_texts([c["text"] for c in all_chunks])

    store = VectorStore(dim=vectors.shape[1])   # fresh store for a full rebuild
    store.add(all_chunks, vectors)
    store.save(str(INDEX_PATH), str(META_PATH))

    report(total, total, "Index rebuilt")
    return {"message": "Index rebuilt", "chunks": len(all_chunks)}


def build_components():
    """Load a (retriever, engine) pair from the CURRENT on-disk index.

    The web server calls this to refresh what it serves after the worker has
    updated the index on disk. Returns (None, None) if there is no index yet.
    """
    if not (INDEX_PATH.exists() and META_PATH.exists()):
        return None, None
    store = VectorStore.load(str(INDEX_PATH), str(META_PATH))
    chunks = list(store.metadata.values())
    embedder = Embedder()
    bm25 = BM25Retriever(chunks) if chunks else None
    retriever = Retriever(embedder, store, sparse_retriever=bm25)
    engine = build_answer_engine(retriever, OpenRouterClient())
    return retriever, engine
