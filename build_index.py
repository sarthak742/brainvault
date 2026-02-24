import logging
import sys
from pathlib import Path
from typing import List

# --- Imports from your independent modules ---
from ingestion.ingest import (
    load_documents,
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_markdown,
    PageRecord,
)
from chunking.chunker import chunk_documents
from embeddings.embeddings import Embedder
from vectorstore.index import VectorStore
from config import get_data_dir, get_vectorstore_dir, get_chunk_size, get_chunk_overlap

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BUILD_INDEX")


def main():
    # 1. Setup Paths using config
    DATA_ROOT = get_data_dir()
    VECTORSTORE_DIR = get_vectorstore_dir()
    INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
    META_PATH = VECTORSTORE_DIR / "metadata.json"

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    # Get chunking config
    chunk_size = get_chunk_size()
    chunk_overlap = get_chunk_overlap()

    logger.info("🚀 Starting Index Build...")
    logger.info(f"   Data Source: {DATA_ROOT}")
    logger.info(f"   Output:      {VECTORSTORE_DIR}")
    logger.info(f"   Chunk Size:  {chunk_size}, Overlap: {chunk_overlap}")

    # 2. Ingestion
    files = load_documents(DATA_ROOT)
    if not files:
        logger.error("❌ No files found in data/ directory.")
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
        except Exception:
            logger.exception(f"Failed to process {file_path.name}")
            continue

        if records:
            all_records.extend(records)
            logger.info(f"   Processed {file_path.name}: {len(records)} pages")

    if not all_records:
        logger.warning("❌ No text extracted. Exiting.")
        return

    # 3. Chunking
    logger.info("✂️  Chunking Documents...")
    chunks = chunk_documents(all_records)
    logger.info(f"   Generated {len(chunks)} chunks.")

    # 4. Embedding
    logger.info("🧠 Generating Embeddings...")
    embedder = Embedder()
    chunk_texts = [c["text"] for c in chunks]
    vectors = embedder.embed_texts(chunk_texts)
    logger.info(f"   Created vectors with shape: {vectors.shape}")

    # 5. Indexing
    logger.info("💾 Saving to Vector Store...")
    store = VectorStore(dim=vectors.shape[1])
    store.add(chunks, vectors)

    # 6. Persistence
    store.save(str(INDEX_PATH), str(META_PATH))
    logger.info(f"✅ Build Complete. Index saved to {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
