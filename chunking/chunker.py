import logging
from typing import List, TypedDict, Optional

from config import get_chunk_size, get_chunk_overlap

# --- Types ---
# REDEFINED here to avoid Circular Import with ingest.py
class PageRecord(TypedDict):
    source: str
    page: Optional[int]
    text: str

class ChunkRecord(TypedDict):
    text: str
    source: str
    start_page: Optional[int]
    end_page: Optional[int]

logger = logging.getLogger(__name__)

# Get defaults from config
DEFAULT_CHUNK_SIZE = get_chunk_size()
DEFAULT_OVERLAP = get_chunk_overlap()


def _mechanical_split(
    text: str,
    source: str,
    start_page: Optional[int],
    end_page: Optional[int],
    chunk_size: int,
    overlap: int
) -> List[ChunkRecord]:
    """
    Splits text purely by character count.
    """
    chunks: List[ChunkRecord] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end]

        chunks.append({
            "text": chunk_text,
            "source": source,
            "start_page": start_page,
            "end_page": end_page
        })

        if end == text_len:
            break

        start = end - overlap

    return chunks


def chunk_documents(
    records: List[PageRecord],
    chunk_size: int = None,
    overlap: int = None
) -> List[ChunkRecord]:
    """
    State-machine based chunker.
    Uses config values if chunk_size and overlap not provided.
    """
    # Use config values if not explicitly provided
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE
    if overlap is None:
        overlap = DEFAULT_OVERLAP

    if not records:
        return []

    chunks: List[ChunkRecord] = []

    # --- State Variables ---
    current_text = ""
    current_source: Optional[str] = None
    current_start_page: Optional[int] = None
    current_end_page: Optional[int] = None

    last_chunk_end_page: Optional[int] = None

    for record in records:
        # 1. Validate Record
        if not record['text'] or not record['text'].strip():
            continue

        source = record['source']
        page_num = record.get('page')  # Use .get() for safety

        # 2. Source Boundary Check
        if current_source is not None and source != current_source:
            if current_text:
                chunks.append({
                    "text": current_text,
                    "source": current_source,
                    "start_page": current_start_page,
                    "end_page": current_end_page
                })
                last_chunk_end_page = current_end_page

            # Hard Reset State
            current_text = ""
            current_start_page = None
            current_end_page = None
            last_chunk_end_page = None
            current_source = None

        # 3. Initialize State
        if current_source is None:
            current_source = source
            current_start_page = page_num
            current_end_page = page_num
            last_chunk_end_page = page_num

        # 4. Split Paragraphs
        paragraphs = record['text'].split('\n\n')

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            sep_len = 2 if current_text else 0
            projected_size = len(current_text) + sep_len + len(paragraph)

            # --- CASE A: Giant Paragraph ---
            if len(paragraph) > chunk_size:
                if current_text:
                    chunks.append({
                        "text": current_text,
                        "source": current_source,
                        "start_page": current_start_page,
                        "end_page": current_end_page
                    })
                    last_chunk_end_page = current_end_page
                    current_text = ""

                sub_chunks = _mechanical_split(
                    paragraph, source, page_num, page_num, chunk_size, overlap
                )
                chunks.extend(sub_chunks)

                current_start_page = page_num
                current_end_page = page_num
                last_chunk_end_page = page_num
                continue

            # --- CASE B: Fits ---
            if projected_size <= chunk_size:
                if current_text:
                    current_text += "\n\n" + paragraph
                else:
                    current_text = paragraph
                    if current_start_page is None:
                        current_start_page = page_num

                current_end_page = page_num

            # --- CASE C: Overflow ---
            else:
                chunks.append({
                    "text": current_text,
                    "source": current_source,
                    "start_page": current_start_page,
                    "end_page": current_end_page
                })
                last_chunk_end_page = current_end_page

                overlap_text = current_text[-overlap:] if len(current_text) > overlap else current_text
                candidate_text = overlap_text + "\n\n" + paragraph

                if len(candidate_text) > chunk_size:
                    sub_chunks = _mechanical_split(
                        paragraph, source, page_num, page_num, chunk_size, overlap
                    )
                    chunks.extend(sub_chunks)
                    current_text = ""
                    current_start_page = page_num
                    current_end_page = page_num
                    last_chunk_end_page = page_num
                else:
                    current_text = candidate_text

                    if page_num is None:
                        current_start_page = None
                    elif last_chunk_end_page is not None:
                        current_start_page = last_chunk_end_page
                    else:
                        current_start_page = page_num

                    current_end_page = page_num

    # 5. Final Flush
    if current_text and current_source is not None:
        chunks.append({
            "text": current_text,
            "source": current_source,
            "start_page": current_start_page,
            "end_page": current_end_page
        })

    # --- Logging ---
    if chunks:
        sizes = [len(c['text']) for c in chunks]
        avg_size = sum(sizes) / len(sizes)
        max_size = max(sizes)
        logger.info(f"Chunking Complete. Generated {len(chunks)} chunks.")
        logger.info(f"Stats: Avg Size={avg_size:.2f}, Max Size={max_size}")
    else:
        logger.warning("Chunking produced NO chunks.")

    return chunks
