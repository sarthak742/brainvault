import logging
import re
import hashlib
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi

# Type alias matching your existing system
ChunkRecord = Dict[str, Any]

logger = logging.getLogger(__name__)

class BM25Retriever:
    """
    Sparse retriever based on BM25 algorithm for keyword matching.
    Includes robust tokenization, metadata indexing, and stable chunk IDs.
    
    WARNING: Scores returned are raw BM25 scores (unbounded). 
    Normalization is required before merging with dense scores.
    """

    # Matches alphanumeric strings, including single characters (e.g., "C", "R", "5")
    TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")

    def __init__(self, chunks: List[ChunkRecord]):
        """
        Initialize and build the BM25 index.
        
        Args:
            chunks: List of chunk records. 
                    If 'chunk_id' is missing, a hash will be generated.
        """
        self.chunks = chunks
        self._corpus_size = len(chunks)
        
        # 1. Enrich & Tokenize Corpus
        tokenized_corpus = []
        for i, chunk in enumerate(self.chunks):
            # Ensure stable ID for Hybrid merging later
            if "chunk_id" not in chunk:
                chunk["chunk_id"] = self._generate_chunk_id(chunk)
            
            # Index Text + Metadata (so "source: doc1" works)
            searchable_text = self._build_searchable_text(chunk)
            tokens = self._tokenize(searchable_text)
            
            if not tokens:
                logger.debug(f"⚠️ Empty BM25 tokens for chunk {chunk['chunk_id']}")
                
            tokenized_corpus.append(tokens)

        # 2. Build Index
        if not tokenized_corpus:
            logger.warning("⚠️ BM25 initialized with empty corpus.")
            self.bm25 = None
        else:
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"✅ BM25 Index built for {self._corpus_size} chunks.")

    def _tokenize(self, text: str) -> List[str]:
        """Robust regex-based tokenizer."""
        return self.TOKEN_PATTERN.findall(text.lower())

    def _build_searchable_text(self, chunk: ChunkRecord) -> str:
        """Combine text and key metadata for broader recall."""
        parts = [chunk.get("text", "")]
        # Add filename/source if present to allow source-specific queries
        if "source" in chunk:
            parts.append(str(chunk["source"]))
        return " ".join(parts)

    def _generate_chunk_id(self, chunk: ChunkRecord) -> str:
        """Create stable hash based on FULL content to avoid collisions."""
        # We use full text + source to guarantee uniqueness
        raw = f"{chunk.get('source', '')}|{chunk.get('text', '')}"
        return hashlib.md5(raw.encode()).hexdigest()

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[float, ChunkRecord]]:
        """
        Retrieve chunks based on keyword overlap.
        
        Returns:
            List of (raw_score, chunk_record), sorted by score descending.
        """
        if not self.bm25 or not query.strip():
            return []

        tokenized_query = self._tokenize(query)
        
        # Logging for debug (crucial for tuning)
        logger.debug(f"BM25 Query Tokens: {tokenized_query}")
        
        scores = self.bm25.get_scores(tokenized_query)
        
        # Defensive Check: Ensure corpus alignment
        if len(scores) != self._corpus_size:
            logger.error(f"CRITICAL: BM25 score count ({len(scores)}) != corpus size ({self._corpus_size})")
            return []

        # Filter & Sort
        scored_chunks = []
        for i, score in enumerate(scores):
            if score > 0:
                scored_chunks.append((float(score), self.chunks[i]))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        return scored_chunks[:k]