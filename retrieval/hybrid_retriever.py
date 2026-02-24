import logging
import hashlib
from typing import List, Tuple, Dict, Any

# Type alias matching your existing system
ChunkRecord = Dict[str, Any]

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Combines Dense (Semantic) and Sparse (Keyword) retrieval results.
    Uses score normalization and weighted linear fusion.
    """

    def __init__(self, dense_retriever, sparse_retriever, alpha: float = 0.5):
        """
        Args:
            dense_retriever: Instance of your FAISS retriever.
            sparse_retriever: Instance of BM25Retriever.
            alpha: Weight for dense scores (0.0 to 1.0). 
                   0.5 = Equal weight.
                   0.7 = Bias towards semantic search.
        """
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha

    def _generate_fallback_id(self, chunk: ChunkRecord) -> str:
        """
        Generate consistent hash for chunks missing 'chunk_id'.
        MUST match BM25Retriever's logic (Source + Text).
        """
        raw = f"{chunk.get('source', '')}|{chunk.get('text', '')}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _normalize_scores(self, results: List[Tuple[float, ChunkRecord]]) -> List[Tuple[float, ChunkRecord]]:
        """
        Min-Max normalization to scale scores to [0, 1].
        """
        if not results:
            return []

        scores = [r[0] for r in results]
        min_s = min(scores)
        max_s = max(scores)
        
        # Edge case: All scores identical or only 1 result
        if max_s == min_s:
            # Treat as max relevance (1.0) if non-zero
            return [(1.0, r[1]) for r in results] if max_s > 0 else [(0.0, r[1]) for r in results]

        normalized = []
        for score, chunk in results:
            # Min-Max Formula
            norm_score = (score - min_s) / (max_s - min_s)
            normalized.append((norm_score, chunk))
            
        return normalized

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[float, ChunkRecord]]:
        """
        Execute Hybrid Retrieval:
        1. Fetch dense & sparse results (oversampling k*2).
        2. Normalize scores.
        3. Merge and deduplicate using chunk_id (with fallback).
        4. Sort and return top k.
        """
        # 1. Fetch Candidates (Oversample to allow effective merging)
        dense_res = self.dense.retrieve(query, k=k*2)
        sparse_res = self.sparse.retrieve(query, k=k*2)

        if not dense_res and not sparse_res:
            logger.warning("HybridRetriever: Both retrievers returned empty results.")
            return []

        # 2. Normalize Scores
        dense_norm = self._normalize_scores(dense_res)
        sparse_norm = self._normalize_scores(sparse_res)

        # 3. Merge Strategy (Weighted Sum)
        combined_scores: Dict[str, float] = {}
        chunk_map: Dict[str, ChunkRecord] = {}

        # Process Dense Results
        for score, chunk in dense_norm:
            # CRITICAL FIX: Fallback ID if missing
            cid = chunk.get("chunk_id") or self._generate_fallback_id(chunk)
            
            chunk_map[cid] = chunk
            # Score = NormalizedDense * Alpha
            combined_scores[cid] = combined_scores.get(cid, 0.0) + (score * self.alpha)

        # Process Sparse Results
        for score, chunk in sparse_norm:
            cid = chunk.get("chunk_id") or self._generate_fallback_id(chunk)
            
            chunk_map[cid] = chunk
            # Additive boosting: If chunk exists, this adds to it. If new, it creates entry.
            # Score += NormalizedSparse * (1 - Alpha)
            combined_scores[cid] = combined_scores.get(cid, 0.0) + (score * (1 - self.alpha))

        # 4. Final Sort & Truncate
        final_results = []
        for cid, score in combined_scores.items():
            final_results.append((score, chunk_map[cid]))
        
        # Sort descending by fused score
        final_results.sort(key=lambda x: x[0], reverse=True)
        
        return final_results[:k]