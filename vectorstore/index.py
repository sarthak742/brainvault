import logging
import json
import os
from typing import List, Dict, Tuple, cast
import numpy as np
import faiss

# Correct Import Path based on your project structure
from chunking.chunker import ChunkRecord

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Local vector store using FAISS IndexIDMap + Explicit Metadata.
    
    Guarantees:
    - IDs are explicitly managed (no implicit sequential drift).
    - Metadata is always synced with Vector IDs.
    - JSON serialization is robust.
    """

    def __init__(self, dim: int = 384) -> None:
        """
        Initialize vector store with explicit ID mapping.
        """
        self.dim = dim
        
        # We wrap IndexFlatIP in IndexIDMap to enable add_with_ids()
        # This allows us to control the ID mapping strictly.
        base_index = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap(base_index)
        
        self.metadata: Dict[int, ChunkRecord] = {}
        self.next_id = 0

    def add(self, chunks: List[ChunkRecord], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the store with explicit IDs.
        """
        # 1. Validation
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} vectors")
        
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dim {embeddings.shape[1]} != index dim {self.dim}")

        count = len(chunks)
        if count == 0:
            return

        # 2. Generate Explicit IDs
        # We generate a range of IDs starting from our current counter
        start_id = self.next_id
        ids = np.arange(start_id, start_id + count).astype(np.int64)
        
        # 3. Add to FAISS (The Math)
        # add_with_ids requires numpy int64 array for IDs
        self.index.add_with_ids(embeddings, ids)
        
        # 4. Add to Metadata (The Text)
        for i, chunk in enumerate(chunks):
            # Explicitly cast to dict to ensure serialization safety later
            self.metadata[int(ids[i])] = cast(ChunkRecord, dict(chunk))
            
        # 5. Update State
        self.next_id += count
        logger.info(f"Added {count} vectors. Next ID: {self.next_id}")

    def search(self, query_vec: np.ndarray, k: int = 5, user_id: str = None) -> List[Tuple[float, ChunkRecord]]:
        """
        Search for most similar chunks.

        If user_id is given, results are restricted to that user's chunks
        (multi-tenant isolation). Because this is a flat, exact index, we search
        the whole index and keep only the user's chunks before taking the top k
        -- correct no matter how different users' vectors are interleaved.
        Chunks with no user_id are treated as the 'default' tenant, so indexes
        built before multi-tenancy keep working.

        Returns: List of (score, ChunkRecord)
        """
        # 1. Guard Clauses
        if self.index.ntotal == 0:
            return []
            
        if query_vec.ndim != 1 or query_vec.shape[0] != self.dim:
            raise ValueError(f"Query shape mismatch. Expected ({self.dim},), got {query_vec.shape}")

        # 2. Reshape for FAISS (needs 1, dim)
        query_vec = query_vec.reshape(1, -1).astype(np.float32)

        # 3. Search. When filtering by tenant we must over-fetch, because the
        # global top-k could be entirely other users' chunks. On a flat index,
        # searching all vectors is the same O(n) cost as any search.
        search_k = self.index.ntotal if user_id is not None else k
        D, I = self.index.search(query_vec, search_k)

        # 4. Map back to Chunks (filtering by tenant if requested)
        results = []
        for j, idx in enumerate(I[0]):
            if idx == -1:
                continue
            if idx not in self.metadata:
                logger.error(f"CRITICAL: Index {idx} found in FAISS but missing in Metadata!")
                continue
            record = self.metadata[idx]
            if user_id is not None and record.get("user_id", "default") != user_id:
                continue
            results.append((float(D[0][j]), record))
            if user_id is not None and len(results) >= k:
                break

        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        """Persist vector index and metadata mapping to disk."""
        try:
            # 1. Save FAISS Index
            faiss.write_index(self.index, index_path)
            
            # 2. Save Metadata
            # Explicitly ensure we are dumping serializable dicts
            # IDs must be strings in JSON key position
            serializable_meta = {str(k): dict(v) for k, v in self.metadata.items()}
            
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(serializable_meta, f, indent=2)
                
            logger.info(f"Saved index ({self.index.ntotal} vectors) to disk.")
        except Exception:
            logger.exception("Failed to save VectorStore")
            raise

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "VectorStore":
        """Load vector store from disk and restore ID counter."""
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Index or Metadata file not found.")
            
        try:
            # 1. Load FAISS
            index = faiss.read_index(index_path)
            
            # 2. Load Metadata
            with open(metadata_path, "r", encoding="utf-8") as f:
                raw_metadata = json.load(f)
            
            # 3. Fix JSON Keys & Restore Types
            metadata = {}
            max_id = -1
            
            for k, v in raw_metadata.items():
                int_id = int(k)
                metadata[int_id] = cast(ChunkRecord, v)
                if int_id > max_id:
                    max_id = int_id
            
            # 4. Reconstruct Instance
            # Check dimension from the loaded index
            store = cls(dim=index.d)
            store.index = index
            store.metadata = metadata
            
            # Restore the counter to be 1 greater than the highest existing ID
            store.next_id = max_id + 1 if max_id >= 0 else 0
            
            logger.info(f"Loaded VectorStore. Vectors: {index.ntotal}, Next ID: {store.next_id}")
            return store
            
        except Exception:
            logger.exception("Failed to load VectorStore")
            raise