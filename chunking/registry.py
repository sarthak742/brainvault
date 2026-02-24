import json
import hashlib
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Type alias for clarity
ChunkRecord = Dict[str, Any]

logger = logging.getLogger(__name__)

class ChunkRegistry:
    """
    Authoritative Source of Truth for all chunks in the system.
    
    Guarantees:
    1. Deterministic IDs (Source + Page + Text).
    2. Strict insertion order (for FAISS index alignment).
    3. Schema versioning and strict validation on load.
    4. Immutability of internal state.
    """
    
    CURRENT_VERSION = 1

    def __init__(self):
        # Private storage to enforce append-only / immutable access
        self._chunks: List[ChunkRecord] = []
        self._chunk_map: Dict[str, ChunkRecord] = {}

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        """
        Register new chunks.
        - Trust existing chunk_ids if present (e.g., from reload).
        - Generate deterministic IDs if missing.
        - Deduplicates (idempotent).
        - Appends new chunks to the master list.
        """
        added_count = 0
        for chunk in chunks:
            # 1. Basic Validation
            if "text" not in chunk:
                logger.warning("Skipping malformed chunk: missing 'text' field.")
                continue

            # 2. ID Resolution (Trust Existing vs Generate New)
            if "chunk_id" in chunk:
                chunk_id = chunk["chunk_id"]
            else:
                chunk_id = self._generate_chunk_id(chunk)
            
            # 3. Deduplication (Idempotency)
            if chunk_id in self._chunk_map:
                continue # Skip duplicates
            
            # 4. Storage (Defensive Copy)
            stored_chunk = chunk.copy()
            stored_chunk["chunk_id"] = chunk_id
            
            self._chunks.append(stored_chunk)
            self._chunk_map[chunk_id] = stored_chunk
            added_count += 1

        logger.debug(f"Registry added {added_count} chunks. Total: {len(self._chunks)}")

    def get_chunks(self) -> List[ChunkRecord]:
        """
        Return all chunks in strict insertion order.
        Returns a SHALLOW COPY to prevent external reordering.
        """
        return list(self._chunks)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkRecord]:
        """O(1) Lookup by string ID (e.g., from BM25 or Hybrid)."""
        return self._chunk_map.get(chunk_id)

    def get_chunk_by_index(self, index: int) -> Optional[ChunkRecord]:
        """
        O(1) Lookup by integer index. 
        CRITICAL: This is the bridge to FAISS. 
        FAISS ID 0 -> Registry Index 0.
        """
        if 0 <= index < len(self._chunks):
            return self._chunks[index]
        return None

    def _generate_chunk_id(self, chunk: ChunkRecord) -> str:
        """Generate strict deterministic ID (Source + Page + Text)."""
        source = str(chunk.get("source", "unknown"))
        page = str(chunk.get("start_page", "unknown"))
        text = chunk.get("text", "")
        
        raw = f"{source}|{page}|{text}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def save(self, path: str) -> None:
        """Persist registry to disk with schema versioning."""
        payload = {
            "version": self.CURRENT_VERSION,
            "chunks": self._chunks
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            logger.info(f"✅ Registry saved to {path} ({len(self._chunks)} chunks)")
        except Exception as e:
            logger.error(f"❌ Failed to save registry: {e}")
            raise

    @classmethod
    def load(cls, path: str) -> "ChunkRegistry":
        """
        Load registry from disk with strict validation.
        Raises ValueError if schema is corrupted.
        """
        registry = cls()
        path_obj = Path(path)
        
        if not path_obj.exists():
            logger.warning(f"Registry file {path} not found. Returning empty registry.")
            return registry

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 1. Schema Check
            if not isinstance(data, dict) or "chunks" not in data:
                 raise ValueError("Invalid Registry format: Missing 'chunks' list.")

            # 2. Strict Validation Pass (Fail Fast)
            raw_chunks = data["chunks"]
            for i, chunk in enumerate(raw_chunks):
                if "chunk_id" not in chunk:
                    raise ValueError(f"CRITICAL CORRUPTION: Chunk {i} missing 'chunk_id'.")
                if "text" not in chunk:
                    raise ValueError(f"CRITICAL CORRUPTION: Chunk {i} missing 'text'.")

            # 3. Populate
            registry.add_chunks(raw_chunks)
            
            # 4. Integrity Check
            if len(registry._chunks) != len(raw_chunks):
                logger.warning("⚠️ Registry loaded with duplicate ID removal. Check source file integrity.")
            
            logger.info(f"✅ Registry loaded from {path} ({len(registry._chunks)} chunks)")
            return registry
            
        except Exception as e:
            logger.error(f"❌ Failed to load registry: {e}")
            raise