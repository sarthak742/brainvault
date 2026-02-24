import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        """
        Loads embedding model into memory.
        """
        try:
            # We do NOT log "Loading..." here to keep library output clean.
            self.model = SentenceTransformer(model_name, device=device)
            self.batch_size = batch_size
        except Exception:
            # We log the exception for debugging, then re-raise cleanly
            logger.exception(f"Failed to load model {model_name}")
            raise

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds multiple texts using internal batching.
        Returns: np.ndarray of shape (N, D), dtype float32, normalized.
        """
        # 1. Handle Empty Input (Preserve (N, D) shape contract)
        if not texts:
            # Get dimension from model config or safe default (384 for MiniLM)
            dim = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)

        # 2. Embed
        # normalize_embeddings=True ensures cosine similarity works via dot product
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 3. Enforce Float32 (FAISS requirement)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string.
        Returns: np.ndarray of shape (D,), dtype float32, normalized.
        """
        # Reuse batch logic to ensure consistent normalization
        return self.embed_texts([query])[0]