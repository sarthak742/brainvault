"""
Semantic cache for BrainVault.

A normal cache keys on the exact string: "what is photosynthesis" and
"explain photosynthesis" would be two different keys and two LLM calls. A
SEMANTIC cache keys on MEANING: it embeds the question and, if a past question
is close enough in vector space, returns that stored answer — zero LLM call,
instant, free. (This is the GPTCache idea you came up with on your own.)

Notes / honest limits:
- It lives in the web server's memory, so it's per-process and cleared on restart.
  For sharing across processes or surviving restarts, move it to Redis later.
- It must be CLEARED when the document set changes, or it could serve an answer
  from before a new document was added. app.py clears it when the index reloads.
"""
import logging
import threading
from typing import Any, Optional, Tuple, List

import numpy as np

from embeddings.embeddings import Embedder

logger = logging.getLogger("brainvault.cache")


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


class SemanticCache:
    def __init__(self, threshold: float = 0.95, max_size: int = 256):
        # threshold: cosine similarity required to count as "the same question".
        # 0.95 is deliberately strict — a wrong cache hit serves a wrong answer.
        self.threshold = threshold
        self.max_size = max_size
        self._embedder = Embedder()
        self._vectors: List[np.ndarray] = []
        self._entries: List[Tuple[str, Any]] = []   # (question, response)
        self._lock = threading.Lock()

    def get(self, question: str) -> Optional[Any]:
        """Return a cached response for a semantically-equivalent question, or None."""
        q = _normalize(self._embedder.embed_texts([question])[0])
        with self._lock:
            if not self._vectors:
                return None
            sims = np.array([float(q @ v) for v in self._vectors])
            best = int(sims.argmax())
            if sims[best] >= self.threshold:
                logger.info("Semantic cache HIT (sim=%.3f): %r", sims[best], question)
                return self._entries[best][1]
        return None

    def put(self, question: str, response: Any) -> None:
        q = _normalize(self._embedder.embed_texts([question])[0])
        with self._lock:
            self._vectors.append(q)
            self._entries.append((question, response))
            if len(self._entries) > self.max_size:      # simple FIFO eviction
                self._vectors.pop(0)
                self._entries.pop(0)

    def clear(self) -> None:
        with self._lock:
            self._vectors.clear()
            self._entries.clear()
            logger.info("Semantic cache cleared (document set changed).")
