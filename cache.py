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
        self._entries: List[Tuple[str, str, Any]] = []   # (user_id, question, response)
        self._lock = threading.Lock()

    def get(self, question: str, user_id: str = "default") -> Optional[Any]:
        """Return a cached response for a semantically-equivalent question from the
        SAME user, or None.

        Scoping by user_id is a correctness requirement, not an optimization:
        without it, one user's cached answer could be served to another user
        asking a similar question -- a cross-tenant leak that bypasses the
        retrieval-level isolation entirely.
        """
        q = _normalize(self._embedder.embed_texts([question])[0])
        with self._lock:
            best_sim, best_resp = -1.0, None
            for vec, (uid, _q, resp) in zip(self._vectors, self._entries):
                if uid != user_id:
                    continue
                sim = float(q @ vec)
                if sim > best_sim:
                    best_sim, best_resp = sim, resp
            if best_resp is not None and best_sim >= self.threshold:
                logger.info("Semantic cache HIT (sim=%.3f, user=%s): %r", best_sim, user_id, question)
                return best_resp
        return None

    def put(self, question: str, response: Any, user_id: str = "default") -> None:
        q = _normalize(self._embedder.embed_texts([question])[0])
        with self._lock:
            self._vectors.append(q)
            self._entries.append((user_id, question, response))
            if len(self._entries) > self.max_size:      # simple FIFO eviction
                self._vectors.pop(0)
                self._entries.pop(0)

    def clear(self) -> None:
        with self._lock:
            self._vectors.clear()
            self._entries.clear()
            logger.info("Semantic cache cleared (document set changed).")
