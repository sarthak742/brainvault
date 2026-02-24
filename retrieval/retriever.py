import logging
from typing import List, Tuple, Dict, Any

# We use "Any" for strict dependency injection to avoid circular imports
from chunking.chunker import ChunkRecord
from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

class Retriever:
    """
    Bridge between the Embedding Model and the Vector Store.
    Now acts as a Router:
    - If sparse_retriever is present -> routes via HybridRetriever.
    - If NOT -> routes via internal Dense logic (legacy behavior).
    """

    def __init__(self, embedder: Any, store: Any, sparse_retriever: Any = None):
        """
        Args:
            embedder: Instance with .embed_query(text) -> np.array
            store: Instance with .search(vector, k) -> List[Tuple[float, ChunkRecord]]
            sparse_retriever: Optional instance of BM25Retriever.
        """
        self.embedder = embedder
        self.store = store
        self.sparse_retriever = sparse_retriever
        
        # Initialize Hybrid Logic if components exist
        self.hybrid_runner = None
        if self.sparse_retriever:
            # We create a delegate to allow HybridRetriever to call dense logic
            # without triggering recursion in self.retrieve()
            dense_delegate = _DenseDelegate(self)
            self.hybrid_runner = HybridRetriever(
                dense_retriever=dense_delegate, 
                sparse_retriever=self.sparse_retriever
            )

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[float, ChunkRecord]]:
        """
        Find most relevant chunks for a query.
        Routes to Hybrid if enabled, otherwise Standard Dense.
        """
        if self.hybrid_runner:
            # Route through Hybrid Fusion
            return self.hybrid_runner.retrieve(query, k=k)
        else:
            # Fallback to pure Dense (Legacy Path)
            return self._retrieve_dense(query, k=k)

    def _retrieve_dense(self, query: str, k: int = 5) -> List[Tuple[float, ChunkRecord]]:
        """
        Internal implementation of dense vector search.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to retriever.")
            return []

        # 1. Embed the query
        query_vector = self.embedder.embed_query(query)

        # 2. Search the index
        results = self.store.search(query_vector, k=k)

        # 3. Sort (Safety measure)
        results.sort(key=lambda x: x[0], reverse=True)

        return results

class _DenseDelegate:
    """
    Helper to expose dense retrieval logic to HybridRetriever
    without causing infinite recursion in Retriever.retrieve().
    """
    def __init__(self, parent: Retriever):
        self.parent = parent

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[float, ChunkRecord]]:
        return self.parent._retrieve_dense(query, k=k)