import pytest
from unittest.mock import Mock, patch
from retrieval.retriever import Retriever

# --- Fixtures ---

@pytest.fixture
def mock_embedder():
    embedder = Mock()
    # Return a fixed dummy vector for consistent assertions
    embedder.embed_query.return_value = [0.1, 0.2] 
    return embedder

@pytest.fixture
def mock_store():
    store = Mock()
    # Mock behavior for dense search
    store.search.return_value = [(0.9, {"text": "dense hit", "id": "1"})]
    return store

@pytest.fixture
def mock_sparse_retriever():
    return Mock()

# --- Tests ---

def test_retriever_basic_dense_search(mock_embedder, mock_store):
    """
    Legacy Test: Verifies that standard retrieval works as expected.
    This ensures we haven't broken the existing contract.
    """
    retriever = Retriever(embedder=mock_embedder, store=mock_store)
    
    results = retriever.retrieve("hello world", k=3)
    
    # Check wiring
    mock_embedder.embed_query.assert_called_once_with("hello world")
    mock_store.search.assert_called_once_with([0.1, 0.2], k=3)
    
    # Check results
    assert len(results) == 1
    assert results[0][1]["text"] == "dense hit"

def test_retriever_routes_to_hybrid_when_configured(mock_embedder, mock_store, mock_sparse_retriever):
    """
    Integration Test: Verifies that passing a sparse_retriever triggers 
    hybrid routing instead of direct dense execution.
    """
    # Patch HybridRetriever to verify it gets initialized and called correctly
    with patch("retrieval.retriever.HybridRetriever") as MockHybridClass:
        # Setup the mock instance returned by the constructor
        mock_hybrid_instance = MockHybridClass.return_value
        mock_hybrid_instance.retrieve.return_value = [(1.0, {"text": "hybrid hit"})]
        
        # Initialize with sparse retriever
        retriever = Retriever(
            embedder=mock_embedder, 
            store=mock_store, 
            sparse_retriever=mock_sparse_retriever
        )
        
        # 1. Verify Constructor Wiring
        MockHybridClass.assert_called_once()
        _, kwargs = MockHybridClass.call_args
        assert kwargs["sparse_retriever"] is mock_sparse_retriever
        assert kwargs["dense_retriever"] is not None # Delegate must be created
        
        # 2. Execute Retrieve
        results = retriever.retrieve("test query", k=10)
        
        # 3. Verify Routing
        # Should call hybrid
        mock_hybrid_instance.retrieve.assert_called_once_with("test query", k=10)
        
        # Should NOT call dense components directly (Hybrid manages that)
        mock_embedder.embed_query.assert_not_called()
        mock_store.search.assert_not_called()
        
        # 4. Verify Result Propagation
        assert len(results) == 1
        assert results[0][1]["text"] == "hybrid hit"

def test_retriever_handles_empty_query(mock_embedder, mock_store):
    """
    Edge Case: Empty queries should return empty lists immediately
    without hitting the backend (saving API calls).
    """
    retriever = Retriever(embedder=mock_embedder, store=mock_store)
    
    # Empty string
    assert retriever.retrieve("") == []
    # Whitespace only
    assert retriever.retrieve("   ") == []
    
    # Ensure no calls made
    mock_embedder.embed_query.assert_not_called()
    mock_store.search.assert_not_called()