import pytest
from unittest.mock import Mock
from llm.answer_engine import AnswerEngine

# --- Fixtures ---

@pytest.fixture
def mock_retriever():
    retriever = Mock()
    # Default: Return nothing unless configured
    retriever.retrieve.return_value = []
    return retriever

@pytest.fixture
def mock_client():
    client = Mock()
    # CRITICAL FIX: Default response MUST have a citation to pass strict validation
    client.generate.return_value = "Mocked Answer [1]."
    return client

@pytest.fixture
def engine(mock_retriever, mock_client):
    return AnswerEngine(retriever=mock_retriever, client=mock_client)

@pytest.fixture
def sample_chunk():
    return {"text": "This is a test chunk.", "source": "doc1", "start_page": 1}

# --- Standard Tests ---

def test_generate_answer_success(engine, mock_retriever, mock_client, sample_chunk):
    """Test standard grounded generation flow with valid citation."""
    mock_retriever.retrieve.return_value = [(0.9, sample_chunk)]
    
    # Mock return value is set in fixture as "Mocked Answer [1]."
    result = engine.generate_answer("test query")
    
    # Assertions
    assert result["grounded"] is True
    assert "Mocked Answer [1]" in result["answer"]
    assert len(result["citations"]) == 1
    
    mock_client.generate.assert_called_once()

def test_generate_answer_refusal_low_score(engine, mock_retriever, mock_client, sample_chunk):
    """Test that weak evidence triggers fail-fast (no LLM call)."""
    # Setup: Chunk score (0.1) < Default Threshold (0.25)
    mock_retriever.retrieve.return_value = [(0.1, sample_chunk)]
    
    result = engine.generate_answer("test query", score_threshold=0.25)
    
    assert result["grounded"] is False
    assert "could not find" in result["answer"].lower()
    assert result["citations"] == []
    
    # Ensure we saved money by NOT calling LLM
    mock_client.generate.assert_not_called()

def test_generate_answer_empty_retrieval(engine, mock_retriever, mock_client):
    """Test behavior when retriever finds nothing."""
    mock_retriever.retrieve.return_value = []
    
    result = engine.generate_answer("test query")
    
    assert result["grounded"] is False
    mock_client.generate.assert_not_called()

def test_context_truncation(engine, mock_retriever, mock_client):
    """Test that context respects max_context_chars and 'at least one chunk' rule."""
    chunk1 = {"text": "AAAA " * 4, "source": "d1"} 
    chunk2 = {"text": "BBBB " * 4, "source": "d2"} 
    
    mock_retriever.retrieve.return_value = [(0.9, chunk1), (0.8, chunk2)]
    
    # Set limit very low so only chunk1 fits.
    # Logic guarantees at least 1 chunk is included.
    engine.generate_answer("q", max_context_chars=10)
    
    # Check what was sent to LLM
    args, _ = mock_client.generate.call_args
    prompt = args[0]
    
    assert "AAAA" in prompt # Chunk 1 must be there (guaranteed inclusion)
    assert "BBBB" not in prompt # Chunk 2 should be dropped

def test_llm_failure_handling(engine, mock_retriever, mock_client, sample_chunk):
    """Test graceful degradation if LLM API fails."""
    mock_retriever.retrieve.return_value = [(0.9, sample_chunk)]
    
    # Simulate API crash
    mock_client.generate.side_effect = RuntimeError("API Down")
    
    result = engine.generate_answer("test query")
    
    # Assertions
    assert result["grounded"] is False
    assert "error" in result["answer"].lower()
    
    # KEY FEATURE: We still return citations so user sees what was found
    # This was the fix in the Engine logic, now tested here.
    assert len(result["citations"]) == 1
    assert result["citations"][0][1]["text"] == "This is a test chunk."

# --- Citation Validation Tests ---

def test_citation_validation_missing(engine, mock_retriever, mock_client, sample_chunk):
    """Test that answer without citation markers is flagged and hides evidence."""
    mock_retriever.retrieve.return_value = [(0.9, sample_chunk)]
    # Explicitly return answer WITHOUT citation
    mock_client.generate.return_value = "This is an answer without proof."

    result = engine.generate_answer("test query")

    # Should be False because no [1] found
    assert result["grounded"] is False
    assert "missing citations" in result["answer"]
    # Evidence Hiding: Citations should be empty on validation failure
    assert result["citations"] == []

def test_citation_validation_invalid_number(engine, mock_retriever, mock_client, sample_chunk):
    """Test that citing a non-existent chunk index is flagged."""
    mock_retriever.retrieve.return_value = [(0.9, sample_chunk)]
    # We only have 1 chunk, but LLM cites [2]
    mock_client.generate.return_value = "This is a hallucination [2]."

    result = engine.generate_answer("test query")

    assert result["grounded"] is False
    assert "invalid citations" in result["answer"]
    assert result["citations"] == []