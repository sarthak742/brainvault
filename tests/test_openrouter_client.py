import pytest
import os
from unittest.mock import Mock, patch
import requests
from llm.client import OpenRouterClient

# --- Fixtures ---

@pytest.fixture
def client():
    """Returns a client instance with a dummy API key."""
    return OpenRouterClient(api_key="dummy_key")

@pytest.fixture
def mock_response():
    """Returns a Mock object simulating a requests.Response."""
    mock = Mock(spec=requests.Response)
    # Default behavior: Success
    mock.status_code = 200
    mock.raise_for_status.return_value = None
    return mock

# --- Tests ---

def test_generate_success(client, mock_response):
    """Test 1: Valid response returns content string."""
    # Setup successful JSON payload
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "Hello, world!"}}
        ]
    }
    
    # FIX 1: Patch where it is USED (llm.client.requests), not where defined
    with patch("llm.client.requests.post", return_value=mock_response) as mock_post:
        result = client.generate("Say hello")
        
        # Assertions
        assert result == "Hello, world!"
        mock_post.assert_called_once()
        
        # FIX 3: Verify Headers & Payload
        args, kwargs = mock_post.call_args
        
        # Check Headers
        assert "headers" in kwargs
        assert "Authorization" in kwargs["headers"]
        assert kwargs["headers"]["Authorization"] == "Bearer dummy_key"
        
        # Check Payload correctness
        assert kwargs["json"]["temperature"] == 0.0
        assert kwargs["json"]["messages"][0]["content"] == "Say hello"

def test_generate_empty_content_raises_error(client, mock_response):
    """Test 2: API returns success but content is None/Empty."""
    # Case A: None content
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": None}}
        ]
    }
    
    with patch("llm.client.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="empty or None content"):
            client.generate("Test prompt")
            
    # Case B: Empty string content
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": ""}}
        ]
    }
    
    with patch("llm.client.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="empty or None content"):
            client.generate("Test prompt")

def test_generate_missing_choices_raises_error(client, mock_response):
    """Test 3: API returns malformed JSON (missing choices)."""
    mock_response.json.return_value = {}  # Empty dict
    
    with patch("llm.client.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="API returned no choices"):
            client.generate("Test prompt")

def test_generate_http_error_raises_error(client, mock_response):
    """Test 4: API endpoint returns 4xx/5xx error."""
    # Simulate HTTP 401 Unauthorized
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Client Error")
    
    with patch("llm.client.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="LLM communication failed"):
            client.generate("Test prompt")

def test_generate_api_key_validation():
    """Test 5: Client requires API Key."""
    # FIX 2: Correct usage of patch.dict on os.environ
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="API Key is required"):
            OpenRouterClient(api_key=None)