import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """
    Client for communicating with LLMs via the OpenRouter API.
    Handles authentication, request validation, and response parsing.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize the LLM client.

        Provider-agnostic: works with any OpenAI-compatible chat-completions
        endpoint (OpenRouter, NVIDIA NIM, Groq, local vLLM). Defaults come from
        config.yaml so switching providers means editing config, not code.

        Args:
            api_key: API key. If None, reads the env var named by
                     `llm_api_key_env` in config.yaml.
            model: Model identifier. Defaults to config `llm_model`.
            base_url: Full endpoint URL. Defaults to config `llm_base_url`.
            timeout: Request timeout in seconds.

        Raises:
            ValueError: If API key is missing.
        """
        from config import get_llm_base_url, get_llm_model, get_llm_api_key, get_llm_key_env

        self.api_key = api_key or get_llm_api_key()

        if not self.api_key:
            raise ValueError(
                f"API Key is required. Set {get_llm_key_env()} in your .env "
                f"or pass it explicitly."
            )

        self.model = model or get_llm_model()
        self.base_url = base_url or get_llm_base_url()
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and retrieve the text response.

        Args:
            prompt: The user prompt string.

        Returns:
            str: The content of the assistant's response.

        Raises:
            ValueError: If prompt is empty.
            RuntimeError: If API fails, response is malformed, or content is empty.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Local RAG System"
        }

        # Temperature 0.0 is critical for RAG to reduce hallucination
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 1000
        }

        try:
            logger.debug(f"Sending request to {self.model}...")
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            # Check for HTTP errors (4xx, 5xx)
            response.raise_for_status()
            
            # Parse Response
            data = response.json()
            
            # Extract content strictly
            if "choices" not in data or not data["choices"]:
                 raise RuntimeError("API returned no choices.")
                 
            content = data["choices"][0]["message"].get("content")
            
            if not content:
                raise RuntimeError("LLM returned empty or None content.")
                
            return content

        except requests.exceptions.RequestException as e:
            # We catch this first to provide a clear network/status error
            logger.error(f"OpenRouter API Network Error: {e}")
            raise RuntimeError(f"LLM communication failed: {e}") from e
            
        except (KeyError, IndexError, TypeError) as e:
            # We catch parsing errors second
            logger.error(f"Malformed response from OpenRouter: {e}")
            raise RuntimeError(f"Malformed LLM response structure: {e}") from e