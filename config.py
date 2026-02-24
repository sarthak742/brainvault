"""Configuration module for loading and accessing config.yaml settings."""
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Try to load .env from project root
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded environment variables from {_env_path}")
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Global config cache
_config: Optional[Dict[str, Any]] = None


def get_project_root() -> Path:
    """Get the project root directory (where this file is located)."""
    # This file IS in the project root (rag_second_brain/)
    return Path(__file__).parent


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config.yaml in project root.

    Returns:
        Configuration dictionary.
    """
    global _config

    if _config is not None:
        return _config

    if config_path is None:
        config_path = get_project_root() / "config.yaml"

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            _config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        _config = {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file: {e}")
        _config = {}

    return _config


def get_config() -> Dict[str, Any]:
    """Get the current configuration, loading if necessary."""
    if _config is None:
        load_config()
    return _config or {}


def get(key: str, default: Any = None) -> Any:
    """
    Get a configuration value by key.

    Args:
        key: Configuration key (supports dot notation like 'data_dir')
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    config = get_config()

    # Support dot notation for nested keys
    if '.' in key:
        parts = key.split('.')
        value = config
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return default
            if value is None:
                return default
        return value

    return config.get(key, default)


def resolve_path(path: str) -> Path:
    """
    Resolve a relative path from the project root.

    Args:
        path: Relative path string

    Returns:
        Absolute Path object
    """
    return get_project_root() / path


# Convenience functions for common paths
def get_data_dir() -> Path:
    """Get the data directory path."""
    return resolve_path(get('data_dir', 'data'))


def get_vectorstore_dir() -> Path:
    """Get the vectorstore directory path."""
    return resolve_path(get('vectorstore_dir', 'vectorstore_data'))


def get_chunk_size() -> int:
    """Get the chunk size setting."""
    return get('chunk_size', 1000)


def get_chunk_overlap() -> int:
    """Get the chunk overlap setting."""
    return get('chunk_overlap', 200)


def get_default_k() -> int:
    """Get the default number of chunks to retrieve."""
    return get('default_k', 5)


def get_score_threshold() -> float:
    """Get the retrieval score threshold."""
    return get('score_threshold', 0.25)


def get_hybrid_alpha() -> float:
    """Get the hybrid retrieval alpha (dense weight)."""
    return get('hybrid_alpha', 0.5)


def get_openrouter_api_key() -> str:
    """Get the OpenRouter API key from environment."""
    return os.getenv('OPENROUTER_API_KEY', '')


def get_sarvam_api_key() -> str:
    """Get the Sarvam API key from environment."""
    return os.getenv('SARVAM_API_KEY', '')


def is_sarvam_ocr_enabled() -> bool:
    """Check if Sarvam OCR is enabled in config and API key is available."""
    return get('sarvam_ocr_enabled', False) and bool(get_sarvam_api_key())


def get_sarvam_api_key() -> str:
    """Get the Sarvam API key from environment."""
    return os.getenv('SARVAM_API_KEY', '')


def is_sarvam_enabled() -> bool:
    """Check if Sarvam OCR is enabled in config and API key is available."""
    return get('sarvam_ocr_enabled', False) and bool(get_sarvam_api_key())


def get_sarvam_language() -> str:
    """Get the Sarvam OCR language setting."""
    return get('sarvam_language', 'en-IN')
