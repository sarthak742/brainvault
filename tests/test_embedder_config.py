"""The Embedder must read batch_size/model from config (single source of truth),
fixing the old 32-vs-64 mismatch.

The Embedder pulls in sentence-transformers at import time, which isn't always
installed in lightweight CI. The config-getter test runs anywhere; the two
Embedder tests skip gracefully when sentence-transformers is absent, and run
for real (with the model load mocked) where it is installed.
"""
import pytest
import config


def test_config_getters_match_yaml():
    assert config.get_embedding_batch_size() == 32
    assert config.get_embedding_model() == "all-MiniLM-L6-v2"


def test_embedder_defaults_to_config_batch_size():
    pytest.importorskip("sentence_transformers")
    from unittest.mock import patch
    with patch("embeddings.embeddings.SentenceTransformer"):
        from embeddings.embeddings import Embedder
        e = Embedder()  # no args -> must be 32 (config), NOT the old hardcoded 64
        assert e.batch_size == 32


def test_explicit_arg_still_overrides_config():
    pytest.importorskip("sentence_transformers")
    from unittest.mock import patch
    with patch("embeddings.embeddings.SentenceTransformer"):
        from embeddings.embeddings import Embedder
        e = Embedder(batch_size=16)
        assert e.batch_size == 16
