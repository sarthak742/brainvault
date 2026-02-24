import pytest
import numpy as np
import os
from chunking.chunker import ChunkRecord
from embeddings.embeddings import Embedder

from vectorstore.index import VectorStore

# --- Fixtures ---

@pytest.fixture(scope="module")
def embedder():
    """Load the heavy model only once for the whole test module."""
    return Embedder()

@pytest.fixture
def store():
    """Create a fresh VectorStore for each test."""
    # MiniLM dim is 384
    return VectorStore(dim=384)

@pytest.fixture
def sample_data():
    """Create distinct semantic chunks for testing."""
    chunks = [
        ChunkRecord(text="Apple is a tasty red fruit.", source="doc1", start_page=1, end_page=1),
        ChunkRecord(text="The iPhone is a powerful smartphone.", source="doc2", start_page=1, end_page=1),
        ChunkRecord(text="Mars is a dusty red planet.", source="doc3", start_page=1, end_page=1),
    ]
    texts = [c["text"] for c in chunks]
    return chunks, texts

# --- Tests ---

def test_add_and_search(store, embedder, sample_data):
    """Test 1: Can we add vectors and find the semantically closest one?"""
    chunks, texts = sample_data
    vectors = embedder.embed_texts(texts)
    
    # 1. Add
    store.add(chunks, vectors)
    assert store.index.ntotal == 3
    assert len(store.metadata) == 3
    
    # 2. Search
    # "fruit" should match "Apple" (ID 0)
    query = "fruit"
    q_vec = embedder.embed_query(query)
    results = store.search(q_vec, k=1)
    
    # 3. Assertions
    assert len(results) == 1
    score, record = results[0]
    
    # Check Text
    assert "Apple" in record["text"]
    # Check Score (Should be high, > 0.3 for MiniLM)
    assert score > 0.3 

def test_save_and_load_roundtrip(store, embedder, sample_data, tmp_path):
    """Test 2: Does persistence work without data loss?"""
    chunks, texts = sample_data
    vectors = embedder.embed_texts(texts)
    store.add(chunks, vectors)
    
    # Paths using pytest's temporary directory
    index_path = str(tmp_path / "test_index.faiss")
    meta_path = str(tmp_path / "test_meta.json")
    
    # 1. Save
    store.save(index_path, meta_path)
    assert os.path.exists(index_path)
    assert os.path.exists(meta_path)
    
    # 2. Load
    loaded_store = VectorStore.load(index_path, meta_path)
    
    # 3. Verify Integrity
    assert loaded_store.index.ntotal == 3
    assert loaded_store.next_id == 3
    assert len(loaded_store.metadata) == 3
    
    # 4. Verify Search on Loaded Store
    # "planet" should match "Mars" (ID 2)
    q_vec = embedder.embed_query("planet")
    results = loaded_store.search(q_vec, k=1)
    assert "Mars" in results[0][1]["text"]

def test_empty_store_behavior(store, embedder):
    """Test 3: Does searching an empty store crash?"""
    q_vec = embedder.embed_query("hello")
    results = store.search(q_vec, k=5)
    
    # Should return empty list, NOT crash, NOT throw
    assert results == []

def test_id_stability_after_reload(store, embedder, sample_data, tmp_path):
    """Test 4: Do we correctly resume ID counting after a restart?"""
    chunks, texts = sample_data
    vectors = embedder.embed_texts(texts)
    
    # Batch A: Add first 2 items (IDs 0, 1)
    store.add(chunks[:2], vectors[:2])
    
    # Save & Load
    index_path = str(tmp_path / "test.index")
    meta_path = str(tmp_path / "test.json")
    store.save(index_path, meta_path)
    new_store = VectorStore.load(index_path, meta_path)
    
    # Batch B: Add 3rd item (Should get ID 2)
    vec_b = vectors[2:3]
    chunk_b = chunks[2:]
    new_store.add(chunk_b, vec_b)
    
    # Assertions
    assert new_store.index.ntotal == 3
    
    # Crucial: Check that next_id was incremented correctly
    assert new_store.next_id == 3
    
    # Verify Metadata keys are exactly 0, 1, 2
    keys = sorted(new_store.metadata.keys())
    assert keys == [0, 1, 2]
    
    # Verify content
    assert new_store.metadata[2]["text"] == chunks[2]["text"]