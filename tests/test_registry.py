import pytest
import os
import json
from chunking.registry import ChunkRegistry

# --- Fixtures ---

@pytest.fixture
def clean_registry_file(tmp_path):
    """Provides a temporary path for registry JSON."""
    return str(tmp_path / "registry_test.json")

@pytest.fixture
def sample_chunks():
    return [
        {"text": "Alpha", "source": "doc1", "start_page": 1},
        {"text": "Beta", "source": "doc1", "start_page": 2},
        {"text": "Gamma", "source": "doc2", "start_page": 1},
    ]

# --- Core Logic Tests ---

def test_registry_deterministic_ids(sample_chunks):
    """Ensure identical content yields identical IDs."""
    reg = ChunkRegistry()
    reg.add_chunks(sample_chunks)
    
    chunks = reg.get_chunks()
    id_alpha = chunks[0]["chunk_id"]
    
    # Re-generate manually
    expected_id = reg._generate_chunk_id(sample_chunks[0])
    assert id_alpha == expected_id

def test_registry_deduplication(sample_chunks):
    """Ensure adding the same chunk twice does not duplicate it."""
    reg = ChunkRegistry()
    
    # Add once
    reg.add_chunks([sample_chunks[0]])
    assert len(reg.get_chunks()) == 1
    
    # Add again (identical content)
    reg.add_chunks([sample_chunks[0]])
    assert len(reg.get_chunks()) == 1  # Should still be 1

def test_registry_insertion_order(sample_chunks):
    """
    CRITICAL: Verify insertion order is preserved.
    This is required for FAISS index alignment.
    """
    reg = ChunkRegistry()
    reg.add_chunks(sample_chunks) # Alpha, Beta, Gamma
    
    stored = reg.get_chunks()
    assert stored[0]["text"] == "Alpha"
    assert stored[1]["text"] == "Beta"
    assert stored[2]["text"] == "Gamma"
    
    # Verify index lookup matches
    assert reg.get_chunk_by_index(0)["text"] == "Alpha"
    assert reg.get_chunk_by_index(2)["text"] == "Gamma"
    assert reg.get_chunk_by_index(99) is None

# --- Persistence Tests ---

def test_registry_save_load_roundtrip(clean_registry_file, sample_chunks):
    """Verify data survives save/load intact."""
    reg_original = ChunkRegistry()
    reg_original.add_chunks(sample_chunks)
    reg_original.save(clean_registry_file)
    
    # Load into new instance
    reg_loaded = ChunkRegistry.load(clean_registry_file)
    
    assert len(reg_loaded.get_chunks()) == 3
    # Check ID preservation
    assert reg_loaded.get_chunks()[0]["chunk_id"] == reg_original.get_chunks()[0]["chunk_id"]

def test_registry_fail_fast_corruption(clean_registry_file):
    """Verify load raises ValueError on missing IDs (Corrupt file)."""
    # Create a corrupt file manually
    corrupt_data = {
        "version": 1,
        "chunks": [{"text": "I have no ID"}] # Missing 'chunk_id'
    }
    
    with open(clean_registry_file, 'w') as f:
        json.dump(corrupt_data, f)
        
    # Attempt load
    with pytest.raises(ValueError) as exc:
        ChunkRegistry.load(clean_registry_file)
    
    assert "CRITICAL CORRUPTION" in str(exc.value)

def test_registry_trust_existing_ids(clean_registry_file):
    """
    Verify that if chunk_id exists in input, it is respected.
    This allows reloading without ID drift.
    """
    reg = ChunkRegistry()
    # Chunk WITH explicit ID
    chunk_with_id = {"text": "Foo", "chunk_id": "explicit_id_123"}
    
    reg.add_chunks([chunk_with_id])
    
    stored = reg.get_chunks()[0]
    assert stored["chunk_id"] == "explicit_id_123"
    # It should NOT have generated a new hash