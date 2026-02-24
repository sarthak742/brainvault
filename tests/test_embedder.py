import numpy as np
from embeddings.embeddings import Embedder



def test_batch_embedding_shape_and_dtype():
    embedder = Embedder()
    texts = ["Apple is a fruit", "Banana is yellow", "Mars is a planet"]

    vecs = embedder.embed_texts(texts)

    assert vecs.shape[0] == len(texts)
    assert vecs.ndim == 2
    assert vecs.dtype == np.float32


def test_embeddings_are_normalized():
    embedder = Embedder()
    texts = ["test sentence one", "test sentence two"]

    vecs = embedder.embed_texts(texts)
    norms = np.linalg.norm(vecs, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-3)


def test_empty_input_returns_empty_matrix():
    embedder = Embedder()
    vecs = embedder.embed_texts([])

    assert vecs.ndim == 2
    assert vecs.shape[0] == 0


def test_query_embedding_shape():
    embedder = Embedder()
    qvec = embedder.embed_query("fruit")

    assert qvec.ndim == 1
    assert qvec.dtype == np.float32
