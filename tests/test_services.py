"""
Unit tests for EmbeddingService and ChromaService.

External dependencies (SentenceTransformer, chromadb.HttpClient) are
replaced with MagicMocks so the tests have no infrastructure requirements.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# Minimal numpy array stub for test_services.py (numpy is not installed locally)
class _FakeNdarray(list):
    """Mimics a 1-D or 2-D numpy ndarray for EmbeddingService tests.

    * .tolist() returns a plain Python list.
    * Iteration over a 2-D array yields _FakeNdarray rows (each with .tolist()).
    """
    def tolist(self):
        return [item.tolist() if isinstance(item, _FakeNdarray) else item for item in self]

    def __iter__(self):
        for item in list.__iter__(self):
            if isinstance(item, list) and not isinstance(item, _FakeNdarray):
                yield _FakeNdarray(item)
            else:
                yield item


# ---------------------------------------------------------------------------
# EmbeddingService tests
# ---------------------------------------------------------------------------

class TestEmbeddingService:
    """Tests for src.services.embedding_service.EmbeddingService."""

    @patch("src.services.embedding_service.SentenceTransformer")
    def test_model_loaded_on_init(self, mock_st_cls):
        """Verify the model is loaded exactly once during construction."""
        from src.services.embedding_service import EmbeddingService

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_cls.return_value = mock_model

        svc = EmbeddingService("all-MiniLM-L6-v2")

        mock_st_cls.assert_called_once_with("all-MiniLM-L6-v2")
        assert svc.model_name == "all-MiniLM-L6-v2"

    @patch("src.services.embedding_service.SentenceTransformer")
    def test_get_embedding_returns_list_of_floats(self, mock_st_cls):
        """get_embedding must return a list of floats."""
        from src.services.embedding_service import EmbeddingService

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        fake_vector = _FakeNdarray([0.1] * 384)
        mock_model.encode.return_value = fake_vector
        mock_st_cls.return_value = mock_model

        svc = EmbeddingService("all-MiniLM-L6-v2")
        result = svc.get_embedding("hello world")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(v, float) for v in result)

    @patch("src.services.embedding_service.SentenceTransformer")
    def test_get_embedding_raises_for_empty_text(self, mock_st_cls):
        """get_embedding must raise ValueError for empty input."""
        from src.services.embedding_service import EmbeddingService

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_cls.return_value = mock_model

        svc = EmbeddingService("all-MiniLM-L6-v2")

        with pytest.raises(ValueError, match="empty text"):
            svc.get_embedding("")

    @patch("src.services.embedding_service.SentenceTransformer")
    def test_get_embedding_raises_for_whitespace_text(self, mock_st_cls):
        from src.services.embedding_service import EmbeddingService

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_cls.return_value = mock_model

        svc = EmbeddingService("all-MiniLM-L6-v2")

        with pytest.raises(ValueError, match="empty text"):
            svc.get_embedding("   ")

    @patch("src.services.embedding_service.SentenceTransformer")
    def test_get_embeddings_batch_returns_list_of_lists(self, mock_st_cls):
        """get_embeddings_batch must return one vector per input text."""
        from src.services.embedding_service import EmbeddingService

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        texts = ["first chunk", "second chunk", "third chunk"]
        # _FakeNdarray is a list of lists; iteration gives each row as a list
        fake_matrix = _FakeNdarray([[0.1] * 384 for _ in texts])
        mock_model.encode.return_value = fake_matrix
        mock_st_cls.return_value = mock_model

        svc = EmbeddingService("all-MiniLM-L6-v2")
        result = svc.get_embeddings_batch(texts)

        assert isinstance(result, list)
        assert len(result) == len(texts)
        for vec in result:
            assert isinstance(vec, list)
            assert len(vec) == 384

    @patch("src.services.embedding_service.SentenceTransformer")
    def test_get_embeddings_batch_raises_for_empty_list(self, mock_st_cls):
        from src.services.embedding_service import EmbeddingService

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_cls.return_value = mock_model

        svc = EmbeddingService("all-MiniLM-L6-v2")

        with pytest.raises(ValueError, match="must not be empty"):
            svc.get_embeddings_batch([])

    @patch("src.services.embedding_service.SentenceTransformer")
    def test_get_embeddings_batch_raises_for_blank_element(self, mock_st_cls):
        from src.services.embedding_service import EmbeddingService

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_cls.return_value = mock_model

        svc = EmbeddingService("all-MiniLM-L6-v2")

        with pytest.raises(ValueError, match="index 1 is empty"):
            svc.get_embeddings_batch(["valid text", "   "])

    @patch("src.services.embedding_service.SentenceTransformer")
    def test_embedding_dimension_property(self, mock_st_cls):
        from src.services.embedding_service import EmbeddingService

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st_cls.return_value = mock_model

        svc = EmbeddingService("some-model")
        assert svc.embedding_dimension == 768


# ---------------------------------------------------------------------------
# ChromaService tests
# ---------------------------------------------------------------------------

class TestChromaService:
    """Tests for src.services.chroma_service.ChromaService."""

    @patch("src.services.chroma_service.chromadb.HttpClient")
    def test_connects_on_init_and_creates_collection(self, mock_http_cls):
        """ChromaService must create/retrieve the collection during __init__."""
        from src.services.chroma_service import ChromaService

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_http_cls.return_value = mock_client

        svc = ChromaService(host="localhost", port=8000, collection_name="test_col")

        mock_client.get_or_create_collection.assert_called_once_with(
            name="test_col",
            metadata={"hnsw:space": "cosine"},
        )

    @patch("src.services.chroma_service.chromadb.HttpClient")
    def test_add_embedding_calls_upsert(self, mock_http_cls):
        """add_embedding must call collection.upsert with correct arguments."""
        from src.services.chroma_service import ChromaService

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_http_cls.return_value = mock_client

        svc = ChromaService()
        embedding = [0.1] * 384
        metadata = {"document_id": "doc1", "chunk_index": 0, "source_url": ""}

        svc.add_embedding("doc1_chunk_0", embedding, "chunk text", metadata)

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert call_kwargs["ids"] == ["doc1_chunk_0"]
        assert call_kwargs["embeddings"] == [embedding]
        assert call_kwargs["documents"] == ["chunk text"]

    @patch("src.services.chroma_service.chromadb.HttpClient")
    def test_add_embeddings_batch_upserts_all(self, mock_http_cls):
        """add_embeddings_batch must upsert all provided items in one call."""
        from src.services.chroma_service import ChromaService

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_http_cls.return_value = mock_client

        svc = ChromaService()
        ids = ["doc1_chunk_0", "doc1_chunk_1"]
        embeddings = [[0.1] * 384, [0.2] * 384]
        documents = ["chunk 0", "chunk 1"]
        metadatas = [
            {"document_id": "doc1", "chunk_index": 0, "source_url": ""},
            {"document_id": "doc1", "chunk_index": 1, "source_url": ""},
        ]

        svc.add_embeddings_batch(ids, embeddings, documents, metadatas)

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert call_kwargs["ids"] == ids

    @patch("src.services.chroma_service.chromadb.HttpClient")
    def test_add_embeddings_batch_skips_empty_list(self, mock_http_cls):
        """add_embeddings_batch must be a no-op when called with empty lists."""
        from src.services.chroma_service import ChromaService

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_http_cls.return_value = mock_client

        svc = ChromaService()
        svc.add_embeddings_batch([], [], [], [])

        mock_collection.upsert.assert_not_called()

    @patch("src.services.chroma_service.chromadb.HttpClient")
    def test_get_collection_count(self, mock_http_cls):
        from src.services.chroma_service import ChromaService

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_http_cls.return_value = mock_client

        svc = ChromaService()
        assert svc.get_collection_count() == 42

    @patch("src.services.chroma_service.chromadb.HttpClient")
    def test_health_check_returns_true_on_success(self, mock_http_cls):
        from src.services.chroma_service import ChromaService

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_http_cls.return_value = mock_client

        svc = ChromaService()
        assert svc.health_check() is True

    @patch("src.services.chroma_service.chromadb.HttpClient")
    def test_none_metadata_values_sanitised(self, mock_http_cls):
        """None values in metadata must be converted to empty strings."""
        from src.services.chroma_service import ChromaService, _sanitize_metadata

        raw = {"document_id": "d1", "source_url": None, "chunk_index": 0}
        sanitized = _sanitize_metadata(raw)
        assert sanitized["source_url"] == ""
        assert sanitized["document_id"] == "d1"
        assert sanitized["chunk_index"] == 0

# ChromaDB connection retry tests
