"""
Shared pytest fixtures and sys.modules mocks for the test suite.

WHY sys.modules MOCKING?
========================
The heavy ML dependencies (sentence-transformers, torch, chromadb) require
C-extension binaries that need a C compiler to build from source on Windows.
Since tests patch the internal service methods via unittest.mock, these packages
never need to actually execute — but Python still tries to import them at module
load time.

By inserting MagicMock stubs into sys.modules BEFORE any src.* import we
prevent the real packages from being loaded at all. This lets the test suite
run with only the lightweight packages in requirements.test.txt.

In Docker (Linux), the real packages are installed via requirements.txt.
"""

import os
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Mock heavy dependencies BEFORE any src.* imports
# ---------------------------------------------------------------------------

# --- chromadb ---
_mock_chroma_collection = MagicMock()
_mock_chroma_collection.count.return_value = 0

_mock_chroma_client = MagicMock()
_mock_chroma_client.get_or_create_collection.return_value = _mock_chroma_collection
_mock_chroma_client.heartbeat.return_value = {}

_mock_chromadb = MagicMock()
_mock_chromadb.HttpClient.return_value = _mock_chroma_client

_mock_chromadb_config = MagicMock()

sys.modules.setdefault("chromadb", _mock_chromadb)
sys.modules.setdefault("chromadb.config", _mock_chromadb_config)

# --- sentence_transformers ---
class _FakeNdarray(list):
    """Minimal numpy array stub: supports .tolist() and yields rows with .tolist()."""
    def tolist(self):
        return [item.tolist() if isinstance(item, _FakeNdarray) else item for item in self]

    def __iter__(self):
        for item in list.__iter__(self):
            if isinstance(item, list) and not isinstance(item, _FakeNdarray):
                yield _FakeNdarray(item)
            else:
                yield item


def _fake_encode(texts, **kw):
    if isinstance(texts, list):
        return _FakeNdarray([[0.1] * 384 for _ in texts])
    return _FakeNdarray([0.1] * 384)


_mock_model = MagicMock()
_mock_model.get_sentence_embedding_dimension.return_value = 384
_mock_model.encode.side_effect = _fake_encode

_mock_st = MagicMock()
_mock_st.SentenceTransformer.return_value = _mock_model

sys.modules.setdefault("sentence_transformers", _mock_st)

# ---------------------------------------------------------------------------
# 2. Ensure project root is on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# 3. Set env vars (no .env file required for tests)
# ---------------------------------------------------------------------------
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("CHROMA_HOST", "localhost")
os.environ.setdefault("CHROMA_PORT", "8000")
os.environ.setdefault("CHROMA_COLLECTION_NAME", "test_collection")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
os.environ.setdefault("FASTAPI_PORT", "8000")

# ---------------------------------------------------------------------------
# 4. Shared fixtures
# ---------------------------------------------------------------------------
import pytest


@pytest.fixture()
def mock_embedding_service():
    mock = MagicMock()
    mock.get_embedding.return_value = [0.1] * 384
    mock.get_embeddings_batch.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
    mock.embedding_dimension = 384
    return mock


@pytest.fixture()
def mock_chroma_service():
    mock = MagicMock()
    mock.get_collection_count.return_value = 0
    mock.add_embedding.return_value = None
    mock.add_embeddings_batch.return_value = None
    mock.health_check.return_value = True
    return mock


@pytest.fixture()
def sample_document():
    return {
        "document_id": "doc-fixture-001",
        "text_content": (
            "Retrieval-Augmented Generation (RAG) combines the power of "
            "large language models with external knowledge retrieval. "
            "By grounding generation in retrieved documents, RAG systems "
            "produce more accurate and up-to-date responses."
        ),
        "source_url": "https://example.com/rag-overview",
    }

# Worker test fixtures

# Task payload fixtures

# Task payload fixtures
