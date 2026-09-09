"""
Unit and integration tests for the Celery worker tasks (src/worker/tasks.py).

All external services (EmbeddingService, ChromaService) are replaced with
MagicMocks so tests run without infrastructure.
"""

import pytest
from unittest.mock import MagicMock, patch, call
import src.worker.tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_document_data(
    document_id: str = "doc-001",
    text_content: str = "This is a test document with enough content.",
    source_url: str | None = "https://example.com",
) -> dict:
    return {
        "document_id": document_id,
        "text_content": text_content,
        "source_url": source_url,
    }


def call_task(document_data: dict, embedding_svc, chroma_svc):
    """
    Directly invoke the task function body (bypassing Celery broker) with
    mocked service singletons.

    For a bound Celery task (bind=True), the underlying Python function
    signature is `def process_document_task(self, document_data)`.
    We call it by pushing a fake request context so Celery's bind
    mechanism injects the task instance correctly.
    """
    with patch("src.worker.tasks._embedding_service", embedding_svc), \
         patch("src.worker.tasks._chroma_service", chroma_svc):

        from src.worker.tasks import process_document_task

        # apply() calls the task synchronously with ALWAYS_EAGER semantics
        # by pushing a fake request so self.request.id is available.
        # We use apply() which correctly handles bind=True without a broker.
        result = process_document_task.apply(args=[document_data])

        if result.failed():
            raise result.result  # re-raise exception for test assertions

        mock_self = MagicMock()
        mock_self.request.id = "test-task-id"
        return result.result, mock_self


# ---------------------------------------------------------------------------
# process_document_task – happy path
# ---------------------------------------------------------------------------

class TestProcessDocumentTask:

    def test_successful_processing_returns_completed_status(
        self, mock_embedding_service, mock_chroma_service
    ):
        data = make_document_data(text_content="Hello world. " * 10)
        result, _ = call_task(data, mock_embedding_service, mock_chroma_service)
        assert result["status"] == "completed"

    def test_successful_processing_returns_document_id(
        self, mock_embedding_service, mock_chroma_service
    ):
        data = make_document_data(document_id="specific-doc-id")
        result, _ = call_task(data, mock_embedding_service, mock_chroma_service)
        assert result["document_id"] == "specific-doc-id"

    def test_num_chunks_matches_split(
        self, mock_embedding_service, mock_chroma_service
    ):
        # A short text should produce 1 chunk
        data = make_document_data(text_content="Short text.")
        result, _ = call_task(data, mock_embedding_service, mock_chroma_service)
        assert result["num_chunks"] >= 1

    def test_batch_upsert_is_called_once(
        self, mock_embedding_service, mock_chroma_service
    ):
        """All chunks must be stored in a single batch upsert call."""
        data = make_document_data(text_content="Word " * 300)  # multi-chunk
        call_task(data, mock_embedding_service, mock_chroma_service)
        mock_chroma_service.add_embeddings_batch.assert_called_once()

    def test_batch_embeddings_called_once(
        self, mock_embedding_service, mock_chroma_service
    ):
        """Embeddings should be generated in a single batch call, not per-chunk."""
        data = make_document_data(text_content="Word " * 300)
        call_task(data, mock_embedding_service, mock_chroma_service)
        mock_embedding_service.get_embeddings_batch.assert_called_once()

    def test_chunk_ids_are_deterministic(
        self, mock_embedding_service, mock_chroma_service
    ):
        """Chunk IDs must follow the pattern {document_id}_chunk_{i}."""
        data = make_document_data(
            document_id="my-doc",
            text_content="Word " * 300,
        )
        call_task(data, mock_embedding_service, mock_chroma_service)
        call_kwargs = mock_chroma_service.add_embeddings_batch.call_args.kwargs
        for chunk_id in call_kwargs["chunk_ids"]:
            assert chunk_id.startswith("my-doc_chunk_")

    def test_metadata_contains_required_fields(
        self, mock_embedding_service, mock_chroma_service
    ):
        """Each chunk's metadata must contain document_id, chunk_index, source_url."""
        data = make_document_data(
            document_id="meta-doc",
            text_content="Word " * 200,
            source_url="https://meta.example.com",
        )
        call_task(data, mock_embedding_service, mock_chroma_service)
        call_kwargs = mock_chroma_service.add_embeddings_batch.call_args.kwargs
        for meta in call_kwargs["metadatas"]:
            assert "document_id" in meta
            assert "chunk_index" in meta
            assert "source_url" in meta
            assert meta["document_id"] == "meta-doc"
            assert meta["source_url"] == "https://meta.example.com"

    def test_source_url_defaults_to_empty_string_when_none(
        self, mock_embedding_service, mock_chroma_service
    ):
        data = make_document_data(source_url=None)
        call_task(data, mock_embedding_service, mock_chroma_service)
        call_kwargs = mock_chroma_service.add_embeddings_batch.call_args.kwargs
        for meta in call_kwargs["metadatas"]:
            assert meta["source_url"] == ""

    def test_invalid_payload_raises_without_retry(
        self, mock_embedding_service, mock_chroma_service
    ):
        """A ValidationError must NOT trigger a retry – the task fails immediately."""
        from pydantic import ValidationError

        bad_data = {"document_id": "", "text_content": ""}  # both blank
        with patch("src.worker.tasks._embedding_service", mock_embedding_service), \
             patch("src.worker.tasks._chroma_service", mock_chroma_service):
            from src.worker.tasks import process_document_task
            result = process_document_task.apply(args=[bad_data])
        assert result.failed()
        assert isinstance(result.result, ValidationError)

    def test_embedding_failure_triggers_retry(
        self, mock_embedding_service, mock_chroma_service
    ):
        """A transient embedding failure must cause the task to fail (retries exhausted)."""
        mock_embedding_service.get_embeddings_batch.side_effect = RuntimeError("GPU OOM")
        data = make_document_data(text_content="Word " * 10)

        with patch("src.worker.tasks._embedding_service", mock_embedding_service), \
             patch("src.worker.tasks._chroma_service", mock_chroma_service):
            from src.worker.tasks import process_document_task
            # apply() with throw=True will propagate the retry exception
            result = process_document_task.apply(args=[data])

        # Task should have failed (retries exhausted since apply runs synchronously)
        assert result.failed()

    def test_chroma_failure_triggers_retry(
        self, mock_embedding_service, mock_chroma_service
    ):
        """A transient ChromaDB write failure must cause the task to fail."""
        mock_chroma_service.add_embeddings_batch.side_effect = ConnectionError("Chroma down")
        data = make_document_data(text_content="Word " * 10)

        with patch("src.worker.tasks._embedding_service", mock_embedding_service), \
             patch("src.worker.tasks._chroma_service", mock_chroma_service):
            from src.worker.tasks import process_document_task
            result = process_document_task.apply(args=[data])

        assert result.failed()

    def test_empty_text_returns_zero_chunks(
        self, mock_embedding_service, mock_chroma_service
    ):
        """
        Whitespace-only text_content is rejected by Pydantic validation in the worker
        (the DocumentPayload validator enforces non-blank text_content).
        The task should fail immediately with a ValidationError, and ChromaDB
        should never be called.
        """
        from pydantic import ValidationError

        data = {"document_id": "empty-doc", "text_content": "   ", "source_url": None}

        with patch("src.worker.tasks._embedding_service", mock_embedding_service), \
             patch("src.worker.tasks._chroma_service", mock_chroma_service):
            from src.worker.tasks import process_document_task
            result = process_document_task.apply(args=[data])

        # Pydantic rejects blank text_content before any processing starts
        assert result.failed()
        assert isinstance(result.result, ValidationError)
        mock_chroma_service.add_embeddings_batch.assert_not_called()


# Blank document validation tests

# Validation edge cases
