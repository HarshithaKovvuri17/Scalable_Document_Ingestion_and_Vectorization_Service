"""
Celery worker – document ingestion tasks.

Follows the task specification:
  celery -A src.worker.tasks worker --loglevel=info --pool=prefork -c 1

The Celery application is named `app` as specified.
EmbeddingService and ChromaService are instantiated once at module level
so the model is loaded a single time per worker process.
"""

import logging
import time
from typing import Dict, Any

from celery import Celery
from celery.utils.log import get_task_logger
from pydantic import ValidationError

from src.config.settings import get_settings, configure_logging
from src.services.embedding_service import EmbeddingService
from src.services.chroma_service import ChromaService
from src.utils.text_splitter import split_text_into_chunks
from src.worker.models import DocumentPayload, TaskResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
configure_logging()
logger = get_task_logger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
settings = get_settings()

# ---------------------------------------------------------------------------
# Celery application  (named `app` as per task spec)
# ---------------------------------------------------------------------------
app = Celery(
    "ingestion_worker",
    broker=settings.redis_url,
    backend=settings.celery_result_backend,
)

app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_reject_on_worker_shutdown=True,
    task_default_retry_delay=300,   # seconds, as per task spec
    task_max_retries=3,
    worker_prefetch_multiplier=1,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# ---------------------------------------------------------------------------
# Service singletons – loaded once per worker process
# ---------------------------------------------------------------------------
logger.info("Initialising EmbeddingService (model: %s)…", settings.EMBEDDING_MODEL_NAME)
_embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL_NAME)
logger.info("EmbeddingService ready.")

logger.info(
    "Initialising ChromaService (host=%s, port=%d, collection=%s)…",
    settings.CHROMA_HOST, settings.CHROMA_PORT, settings.CHROMA_COLLECTION_NAME,
)
_chroma_service = ChromaService(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT,
    collection_name=settings.CHROMA_COLLECTION_NAME,
)
logger.info("ChromaService ready.")


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@app.task(
    bind=True,
    name="process_document_task",
    max_retries=3,
    default_retry_delay=300,
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_document_task(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    End-to-end document ingestion pipeline:

    1. Validate payload with Pydantic.
    2. Split text into overlapping chunks (500 chars, 50 overlap).
    3. Batch-generate embeddings via Sentence Transformers.
    4. Batch-upsert all chunks into ChromaDB.
    5. ACK message only after full success (task_acks_late=True).

    Retries up to 3 times on transient failures.
    ValidationErrors fail immediately without retry.
    """
    task_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Validate payload
    # ------------------------------------------------------------------
    try:
        payload = DocumentPayload(**document_data)
    except ValidationError as exc:
        logger.error(
            "Task %s: payload validation failed for document '%s': %s",
            self.request.id,
            document_data.get("document_id", "<unknown>"),
            exc,
        )
        raise  # No retry for malformed payloads

    document_id = payload.document_id
    text_content = payload.text_content
    source_url = payload.source_url

    logger.info(
        "Processing document: %s (text_length=%d, source_url=%s)",
        document_id, len(text_content), source_url or "N/A",
    )

    try:
        # ------------------------------------------------------------------
        # 2. Text chunking
        # ------------------------------------------------------------------
        chunks = split_text_into_chunks(text_content, chunk_size=500, chunk_overlap=50)

        if not chunks:
            logger.warning("Document '%s' produced zero chunks.", document_id)
            result = TaskResult(
                status="completed", document_id=document_id, num_chunks=0,
                message="No chunks generated; document may be empty.",
            )
            return result.model_dump()

        logger.info("Document '%s' split into %d chunk(s).", document_id, len(chunks))

        # ------------------------------------------------------------------
        # 3. Batch embedding generation
        # ------------------------------------------------------------------
        embeddings = _embedding_service.get_embeddings_batch(chunks)

        # ------------------------------------------------------------------
        # 4. Batch upsert into ChromaDB
        # ------------------------------------------------------------------
        chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "chunk_index": i,
                "source_url": source_url or "",
                "text_content": chunks[i],
            }
            for i in range(len(chunks))
        ]

        _chroma_service.add_embeddings_batch(
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        elapsed = time.perf_counter() - task_start
        logger.info(
            "Document '%s' ingested successfully. chunks=%d, elapsed=%.3fs.",
            document_id, len(chunks), elapsed,
        )

        result = TaskResult(
            status="completed",
            document_id=document_id,
            num_chunks=len(chunks),
            message=f"Processed in {elapsed:.3f}s",
        )
        return result.model_dump()

    except Exception as exc:
        elapsed = time.perf_counter() - task_start
        logger.error(
            "Error processing document '%s' after %.3fs: %s",
            document_id, elapsed, exc, exc_info=True,
        )
        raise self.retry(exc=exc)

# Exponential backoff retry policy
