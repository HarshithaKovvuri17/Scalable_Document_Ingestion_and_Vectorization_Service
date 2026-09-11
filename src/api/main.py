"""
FastAPI application – Document Ingestion Service.

Follows the task specification:
  - POST /ingest  → 202 Accepted + {task_id, status}
  - GET  /health  → 200 OK + {"status": "ok"}
  - Returns 400 Bad Request for missing/blank document_id or text_content.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.config.settings import configure_logging, get_settings
from src.worker.tasks import app as celery_app
from src.worker.tasks import process_document_task

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
configure_logging()
logger = logging.getLogger(__name__)

settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(
        "Document Ingestion Service starting. Redis=%s:%d | Chroma=%s:%d | Model=%s",
        settings.REDIS_HOST, settings.REDIS_PORT,
        settings.CHROMA_HOST, settings.CHROMA_PORT,
        settings.EMBEDDING_MODEL_NAME,
    )
    yield
    logger.info("Document Ingestion Service shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document Ingestion Service",
    description=(
        "Production-ready backend service for a RAG system. "
        "Ingests documents, chunks and vectorizes them asynchronously, "
        "and stores embeddings in ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Override 422 validation errors → 400 Bad Request (as per task spec)
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Convert Pydantic validation errors to 400 Bad Request per task spec."""
    errors = exc.errors()
    details = "; ".join(
        f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
        for e in errors
    )
    logger.warning("Request validation failed on %s: %s", request.url, details)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": f"Document ID and text content are required. {details}"},
    )


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class IngestDocumentRequest(BaseModel):
    """Payload for POST /ingest."""

    document_id: str = Field(..., min_length=1, description="Unique document identifier.")
    text_content: str = Field(..., min_length=1, description="Raw text to vectorize.")
    source_url: Optional[str] = Field(default=None, description="Optional origin URL.")

    @field_validator("document_id")
    @classmethod
    def document_id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("document_id must not be blank.")
        return v.strip()

    @field_validator("text_content")
    @classmethod
    def text_content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text_content must not be blank.")
        return v


class TaskStatusResponse(BaseModel):
    task_id: str
    state: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest", status_code=202)
async def ingest_document(request: IngestDocumentRequest):
    """
    Accepts a document for asynchronous processing and vectorization.

    Returns 202 Accepted immediately with a task_id for tracking.
    Returns 400 Bad Request if document_id or text_content is missing or blank.
    """
    if not request.document_id or not request.text_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document ID and text content are required.",
        )

    logger.info(
        "Received ingest request: document_id='%s', text_length=%d",
        request.document_id, len(request.text_content),
    )

    try:
        task = process_document_task.delay(request.model_dump())
    except Exception as exc:
        logger.error("Failed to enqueue task for document '%s': %s", request.document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enqueue document processing task. Please try again.",
        ) from exc

    logger.info("Task %s enqueued for document '%s'.", task.id, request.document_id)

    return {"task_id": task.id, "status": "Processing enqueued"}


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Poll the status of a submitted ingestion task."""
    result = AsyncResult(task_id, app=celery_app)
    response = TaskStatusResponse(task_id=task_id, state=result.state)
    if result.state == "SUCCESS":
        response.result = result.result
    elif result.state == "FAILURE":
        response.error = str(result.result)
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint – returns 200 OK when the service is operational."""
    return {"status": "ok"}

# Validation error override handler

# HTTP 400 validation error override
