"""
Pydantic models for Celery worker task payloads.

These models validate and document the data contract between the API layer
and the worker, catching malformed payloads before any processing begins.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class DocumentPayload(BaseModel):
    """
    The validated payload for a document ingestion task.

    This mirrors the API request model but lives in the worker package
    to enforce independent validation at the task boundary.
    """

    document_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the document.",
        examples=["doc-001"],
    )
    text_content: str = Field(
        ...,
        min_length=1,
        description="Raw text content of the document to be vectorized.",
        examples=["The quick brown fox jumps over the lazy dog."],
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Optional originating URL of the document.",
        examples=["https://example.com/article"],
    )

    @field_validator("document_id")
    @classmethod
    def document_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("document_id must not be blank or whitespace-only.")
        return v.strip()

    @field_validator("text_content")
    @classmethod
    def text_content_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text_content must not be blank or whitespace-only.")
        return v


class TaskResult(BaseModel):
    """Schema for the result returned by a completed processing task."""

    status: str = Field(..., description="Final task status: 'completed' or 'failed'.")
    document_id: str = Field(..., description="ID of the processed document.")
    num_chunks: int = Field(..., description="Number of chunks generated and stored.")
    message: Optional[str] = Field(default=None, description="Additional context message.")
