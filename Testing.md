# Complete Testing Guide

This guide provides step-by-step instructions for testing the **Scalable Document Ingestion and Vectorization Service**. It covers local unit and integration testing using `pytest`, full end-to-end containerized testing with Docker Compose, and manual API verification via PowerShell (`Invoke-RestMethod` / `curl.exe`).

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Automated Testing with Pytest](#2-automated-testing-with-pytest)
   - [Running the Test Suite](#running-the-test-suite)
   - [Test Suite Breakdown](#test-suite-breakdown)
3. [End-to-End Containerized Testing (Docker)](#3-end-to-end-containerized-testing-docker)
   - [Starting Services](#starting-services)
   - [Verifying Services](#verifying-services)
4. [Manual API Endpoint Testing](#4-manual-api-endpoint-testing)
   - [1. Health Check Endpoint](#1-health-check-endpoint)
   - [2. Document Ingestion Endpoint](#2-document-ingestion-endpoint)
   - [3. Task Status Endpoint](#3-task-status-endpoint)
5. [Error & Edge Case Testing](#5-error--edge-case-testing)
6. [Celery Task Retry & Error Resiliency Testing](#6-celery-task-retry--error-resiliency-testing)
7. [Verification Checklist](#7-verification-checklist)

---

## 1. Prerequisites

Ensure your development environment has the following installed:
- **Python 3.9+** (or Python 3.13 inside local `.venv`)
- **Docker** and **Docker Compose** (for end-to-end multi-container testing)
- **PowerShell 5.1+** or **PowerShell Core 7+** (for manual HTTP request testing via `Invoke-RestMethod` or `curl.exe`)

---

## 2. Automated Testing with Pytest

The automated test suite uses `pytest` and mocks external dependencies (ChromaDB and HuggingFace Transformers) to ensure fast, reproducible execution without requiring running databases or network calls.

### Running the Test Suite

Run all unit and component tests from the repository root:

```powershell
# Run pytest in verbose mode
.\.venv\Scripts\Activate.ps1
pytest tests/ -v

# Run with output capture disabled (to view live logs)
.\.venv\Scripts\Activate.ps1
pytest tests/ -v -s
```

---

### Test Suite Breakdown

| Module | Location | Description | Test Count |
| :--- | :--- | :--- | :---: |
| **API Endpoints** | `tests/test_api.py` | Validates FastAPI request validation, `400` status overrides, task submission responses, and `/health` response. | 17 |
| **Core Services** | `tests/test_services.py` | Tests `EmbeddingService` vector generation (384-d), batching, and `ChromaService` client initialization & batched upserts. | 17 |
| **Utilities** | `tests/test_utils.py` | Validates text chunking logic (500-char windowing, 50-char overlap, boundary conditions, invalid parameters). | 19 |
| **Celery Worker** | `tests/test_worker.py` | Tests async document processing task execution, payload validation, ChromaDB batched upserts, and retry behavior. | 12 |
| **Total** | | | **65 Passed** |

---

## 3. End-to-End Containerized Testing (Docker)

To test the complete workflow using live containers for **FastAPI**, **Celery Worker**, **Redis**, and **ChromaDB**:

### Starting Services

1. Build and start all services in detached mode:
   ```powershell
   docker-compose up --build -d
   ```

2. Check container status:
   ```powershell
   docker-compose ps
   ```
   All 4 services (`api`, `worker`, `redis`, `chroma`) should report status `Up`.

3. Stream container logs:
   ```powershell
   docker-compose logs -f
   ```
   Or check specific worker logs:
   ```powershell
   docker-compose logs -f worker
   ```

---

## 4. Manual API Endpoint Testing

### 1. Health Check Endpoint

Verify that the FastAPI service is running and healthy:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
# OR using curl.exe
curl.exe -i -X GET http://localhost:8000/health
```

**Expected Response:**
```http
HTTP/1.1 200 OK
content-type: application/json

{"status":"ok"}
```

---

## 2. Document Ingestion Endpoint

Submit a document payload to trigger asynchronous vectorization and storage:

```powershell
$body = @{
    document_id = "doc_test_001"
    text_content = "Retrieval-Augmented Generation (RAG) combines computational search with large language models to generate grounded responses."
    source_url = "https://example.com/rag-paper.pdf"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/ingest" -Method Post -ContentType "application/json" -Body $body

# OR using curl.exe
curl.exe -i -X POST "http://localhost:8000/ingest" -H "Content-Type: application/json" -d "{\"document_id\": \"doc_test_001\", \"text_content\": \"Retrieval-Augmented Generation text content.\", \"source_url\": \"https://example.com/rag-paper.pdf\"}"
```

**Expected Response:**
```http
HTTP/1.1 202 Accepted
content-type: application/json

{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "Processing enqueued"
}
```

---

## 3. Task Status Endpoint

Check the execution status of a submitted Celery task using the `task_id` returned from `/ingest`:

```powershell
# Store the task_id returned from /ingest (do not include angle brackets < >)
$taskId = "2aac03b3-af46-43f2-ac34-d70756d0c45f"
Invoke-RestMethod -Uri "http://localhost:8000/task/$taskId" -Method Get

# OR using curl.exe
curl.exe -i -X GET "http://localhost:8000/task/$taskId"
```

**Expected Response (Completed Task):**
```http
HTTP/1.1 200 OK
content-type: application/json

{
  "task_id": "2aac03b3-af46-43f2-ac34-d70756d0c45f",
  "state": "SUCCESS",
  "result": {
    "status": "completed",
    "document_id": "doc_test_001",
    "num_chunks": 1,
    "message": "Processed in 0.561s"
  }
}
```

---

## 5. Error & Edge Case Testing

Test how the service handles invalid requests. Per specification, all invalid requests return `HTTP 400 Bad Request` with structured error details.

### Test Case 1: Missing Required `document_id`

```powershell
$body = @{ text_content = "Valid text content but missing document_id" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/ingest" -Method Post -ContentType "application/json" -Body $body
```
**Expected Response:** `HTTP 400 Bad Request`

---

### Test Case 2: Blank / Whitespace `text_content`

```powershell
$body = @{ text_content = "Valid text content but missing document_id" } | ConvertTo-Json

try {
    Invoke-RestMethod -Uri "http://localhost:8000/ingest" -Method Post -ContentType "application/json" -Body $body
} catch {
    Write-Host "HTTP Status:" $_.Exception.Response.StatusCode.value__
    Write-Host "Response Body:" $_.ErrorDetails.Message
}

```
**Expected Response:** `HTTP 400 Bad Request`

---

### Test Case 3: Invalid JSON Body

```powershell
try {
    Invoke-RestMethod -Uri "http://localhost:8000/ingest" -Method Post -ContentType "application/json" -Body "not json"
} catch {
    Write-Host "HTTP Status:" $_.Exception.Response.StatusCode.value__
    Write-Host "Response Body:" $_.ErrorDetails.Message
}

```
**Expected Response:** `HTTP 400 Bad Request`

---

## 6. Celery Task Retry & Error Resiliency Testing

The Celery worker includes exponential backoff retry logic (`task_default_retry_delay=300`, `max_retries=3`) for transient ChromaDB connection issues:

1. Submit a document via `/ingest`.
2. Stop the `chroma` container to simulate database failure:
   ```powershell
   docker-compose stop chroma
   ```
3. Observe worker logs (`docker-compose logs -f worker`) to confirm retry attempts with warning logs.
4. Restart `chroma`:
   ```powershell
   docker-compose start chroma
   ```
5. Confirm task completes successfully on retry.

---

## 7. Verification Checklist

- [x] **Pytest Unit & Integration Tests**: 65/65 tests pass.
- [x] **Health Check**: `GET /health` returns `200 OK` with `{"status": "ok"}`.
- [x] **Document Ingestion**: `POST /ingest` queues task and returns `202 Accepted`.
- [x] **Task Status Monitoring**: `GET /task/{task_id}` correctly tracks `PENDING` $\rightarrow$ `SUCCESS`.
- [x] **Validation Error Override**: Request validation failures return `400 Bad Request` instead of `422`.
- [x] **Chunking & Vectorization**: Text is chunked into 500-character windows with 50-character overlap and embedded into 384-dimensional vectors stored in ChromaDB.

<!-- Edge case PowerShell instructions -->
