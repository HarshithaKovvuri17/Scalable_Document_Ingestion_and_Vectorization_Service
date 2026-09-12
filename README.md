# Scalable Document Ingestion and Vectorization Service

A scalable backend service for asynchronously ingesting documents, splitting text into manageable chunks, generating vector embeddings, and storing those embeddings in ChromaDB for semantic search and Retrieval-Augmented Generation (RAG) applications.

---

# 🚀 Project Overview

The **Scalable Document Ingestion and Vectorization Service** is a backend service designed to process documents asynchronously and convert their text into vector embeddings.

The project is designed as the **document ingestion and vectorization layer of a Retrieval-Augmented Generation (RAG) system**.

The service accepts a document through a REST API, validates the input, places the processing request into a Celery task queue, processes the document using a background worker, splits the text into smaller chunks, generates vector embeddings using a Sentence Transformer model, and stores the resulting vectors in ChromaDB.

The API remains responsive because expensive document-processing operations are performed asynchronously.

### High-Level Flow

```text
Document
   |
   v
FastAPI
   |
   v
Redis
   |
   v
Celery Worker
   |
   +----> Text Chunking
   |
   +----> Embedding Generation
   |
   +----> Metadata Creation
   |
   v
ChromaDB
   |
   v
Vector Storage
```

---

# 🎯 Problem Statement

When large documents are used in RAG or semantic-search systems, the complete document cannot always be processed efficiently as a single piece of text.

A document usually needs to go through several stages:

1. Text extraction or reception.
2. Text validation.
3. Text chunking.
4. Embedding generation.
5. Vector storage.
6. Status tracking.
7. Error handling and retry processing.

If all these operations are performed synchronously inside an API request, the API can become slow and unresponsive.

This project solves that problem by separating:

* API request handling
* Background task processing
* Embedding generation
* Vector database operations

---

# 🎯 Project Objectives

The main objectives of the project are:

* Build a REST API for document ingestion.
* Process documents asynchronously.
* Use Redis as a task broker and result backend.
* Use Celery for background processing.
* Split documents into smaller overlapping chunks.
* Generate semantic vector embeddings.
* Store embeddings in ChromaDB.
* Support batch embedding generation.
* Support batch vector storage.
* Provide task-status monitoring.
* Implement request validation.
* Implement retry handling for transient failures.
* Make processing idempotent.
* Containerize the complete system using Docker.
* Provide automated testing using pytest.

---

# ⭐ Key Features

## 1. Asynchronous Document Processing

The `/ingest` API does not perform the complete vectorization process synchronously.

Instead, it creates a Celery task and immediately returns a task ID.

This prevents long-running embedding and database operations from blocking the API.

---

## 2. FastAPI REST API

The project exposes three main API endpoints:

| Method | Endpoint          | Purpose           |
| ------ | ----------------- | ----------------- |
| GET    | `/health`         | Check API health  |
| POST   | `/ingest`         | Submit a document |
| GET    | `/task/{task_id}` | Check task status |

---

## 3. Request Validation

The API validates incoming document requests using Pydantic.

The request contains:

* `document_id`
* `text_content`
* `source_url` — optional

Invalid requests are rejected.

The project also overrides FastAPI's default validation behavior so validation errors are returned as:

```text
HTTP 400 Bad Request
```

instead of the default HTTP 422 response.

---

## 4. Text Chunking

The main worker pipeline uses:

```text
Chunk Size : 500 characters
Overlap    : 50 characters
```

The overlap allows neighboring chunks to share some contextual information.

---

## 5. Batch Embedding Generation

The project uses:

```text
Sentence Transformers
```

with:

```text
all-MiniLM-L6-v2
```

The model generates:

```text
384-dimensional embeddings
```

All chunks belonging to a document are embedded as a batch rather than making one model call for every chunk.

---

## 6. ChromaDB Vector Storage

The generated embeddings are stored in ChromaDB.

Each stored chunk contains:

* Chunk ID
* Chunk text
* Embedding vector
* Document metadata

The ChromaDB collection is configured to use cosine similarity.

```text
hnsw:space = cosine
```

---

## 7. Batch Upsert

Instead of individually inserting every chunk, the worker performs a batch upsert.

```text
Chunk 1
Chunk 2
Chunk 3
Chunk 4
   |
   v
Batch Upsert
   |
   v
ChromaDB
```

This reduces database round trips and improves efficiency.

---

## 8. Idempotent Processing

Chunk IDs are deterministic.

The format is:

```text
{document_id}_chunk_{index}
```

For example:

```text
doc_001_chunk_0
doc_001_chunk_1
doc_001_chunk_2
```

Because ChromaDB uses `upsert`, processing the same document again does not unnecessarily create duplicate chunks.

This is especially useful when a Celery task is retried.

---

## 9. Celery Retry Support

The worker supports retry handling for transient failures.

Configuration:

```text
Maximum Retries : 3
Retry Delay     : 300 seconds
```

For example, if ChromaDB becomes temporarily unavailable, the worker can retry the task.

Invalid input is not retried because retrying the same invalid payload would not fix the problem.

---

## 10. Persistent Vector Storage

ChromaDB uses a Docker volume named:

```text
chroma_data
```

This allows stored vector data to persist even when the ChromaDB container is recreated.

---

# 🏗️ System Architecture

The system consists of four main runtime services:

1. FastAPI
2. Redis
3. Celery Worker
4. ChromaDB

### Architecture Diagram

```text
                         CLIENT
                    cURL / Postman
                          |
                          | HTTP
                          v
                +----------------------+
                |      FASTAPI API     |
                |      Port 8000       |
                +----------+-----------+
                           |
                           | Celery Task
                           v
                +----------------------+
                |        REDIS          |
                |                       |
                | Message Broker        |
                | Result Backend        |
                | Port 6379             |
                +----------+-----------+
                           |
                           | Consume Task
                           v
                +----------------------+
                |    CELERY WORKER      |
                |                       |
                |  1. Validate          |
                |  2. Chunk Text        |
                |  3. Generate Vectors  |
                |  4. Store Vectors    |
                +----------+-----------+
                           |
                           | HTTP
                           v
                +----------------------+
                |       CHROMADB        |
                |    Vector Database    |
                |                       |
                | IDs                   |
                | Text                  |
                | Embeddings            |
                | Metadata              |
                +----------------------+
```

---

# 🔄 Complete Data Flow

The complete processing flow is:

```text
                    POST /ingest
                          |
                          v
                +-------------------+
                | Pydantic Validation|
                +---------+---------+
                          |
                          v
                +-------------------+
                | Celery Task       |
                | Creation          |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Redis Queue       |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Celery Worker     |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Validate Payload  |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Split Text        |
                | 500 chars         |
                | 50 char overlap   |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Generate          |
                | Embeddings        |
                | 384 dimensions    |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Create Chunk IDs  |
                | and Metadata      |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Batch Upsert      |
                | into ChromaDB     |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Task Completed    |
                +---------+---------+
                          |
                          v
                GET /task/{task_id}
```

---

# 🛠️ Technology Stack

| Technology                | Purpose                          |
| ------------------------- | -------------------------------- |
| **Python 3.9+**           | Main programming language        |
| **FastAPI**               | REST API framework               |
| **Uvicorn**               | ASGI application server          |
| **Pydantic v2**           | Request and data validation      |
| **Pydantic Settings**     | Environment configuration        |
| **Celery**                | Asynchronous task processing     |
| **Redis**                 | Celery broker and result backend |
| **Sentence Transformers** | Embedding generation             |
| **all-MiniLM-L6-v2**      | Embedding model                  |
| **PyTorch**               | Machine-learning runtime         |
| **ChromaDB**              | Vector database                  |
| **Docker**                | Containerization                 |
| **Docker Compose**        | Multi-container orchestration    |
| **pytest**                | Automated testing                |
| **pytest-asyncio**        | Async testing                    |
| **pytest-mock**           | Mocking dependencies             |
| **cURL**                  | Manual API testing               |
| **Postman**               | Optional API testing             |

---

# 📁 Project Structure

```text
Scalable_Document_Ingestion_and_Vectorization_Service/
│
├── .env
├── .env.example
├── .gitignore
├── Dockerfile.api
├── Dockerfile.worker
├── docker-compose.yml
├── pytest.ini
├── README.md
├── Testing.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   └── main.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chroma_service.py
│   │   └── embedding_service.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── text_splitter.py
│   │
│   └── worker/
│       ├── __init__.py
│       ├── models.py
│       └── tasks.py
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_api.py
    ├── test_services.py
    ├── test_utils.py
    └── test_worker.py
```


---


# 🔌 API Documentation

## 1. Health Check

### Endpoint

```http
GET /health
```

### Purpose

Checks whether the FastAPI service is running.

### PowerShell Request

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
# OR using curl.exe
curl.exe -i -X GET http://localhost:8000/health
```

### Expected Response

```http
HTTP/1.1 200 OK
```

```json
{
  "status": "ok"
}
```

---

# 2. Document Ingestion

### Endpoint

```http
POST /ingest
```

### Purpose

Submits a document for asynchronous processing.

### Request Body

```json
{
  "document_id": "doc_test_001",
  "text_content": "Retrieval-Augmented Generation combines search with large language models.",
  "source_url": "https://example.com/document.pdf"
}
```

The `source_url` field is optional.

### PowerShell Request

```powershell
$body = @{
    document_id = "doc_test_001"
    text_content = "Retrieval-Augmented Generation combines search with large language models."
    source_url = "https://example.com/document.pdf"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/ingest" -Method Post -ContentType "application/json" -Body $body
```

### Expected Status

```text
HTTP 202 Accepted
```

### Example Response

```json
{
  "task_id": "generated-task-id",
  "status": "Processing enqueued"
}
```

The task ID is required for checking the processing status.

---

# 3. Task Status

### Endpoint

```http
GET /task/{task_id}
```

### Purpose

Returns the current state of a Celery task.

### PowerShell Request

```powershell
# Store the task_id returned from /ingest (do not include angle brackets < >)
$taskId = "YOUR_TASK_ID"
Invoke-RestMethod -Uri "http://localhost:8000/task/$taskId" -Method Get
# OR using curl.exe
curl.exe -i -X GET "http://localhost:8000/task/$taskId"
```

### Possible States

```text
PENDING
STARTED
SUCCESS
FAILURE
RETRY
```

A successful task returns information about the processed document and the number of chunks created.

---

# ✂️ Text Chunking

Large documents are divided into smaller chunks before embedding.

The primary processing pipeline uses:

```text
Chunk Size = 500 characters
Overlap    = 50 characters
```

Example:

```text
Chunk 1
|---------------------------------------------|
0                                           500


Chunk 2
                                      |---------------------------------------------|
                                      450                                      950
```

The overlapping region helps preserve contextual information between consecutive chunks.

---

# 🧠 Embedding Generation

The project uses the Sentence Transformers library.

Default model:

```text
all-MiniLM-L6-v2
```

The model generates:

```text
384-dimensional vectors
```

Processing:

```text
Text Chunk
    |
    v
Sentence Transformer
    |
    v
384-Dimensional Vector
```

Example:

```text
[0.021, -0.132, 0.457, ...]
```

The complete vector contains 384 numerical values.

These values represent the semantic meaning of the corresponding text chunk.

---

# ⚡ Batch Processing

The project uses batch processing at two major stages.

## Batch Embedding

Instead of:

```text
Chunk 1 -> Model
Chunk 2 -> Model
Chunk 3 -> Model
Chunk 4 -> Model
```

the worker sends:

```text
Chunk 1
Chunk 2
Chunk 3
Chunk 4
    |
    v
Embedding Model
    |
    +----> Vector 1
    +----> Vector 2
    +----> Vector 3
    +----> Vector 4
```

This improves efficiency.

## Batch ChromaDB Upsert

The vectors are also stored in a batch.

```text
All Chunks + Embeddings
          |
          v
     Batch Upsert
          |
          v
       ChromaDB
```

This reduces the number of database operations.

---

# 🗄️ Vector Storage

ChromaDB stores:

```text
Chunk ID
Chunk Text
Embedding Vector
Metadata
```

Example conceptual record:

```json
{
  "id": "doc_001_chunk_0",
  "document": "This is a document chunk.",
  "embedding": "[384 dimensional vector]",
  "metadata": {
    "document_id": "doc_001"
  }
}
```

The ChromaDB collection uses cosine similarity:

```text
hnsw:space = cosine
```

Cosine similarity is useful for comparing semantic similarity between embedding vectors.

---

# 🔄 Asynchronous Processing

The API uses Celery to process documents asynchronously.

The flow is:

```text
Client
   |
   v
FastAPI
   |
   v
Celery Task
   |
   v
Redis
   |
   v
Celery Worker
   |
   v
Document Processing
```

The API does not have to wait for:

* Text chunking
* Embedding generation
* ChromaDB storage

This improves API responsiveness.

---

# 🛡️ Error Handling

The application handles errors at both the API and worker levels.

## API-Level Validation

Invalid requests include:

* Missing `document_id`
* Blank `document_id`
* Missing `text_content`
* Blank `text_content`
* Invalid JSON

These requests are rejected with:

```text
HTTP 400 Bad Request
```

---

## Worker-Level Errors

Worker errors are divided into:

### Validation Errors

These are not retried.

Examples:

```text
Invalid document ID
Invalid text content
Invalid task payload
```

### Transient Errors

These can be retried.

Examples:

```text
ChromaDB unavailable
Temporary connection failure
Embedding processing failure
```

---

# 🔁 Retry Mechanism

The Celery worker supports retry behavior.

Configuration:

```text
Maximum Retries = 3
Retry Delay     = 300 seconds
```

Example:

```text
Worker
   |
   v
ChromaDB Request
   |
   X
Connection Failure
   |
   v
Retry Task
   |
   v
ChromaDB Available
   |
   v
Task Success
```

This improves system resiliency against temporary infrastructure failures.

---

# 🔐 Idempotency

Each document chunk receives a deterministic ID.

Format:

```text
{document_id}_chunk_{index}
```

Example:

```text
document_123_chunk_0
document_123_chunk_1
document_123_chunk_2
```

The worker uses ChromaDB's `upsert` operation.

Therefore, if the same document is processed again, the existing records can be updated instead of creating duplicates.

This is particularly useful when a task is retried.

---

# 🐳 Docker Architecture

The complete system runs using Docker Compose.

The application contains four services:

```text
+-------------------+
|      FastAPI      |
|     Port 8000     |
+---------+---------+
          |
          v
+-------------------+
|       Redis       |
|     Port 6379     |
+---------+---------+
          |
          v
+-------------------+
|   Celery Worker   |
+---------+---------+
          |
          v
+-------------------+
|      ChromaDB     |
| Host Port: 8001   |
| Container: 8000   |
+-------------------+
```

---

# 🧱 Docker Services

## FastAPI

Runs the REST API.

Host:

```text
http://localhost:8000
```

---

## Redis

Acts as:

1. Celery message broker.
2. Celery result backend.

Port:

```text
6379
```

Docker service name:

```text
redis
```

---

## Celery Worker

Consumes tasks from Redis and executes the document processing pipeline.

Processing:

```text
Validate
   |
Chunk
   |
Embed
   |
Store
```

---

## ChromaDB

Acts as the vector database.

Inside Docker:

```text
http://chroma:8000
```

From the host:

```text
http://localhost:8001
```

---

# 💾 Persistent Storage

ChromaDB uses the Docker volume:

```text
chroma_data
```

This allows vector data to remain available even if the ChromaDB container is recreated.

---

# ⚙️ Environment Variables

| Variable                 | Default                   | Purpose                 |
| ------------------------ | ------------------------- | ----------------------- |
| `REDIS_HOST`             | `redis`                   | Redis hostname          |
| `REDIS_PORT`             | `6379`                    | Redis port              |
| `CHROMA_HOST`            | `chroma`                  | ChromaDB hostname       |
| `CHROMA_PORT`            | `8000`                    | ChromaDB container port |
| `CHROMA_COLLECTION_NAME` | `rag_document_embeddings` | Vector collection       |
| `EMBEDDING_MODEL_NAME`   | `all-MiniLM-L6-v2`        | Embedding model         |
| `FASTAPI_PORT`           | `8000`                    | API port                |

---

# 💻 Installation and Setup

## Prerequisites

Install the following:

* Python 3.9 or higher
* Docker Desktop
* Docker Compose
* cURL or Postman
* Git
* VS Code or another code editor

---

# 📦 Local Python Environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🐳 Running with Docker Compose

Build and start all services:

```bash
docker-compose up --build -d
```

Check service status:

```bash
docker-compose ps
```

View all logs:

```bash
docker-compose logs -f
```

View worker logs:

```bash
docker-compose logs -f worker
```

Stop the services:

```bash
docker-compose down
```

---

# 📚 Swagger API Documentation

Once the application is running, open:

```text
http://localhost:8000/docs
```

FastAPI automatically provides an interactive Swagger UI.

From Swagger UI, you can test:

```text
GET  /health
POST /ingest
GET  /task/{task_id}
```

---

# 🧪 Testing

The project includes automated, integration, end-to-end, and manual API testing.

---

## Automated Testing

Run all tests:

```bash
python -m pytest tests/ -v
```

Run tests with live output:

```bash
python -m pytest tests/ -v -s
```

Run individual test modules:

```bash
python -m pytest tests/test_api.py -v
```

```bash
python -m pytest tests/test_services.py -v
```

```bash
python -m pytest tests/test_utils.py -v
```

```bash
python -m pytest tests/test_worker.py -v
```

---

# 📊 Test Suite

The automated test suite contains **65 tests**.

| Test Module        | Purpose                         |  Tests |
| ------------------ | ------------------------------- | -----: |
| `test_api.py`      | API endpoints and validation    |     17 |
| `test_services.py` | Embedding and ChromaDB services |     17 |
| `test_utils.py`    | Text chunking                   |     19 |
| `test_worker.py`   | Celery worker processing        |     12 |
| **Total**          |                                 | **65** |

The expected result is:

```text
65 passed
```

---

# 🔬 Testing Approach

## API Testing

Tests:

* Health endpoint
* Document ingestion
* Task submission
* Request validation
* Error responses

## Service Testing

Tests:

* Embedding generation
* Batch embeddings
* ChromaDB initialization
* Batch upserts

External dependencies are mocked where appropriate.

## Utility Testing

Tests:

* Chunk size
* Chunk overlap
* Boundary conditions
* Invalid parameters

## Worker Testing

Tests:

* Payload validation
* Document processing
* Chunk generation
* Embedding generation
* ChromaDB upserts
* Metadata
* Deterministic IDs
* Retry behavior

---

# 🔍 Manual API Testing

## Health Check

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

Expected:

```json
{
  "status": "ok"
}
```

---

## Document Ingestion

```powershell
$body = @{
    document_id = "doc_test_001"
    text_content = "Retrieval-Augmented Generation combines search with large language models."
    source_url = "https://example.com/document.pdf"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/ingest" -Method Post -ContentType "application/json" -Body $body
```

Expected:

```text
HTTP 202 Accepted
```

The response provides a task ID.

---

## Task Status

Use the returned task ID (do not include angle brackets `< >`):

```powershell
$taskId = "YOUR_TASK_ID"
Invoke-RestMethod -Uri "http://localhost:8000/task/$taskId" -Method Get
```

The task should eventually reach:

```text
SUCCESS
```

---

# ❌ Error Testing

The application also tests invalid requests.

Examples include:

### Missing Document ID

```json
{
  "text_content": "Valid text content"
}
```

Expected:

```text
HTTP 400 Bad Request
```

---

### Blank Text

```json
{
  "document_id": "doc_002",
  "text_content": "   "
}
```

Expected:

```text
HTTP 400 Bad Request
```

---

### Invalid JSON

```text
{ invalid json body }
```

Expected:

```text
HTTP 400 Bad Request
```

---

# 🔥 Celery Retry Testing

To simulate a ChromaDB failure:

First submit a document.

Then stop ChromaDB:

```powershell
docker-compose stop chroma
```

Check worker logs:

```powershell
docker-compose logs -f worker
```

The worker should detect the processing failure and attempt retry behavior.

Restart ChromaDB:

```powershell
docker-compose start chroma
```

The task can then complete successfully when the service becomes available again.

---

# 📈 Performance and Scalability

The architecture is designed to separate API traffic from document-processing workloads.

Important performance techniques include:

1. Asynchronous processing using Celery.
2. Redis-based task queue.
3. Batch embedding generation.
4. Batch ChromaDB upsert.
5. Loading the embedding model once per worker process.
6. Pre-downloading the model during Docker image creation.
7. Controlled task distribution.
8. Persistent ChromaDB storage.
9. Deterministic chunk IDs.
10. Idempotent writes.

---

# 📊 Scaling the Worker Layer

The API and worker are independent components.

Therefore, additional Celery workers can be added when document-processing volume increases.

Conceptually:

```text
                    Redis
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Worker 1    Worker 2    Worker 3
          |           |           |
          +-----------+-----------+
                      |
                      v
                   ChromaDB
```

This allows background processing capacity to scale independently from the API.

---

# 🔒 Security Considerations

For production deployment:

* Never commit `.env` files containing secrets.
* Keep credentials outside source control.
* Use a secret-management system where appropriate.
* Restrict public access to Redis.
* Restrict public access to ChromaDB.
* Add API authentication.
* Add authorization.
* Add rate limiting.
* Add request-size limits.
* Validate incoming content.
* Use HTTPS.
* Monitor worker failures.
* Monitor queue depth.
* Monitor database availability.
* Pin dependency versions.

---

# 📌 Complete End-to-End Architecture

```text
                         CLIENT
                    cURL / Postman
                          |
                          v
                   +-------------+
                   |   FastAPI   |
                   |   REST API  |
                   +------+------+
                          |
                          v
                   +-------------+
                   |    Redis    |
                   | Task Queue  |
                   +------+------+
                          |
                          v
                   +-------------+
                   |   Celery    |
                   |   Worker    |
                   +------+------+
                          |
              +-----------+-----------+
              |           |           |
              v           v           v
           Validate     Chunk       Embed
                                      |
                                      v
                               384-Dimensional
                                  Vectors
                                      |
                                      v
                              +---------------+
                              |    ChromaDB   |
                              | Vector Storage|
                              +---------------+
```

---

# 📝 Complete Processing Summary

The complete processing sequence is:

```text
1. Client sends document
             |
             v
2. FastAPI validates request
             |
             v
3. Celery task is created
             |
             v
4. Redis stores the task
             |
             v
5. Celery worker receives task
             |
             v
6. Worker validates payload
             |
             v
7. Text is divided into chunks
             |
             v
8. Sentence Transformer generates embeddings
             |
             v
9. Deterministic chunk IDs are generated
             |
             v
10. Embeddings and metadata are batch-upserted
             |
             v
11. ChromaDB stores the vectors
             |
             v
12. Celery task completes
             |
             v
13. Client checks status using task ID
```

---

# 🏆 Conclusion

The **Scalable Document Ingestion and Vectorization Service** provides a modular and scalable foundation for document processing in modern RAG and semantic-search applications.

The project combines:

* **FastAPI** for REST APIs
* **Celery** for asynchronous processing
* **Redis** for task management
* **Sentence Transformers** for embeddings
* **ChromaDB** for vector storage
* **Docker Compose** for container orchestration
* **pytest** for automated testing

The architecture separates API handling from background document processing, allowing the system to remain responsive while processing documents.

The use of batch processing, deterministic IDs, idempotent upserts, retry handling, and persistent vector storage makes the system suitable as a foundation for a larger production-oriented RAG architecture.

```text
              SCALABLE DOCUMENT
             INGESTION PIPELINE

                    |
                    v

                 FastAPI
                    |
                    v
                  Redis
                    |
                    v
               Celery Worker
                    |
          +---------+---------+
          |         |         |
          v         v         v
       Validate   Chunk     Embed
                              |
                              v
                          ChromaDB
                              |
                              v
                       Vector Storage
```

**This project forms the ingestion and vectorization foundation required before implementing semantic retrieval and the final RAG generation layer.**
