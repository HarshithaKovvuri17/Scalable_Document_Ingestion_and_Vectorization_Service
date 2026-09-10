"""
Unit tests for the FastAPI API layer (src/api/main.py).

Tests match the task specification exactly:
  - POST /ingest returns 202 with task_id and status
  - Invalid requests return 400 Bad Request (not 422)
  - GET /health returns {"status": "ok"}
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixture: FastAPI app with Celery task mocked
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_and_task():
    """Import the FastAPI app once with Celery tasks mocked."""
    import src.worker.tasks
    with patch("src.worker.tasks.process_document_task") as mock_task:
        mock_result = MagicMock()
        mock_result.id = "test-task-id"
        mock_task.delay.return_value = mock_result

        from src.api.main import app as _app
        yield _app, mock_task


@pytest.fixture(scope="module")
def client(app_and_task):
    _app, _ = app_and_task
    return TestClient(_app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# POST /ingest – happy path
# ---------------------------------------------------------------------------

class TestIngestEndpoint:

    def test_returns_202_for_valid_request(self, client, app_and_task):
        _, mock_task = app_and_task
        mock_task.delay.return_value = MagicMock(id="t-202")
        response = client.post(
            "/ingest",
            json={
                "document_id": "doc-001",
                "text_content": "Some meaningful document text content here.",
                "source_url": "https://example.com",
            },
        )
        assert response.status_code == 202

    def test_response_contains_task_id(self, client, app_and_task):
        _, mock_task = app_and_task
        mock_task.delay.return_value = MagicMock(id="unique-task-abc")
        response = client.post(
            "/ingest",
            json={"document_id": "doc-002", "text_content": "Another document."},
        )
        assert response.json()["task_id"] == "unique-task-abc"

    def test_response_status_is_processing_enqueued(self, client, app_and_task):
        _, mock_task = app_and_task
        mock_task.delay.return_value = MagicMock(id="t-status")
        response = client.post(
            "/ingest",
            json={"document_id": "doc-003", "text_content": "Content here."},
        )
        assert response.json()["status"] == "Processing enqueued"

    def test_delay_is_called_once(self, client, app_and_task):
        _, mock_task = app_and_task
        mock_task.delay.reset_mock()
        mock_task.delay.return_value = MagicMock(id="t-delay")
        client.post(
            "/ingest",
            json={"document_id": "doc-004", "text_content": "Trigger task."},
        )
        mock_task.delay.assert_called_once()

    def test_source_url_is_optional(self, client, app_and_task):
        _, mock_task = app_and_task
        mock_task.delay.return_value = MagicMock(id="t-opt")
        response = client.post(
            "/ingest",
            json={"document_id": "doc-005", "text_content": "No source URL."},
        )
        assert response.status_code == 202

    # ---------------------------------------------------------------------------
    # POST /ingest – validation → 400 Bad Request (per task spec)
    # ---------------------------------------------------------------------------

    def test_missing_document_id_returns_400(self, client):
        response = client.post("/ingest", json={"text_content": "Some text."})
        assert response.status_code == 400

    def test_missing_text_content_returns_400(self, client):
        response = client.post("/ingest", json={"document_id": "doc-006"})
        assert response.status_code == 400

    def test_blank_document_id_returns_400(self, client):
        response = client.post(
            "/ingest", json={"document_id": "   ", "text_content": "Valid content."}
        )
        assert response.status_code == 400

    def test_blank_text_content_returns_400(self, client):
        response = client.post(
            "/ingest", json={"document_id": "doc-007", "text_content": "   "}
        )
        assert response.status_code == 400

    def test_empty_payload_returns_400(self, client):
        assert client.post("/ingest", json={}).status_code == 400

    def test_non_json_body_returns_400(self, client):
        response = client.post(
            "/ingest",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /health – per task spec: returns {"status": "ok"}
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_body_is_status_ok(self, client):
        body = client.get("/health").json()
        assert body == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /task/{task_id}
# ---------------------------------------------------------------------------

class TestTaskStatusEndpoint:

    def test_returns_200_for_any_task_id(self, client):
        with patch("src.api.main.AsyncResult") as mock_ar:
            mock_ar.return_value = MagicMock(state="PENDING", result=None)
            assert client.get("/task/some-task-id").status_code == 200

    def test_returns_success_result(self, client):
        with patch("src.api.main.AsyncResult") as mock_ar:
            mock_ar.return_value = MagicMock(
                state="SUCCESS",
                result={"status": "completed", "document_id": "doc1", "num_chunks": 3},
            )
            body = client.get("/task/success-task").json()
            assert body["state"] == "SUCCESS"
            assert body["result"]["num_chunks"] == 3

    def test_returns_failure_error(self, client):
        with patch("src.api.main.AsyncResult") as mock_ar:
            mock_ar.return_value = MagicMock(state="FAILURE", result=RuntimeError("fail"))
            body = client.get("/task/fail-task").json()
            assert body["state"] == "FAILURE"
            assert body["error"] is not None
