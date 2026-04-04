"""Integration tests for API endpoints."""

from io import BytesIO
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create sample image bytes for upload testing."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def mock_celery() -> Generator[MagicMock, None, None]:
    """Mock Celery task dispatch to avoid Redis connection."""
    with patch("app.api.routes.celery_app") as mock:
        mock.send_task = MagicMock()
        yield mock


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestUploadEndpoint:
    """Tests for upload endpoint."""

    def test_upload_image(
        self,
        client: TestClient,
        sample_image_bytes: bytes,
        mock_celery: MagicMock,
    ) -> None:
        """Test successful image upload."""
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "PENDING"
        assert "message" in data

        # Verify Celery task was dispatched
        mock_celery.send_task.assert_called_once()

    def test_upload_with_instruction(
        self,
        client: TestClient,
        sample_image_bytes: bytes,
        mock_celery: MagicMock,
    ) -> None:
        """Test upload with edit instruction."""
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
            data={
                "mode": "edit",
                "instruction": "Place on marble table",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PENDING"

    def test_upload_remove_bg_mode(
        self,
        client: TestClient,
        sample_image_bytes: bytes,
        mock_celery: MagicMock,
    ) -> None:
        """Test upload with remove_bg mode."""
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
            data={"mode": "remove_bg"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PENDING"

    def test_upload_invalid_mode(
        self,
        client: TestClient,
        sample_image_bytes: bytes,
        mock_celery: MagicMock,
    ) -> None:
        """Test upload with invalid mode returns 400."""
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
            data={"mode": "invalid_mode"},
        )

        assert response.status_code == 400

    def test_upload_non_image(self, client: TestClient, mock_celery: MagicMock) -> None:
        """Test upload rejects non-image files."""
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )

        assert response.status_code == 400
        assert "image" in response.json()["detail"].lower()


class TestTaskStatusEndpoint:
    """Tests for task status endpoint."""

    def test_task_status_not_found(self, client: TestClient) -> None:
        """Test status for non-existent task."""
        response = client.get("/api/v1/task-status/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_task_status_after_upload(
        self,
        client: TestClient,
        sample_image_bytes: bytes,
        mock_celery: MagicMock,
    ) -> None:
        """Test status after uploading an image."""
        # Upload first
        upload_response = client.post(
            "/api/v1/upload",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
        task_id = upload_response.json()["task_id"]

        # Check status
        status_response = client.get(f"/api/v1/task-status/{task_id}")

        assert status_response.status_code == 200
        data = status_response.json()
        assert data["task_id"] == task_id
        assert "status" in data
        assert "progress" in data


class TestResultEndpoint:
    """Tests for result endpoint."""

    def test_result_not_found(self, client: TestClient) -> None:
        """Test result for non-existent task."""
        response = client.get("/api/v1/result/nonexistent-id")

        assert response.status_code == 404

    def test_result_after_upload(
        self,
        client: TestClient,
        sample_image_bytes: bytes,
        mock_celery: MagicMock,
    ) -> None:
        """Test result after uploading an image."""
        # Upload first
        upload_response = client.post(
            "/api/v1/upload",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
        task_id = upload_response.json()["task_id"]

        # Get result
        result_response = client.get(f"/api/v1/result/{task_id}")

        assert result_response.status_code == 200
        data = result_response.json()
        assert data["task_id"] == task_id
        assert "created_at" in data
        assert "updated_at" in data
