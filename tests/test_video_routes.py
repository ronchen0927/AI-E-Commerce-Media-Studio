"""API tests for video endpoints."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.schemas.video import StoryboardResponse, VideoScene


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    img = Image.new("RGB", (100, 100), color="green")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def sample_storyboard() -> StoryboardResponse:
    return StoryboardResponse(
        image_path="storage/abc/original.png",
        product_summary="Green product",
        style="cinematic",
        source="ai",
        scenes=[
            VideoScene(
                id=1,
                shot_type="Close-up",
                camera_motion="Static shot",
                visual_description="Green surface",
                duration_seconds=5,
                prompt="Close-up, static shot...",
            )
        ],
    )


class TestStoryboardEndpoint:
    def test_storyboard_returns_200(
        self,
        client: TestClient,
        sample_image_bytes: bytes,
        sample_storyboard: StoryboardResponse,
    ) -> None:
        with (
            patch("app.api.video_routes.StoryboardService") as mock_svc_cls,
            patch("app.api.video_routes.get_storage_service") as mock_storage_dep,
        ):
            mock_storage = MagicMock()
            mock_storage.upload = AsyncMock(return_value="storage/abc/original.png")
            mock_storage_dep.return_value = mock_storage

            mock_svc = MagicMock()
            mock_svc.generate = AsyncMock(return_value=sample_storyboard)
            mock_svc_cls.return_value = mock_svc

            response = client.post(
                "/api/v1/video/storyboard",
                files={"image": ("product.png", sample_image_bytes, "image/png")},
                data={"style": "cinematic", "num_scenes": "3"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "scenes" in data
        assert "image_path" in data
        assert data["source"] in ("ai", "template")

    def test_storyboard_rejects_non_image(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/video/storyboard",
            files={"image": ("file.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400


class TestVideoGenerateEndpoint:
    def test_generate_returns_task_id(
        self,
        client: TestClient,
        sample_storyboard: StoryboardResponse,
    ) -> None:
        with patch("app.api.video_routes.celery_app") as mock_celery:
            mock_celery.send_task = MagicMock()

            response = client.post(
                "/api/v1/video/generate",
                json={
                    "image_path": "storage/abc/original.png",
                    "scenes": [s.model_dump() for s in sample_storyboard.scenes],
                },
            )

        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data

    def test_generate_rejects_empty_scenes(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/video/generate",
            json={"image_path": "storage/abc/original.png", "scenes": []},
        )
        assert response.status_code == 422


class TestVideoStatusEndpoint:
    def test_status_not_found(self, client: TestClient) -> None:
        response = client.get("/api/v1/video/status/nonexistent-id")
        assert response.status_code == 404

    def test_status_after_generate(
        self,
        client: TestClient,
        sample_storyboard: StoryboardResponse,
    ) -> None:
        with patch("app.api.video_routes.celery_app") as mock_celery:
            mock_celery.send_task = MagicMock()
            gen_response = client.post(
                "/api/v1/video/generate",
                json={
                    "image_path": "storage/abc/original.png",
                    "scenes": [s.model_dump() for s in sample_storyboard.scenes],
                },
            )
        task_id = gen_response.json()["task_id"]

        with patch("app.api.video_routes.AsyncResult") as mock_result_cls:
            mock_result = MagicMock()
            mock_result.state = "GENERATING_VIDEO"
            mock_result.ready.return_value = False
            mock_result.info = {"clips_done": 0, "clips_total": 1, "clips_failed": []}
            mock_result_cls.return_value = mock_result

            status_response = client.get(f"/api/v1/video/status/{task_id}")

        assert status_response.status_code == 200
        data = status_response.json()
        assert data["task_id"] == task_id
        assert "clips_total" in data
