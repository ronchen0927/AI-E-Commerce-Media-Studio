"""Unit tests for video Celery task."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def scenes() -> list[dict[str, Any]]:
    return [
        {
            "id": 1,
            "shot_type": "Close-up",
            "camera_motion": "Static shot",
            "visual_description": "Product on white",
            "duration_seconds": 5,
            "prompt": "Close-up, static shot...",
        },
        {
            "id": 2,
            "shot_type": "Wide shot",
            "camera_motion": "Slow pan",
            "visual_description": "Product in scene",
            "duration_seconds": 5,
            "prompt": "Wide shot, slow pan...",
        },
    ]


class TestProcessVideoAsync:
    @pytest.mark.asyncio
    async def test_all_clips_succeed(
        self, scenes: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        image_path = str(tmp_path / "product.png")
        Path(image_path).write_bytes(b"fake_image")

        fake_clip = str(tmp_path / "clip_00.mp4")
        Path(fake_clip).write_bytes(b"fake_mp4")
        fake_final = str(tmp_path / "final.mp4")
        Path(fake_final).write_bytes(b"fake_final")

        mock_task = MagicMock()
        mock_video_service = MagicMock()
        mock_video_service.generate_clip = AsyncMock(return_value=fake_clip)
        mock_video_service.concatenate_clips = MagicMock(return_value=fake_final)
        mock_storage = MagicMock()
        mock_storage.upload = AsyncMock(return_value="storage/task123/final.mp4")

        svc_patch = patch(
            "app.tasks.video_processing.VideoService",
            return_value=mock_video_service,
        )
        storage_patch = patch(
            "app.tasks.video_processing._get_storage",
            return_value=mock_storage,
        )
        with (
            svc_patch,
            storage_patch,
            patch("app.tasks.video_processing.get_settings"),
        ):
            from app.tasks.video_processing import _process_video_async

            result = await _process_video_async(
                task=mock_task,
                task_id="task123",
                image_path=image_path,
                scenes_data=scenes,
            )

        assert result["status"] == "COMPLETED"
        assert result["result_url"] == "storage/task123/final.mp4"
        assert result["clips_failed"] == []

    @pytest.mark.asyncio
    async def test_partial_clip_failure(
        self, scenes: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        image_path = str(tmp_path / "product.png")
        Path(image_path).write_bytes(b"fake_image")

        fake_clip = str(tmp_path / "clip_00.mp4")
        Path(fake_clip).write_bytes(b"fake_mp4")
        fake_final = str(tmp_path / "final.mp4")
        Path(fake_final).write_bytes(b"fake_final")

        mock_task = MagicMock()
        mock_video_service = MagicMock()
        mock_video_service.generate_clip = AsyncMock(
            side_effect=[fake_clip, Exception("Replicate timeout")]
        )
        mock_video_service.concatenate_clips = MagicMock(return_value=fake_final)
        mock_storage = MagicMock()
        mock_storage.upload = AsyncMock(return_value="storage/task123/final.mp4")

        svc_patch = patch(
            "app.tasks.video_processing.VideoService",
            return_value=mock_video_service,
        )
        storage_patch = patch(
            "app.tasks.video_processing._get_storage",
            return_value=mock_storage,
        )
        with (
            svc_patch,
            storage_patch,
            patch("app.tasks.video_processing.get_settings"),
        ):
            from app.tasks.video_processing import _process_video_async

            result = await _process_video_async(
                task=mock_task,
                task_id="task123",
                image_path=image_path,
                scenes_data=scenes,
            )

        assert result["status"] == "COMPLETED_PARTIAL"
        assert 1 in result["clips_failed"]
        assert result["result_url"] == "storage/task123/final.mp4"
