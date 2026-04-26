"""Unit tests for VideoService."""

import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.video import VideoScene
from app.services.video_service import VideoService, VideoServiceError


@pytest.fixture
def sample_scene() -> VideoScene:
    return VideoScene(
        id=1,
        shot_type="Close-up",
        camera_motion="Static shot",
        visual_description="Product on white surface",
        duration_seconds=5,
        prompt="Close-up, static shot. Product centered on white surface...",
    )


@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    path = tmp_path / "product.png"
    img.save(str(path))
    return str(path)


class TestGenerateClip:
    @pytest.mark.asyncio
    async def test_generate_clip_returns_path(
        self,
        sample_image_path: str,
        sample_scene: VideoScene,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "clips"
        output_dir.mkdir()

        mock_video_bytes = b"fake_mp4_bytes"

        with (
            patch("app.services.video_service.replicate") as mock_replicate,
            patch("app.services.video_service.httpx") as mock_httpx,
        ):
            mock_replicate.Client.return_value.async_run = AsyncMock(
                return_value="https://replicate.delivery/fake/clip.mp4"
            )
            mock_response = MagicMock()
            mock_response.content = mock_video_bytes
            mock_response.raise_for_status = MagicMock()
            mock_httpx.AsyncClient.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            service = VideoService(
                replicate_token="test-token",
                video_model="wan-video/wan-2.2-i2v-fast",
            )
            result_path = await service.generate_clip(
                image_path=sample_image_path,
                scene=sample_scene,
                output_dir=str(output_dir),
                clip_index=0,
            )

        assert result_path.endswith(".mp4")
        assert os.path.exists(result_path)
        assert open(result_path, "rb").read() == mock_video_bytes

    @pytest.mark.asyncio
    async def test_generate_clip_raises_on_replicate_error(
        self,
        sample_image_path: str,
        sample_scene: VideoScene,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "clips"
        output_dir.mkdir()

        with patch("app.services.video_service.replicate") as mock_replicate:
            mock_replicate.Client.return_value.async_run = AsyncMock(
                side_effect=Exception("Replicate error")
            )
            service = VideoService(
                replicate_token="test-token",
                video_model="wan-video/wan-2.2-i2v-fast",
            )
            with pytest.raises(Exception, match="Replicate error"):
                await service.generate_clip(
                    image_path=sample_image_path,
                    scene=sample_scene,
                    output_dir=str(output_dir),
                    clip_index=0,
                )


class TestConcatenateClips:
    def test_concatenate_multiple_clips_uses_xfade(self, tmp_path: Path) -> None:
        clip_paths = [str(tmp_path / f"clip_0{i}.mp4") for i in range(3)]
        for p in clip_paths:
            Path(p).write_bytes(b"fake")

        output_path = str(tmp_path / "final.mp4")

        mock_ffprobe = MagicMock(returncode=0)
        mock_ffprobe.stdout = "5.0\n"
        mock_ffmpeg = MagicMock(returncode=0)

        with patch("app.services.video_service.subprocess.run") as mock_run:
            mock_run.side_effect = [
                mock_ffprobe,
                mock_ffprobe,
                mock_ffprobe,
                mock_ffmpeg,
            ]
            service = VideoService(
                replicate_token="test-token", video_model="wan-video/wan-2.2-i2v-fast"
            )
            result = service.concatenate_clips(clip_paths, output_path)

        assert result == output_path
        # Last call should be the ffmpeg xfade call
        last_call_args = mock_run.call_args_list[-1][0][0]
        assert "ffmpeg" in last_call_args
        assert "-filter_complex" in last_call_args
        assert any("xfade" in str(a) for a in last_call_args)

    def test_concatenate_single_clip_uses_simple_concat(self, tmp_path: Path) -> None:
        clip_paths = [str(tmp_path / "clip_00.mp4")]
        Path(clip_paths[0]).write_bytes(b"fake")
        output_path = str(tmp_path / "final.mp4")

        with patch("app.services.video_service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            service = VideoService(
                replicate_token="test-token", video_model="wan-video/wan-2.2-i2v-fast"
            )
            result = service.concatenate_clips(clip_paths, output_path)

        assert result == output_path
        call_args = mock_run.call_args[0][0]
        assert "ffmpeg" in call_args
        assert "concat" in call_args

    def test_concatenate_raises_on_ffmpeg_error(self, tmp_path: Path) -> None:
        clip_paths = [str(tmp_path / "clip_00.mp4")]
        Path(clip_paths[0]).write_bytes(b"fake")
        output_path = str(tmp_path / "final.mp4")

        with patch("app.services.video_service.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "ffmpeg", stderr=b"error"
            )
            service = VideoService(
                replicate_token="test-token",
                video_model="wan-video/wan-2.2-i2v-fast",
            )
            with pytest.raises(VideoServiceError, match="FFmpeg"):
                service.concatenate_clips(clip_paths, output_path)
