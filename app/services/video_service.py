"""Video generation service: Replicate image-to-video and FFmpeg concatenation."""

import asyncio
import io
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx
import replicate

from app.schemas.video import VideoScene

logger = logging.getLogger(__name__)

# Seconds to wait before each retry on rate-limit (3 attempts total)
_RATE_LIMIT_DELAYS = (10, 30, 90)


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


class VideoServiceError(Exception):
    pass


def _build_input(model: str, img_bytes: bytes, scene: "VideoScene") -> dict[str, Any]:
    """Build Replicate input dict for the configured video model."""
    if "wan" in model.lower():
        fps = 16
        num_frames = max(81, min(121, round(scene.duration_seconds * fps)))
        return {
            "prompt": scene.prompt,
            "image": io.BytesIO(img_bytes),
            "num_frames": num_frames,
            "frames_per_second": fps,
        }
    # minimax/video-01 and compatible models
    return {
        "prompt": scene.prompt,
        "first_frame_image": io.BytesIO(img_bytes),
    }


class VideoService:
    def __init__(self, replicate_token: str, video_model: str) -> None:
        self._replicate_token = replicate_token
        self._video_model = video_model

    async def generate_clip(
        self,
        image_path: str,
        scene: VideoScene,
        output_dir: str,
        clip_index: int,
    ) -> str:
        rep_client = replicate.Client(api_token=self._replicate_token)  # type: ignore[attr-defined]

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        delays = iter(_RATE_LIMIT_DELAYS)
        while True:
            try:
                raw = await rep_client.async_run(
                    self._video_model,
                    input=_build_input(self._video_model, img_bytes, scene),
                )
                break
            except Exception as exc:
                delay = next(delays, None)
                if delay is None or not _is_rate_limit(exc):
                    raise
                logger.warning(
                    "Clip %d hit rate limit, retrying in %ds…", clip_index, delay
                )
                await asyncio.sleep(delay)

        output_url = str(raw[0]) if isinstance(raw, list) else str(raw)

        async with httpx.AsyncClient(timeout=120.0) as http_client:
            response = await http_client.get(output_url)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                logger.error(
                    "Failed to download clip %d from %s (status %d)",
                    clip_index,
                    output_url,
                    response.status_code,
                )
                raise

            clip_path = str(Path(output_dir) / f"clip_{clip_index:02d}.mp4")
            with open(clip_path, "wb") as f:
                f.write(response.content)

        logger.info(f"Clip {clip_index} generated: {clip_path}")
        return clip_path

    def concatenate_clips(self, clip_paths: list[str], output_path: str) -> str:
        filelist = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        try:
            for path in clip_paths:
                filelist.write(f"file '{path}'\n")
            filelist.flush()
            filelist.close()

            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        filelist.name,
                        "-c",
                        "copy",
                        output_path,
                        "-y",
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode(errors="replace") if e.stderr else ""
                raise VideoServiceError(f"FFmpeg concatenation failed: {stderr}") from e
        finally:
            Path(filelist.name).unlink(missing_ok=True)

        logger.info(f"Clips concatenated: {output_path}")
        return output_path
