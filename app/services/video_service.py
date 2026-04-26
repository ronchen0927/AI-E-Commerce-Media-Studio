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

# Crossfade duration between clips in seconds
_XFADE_DURATION = 0.5


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


def _get_clip_duration(clip_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            clip_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


class VideoServiceError(Exception):
    pass


def _build_input(
    model: str,
    img_bytes: bytes,
    scene: "VideoScene",
    last_frame_path: str | None = None,
) -> dict[str, Any]:
    """Build Replicate input dict for the configured video model."""
    if "wan" in model.lower():
        fps = 16
        num_frames = max(81, min(121, round(scene.duration_seconds * fps)))
        inp: dict[str, Any] = {
            "prompt": scene.prompt,
            "image": io.BytesIO(img_bytes),
            "num_frames": num_frames,
            "frames_per_second": fps,
        }
        if last_frame_path:
            with open(last_frame_path, "rb") as f:
                inp["last_image"] = io.BytesIO(f.read())
        return inp
    # wan-video/wan-2.2-i2v-fast and compatible models
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
        last_frame_path: str | None = None,
    ) -> str:
        rep_client = replicate.Client(api_token=self._replicate_token)  # type: ignore[attr-defined]

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        delays = iter(_RATE_LIMIT_DELAYS)
        while True:
            try:
                raw = await rep_client.async_run(
                    self._video_model,
                    input=_build_input(
                        self._video_model, img_bytes, scene, last_frame_path
                    ),
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

    def extract_last_frame(self, clip_path: str, output_path: str) -> None:
        """Extract the last frame of a clip as a PNG for last_image conditioning."""
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-sseof",
                    "-1",
                    "-i",
                    clip_path,
                    "-vframes",
                    "1",
                    "-update",
                    "1",
                    output_path,
                    "-y",
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            raise VideoServiceError(f"Failed to extract last frame: {stderr}") from e

    def concatenate_clips(self, clip_paths: list[str], output_path: str) -> str:
        if len(clip_paths) == 1:
            # Single clip — use simple concat demuxer (no xfade needed)
            self._concat_simple([clip_paths[0]], output_path)
            logger.info(f"Clips concatenated: {output_path}")
            return output_path

        try:
            self._concat_xfade(clip_paths, output_path)
        except (VideoServiceError, subprocess.CalledProcessError, ValueError):
            # xfade failed (e.g. codec mismatch) — fall back to hard cut
            logger.warning("xfade concat failed, falling back to hard cut")
            self._concat_simple(clip_paths, output_path)

        logger.info(f"Clips concatenated: {output_path}")
        return output_path

    def _concat_xfade(self, clip_paths: list[str], output_path: str) -> None:
        """Concatenate clips with FFmpeg xfade crossfade transitions."""
        durations = [_get_clip_duration(p) for p in clip_paths]

        inputs: list[str] = []
        for p in clip_paths:
            inputs += ["-i", p]

        # Build xfade filtergraph chain
        # Each xfade offset = cumulative duration of preceding clips
        # minus one fade_duration per already-applied fade
        filter_parts: list[str] = []
        prev_label = "[0:v]"
        offset = 0.0

        for i in range(1, len(clip_paths)):
            offset += durations[i - 1] - _XFADE_DURATION
            is_last = i == len(clip_paths) - 1
            out_label = "[vout]" if is_last else f"[v{i:02d}]"
            filter_parts.append(
                f"{prev_label}[{i}:v]xfade=transition=fade"
                f":duration={_XFADE_DURATION:.3f}:offset={offset:.3f}{out_label}"
            )
            prev_label = out_label

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    *inputs,
                    "-filter_complex",
                    ";".join(filter_parts),
                    "-map",
                    "[vout]",
                    output_path,
                    "-y",
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            raise VideoServiceError(f"FFmpeg xfade failed: {stderr}") from e

    def _concat_simple(self, clip_paths: list[str], output_path: str) -> None:
        """Concatenate clips with FFmpeg concat demuxer (hard cut, no re-encode)."""
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
                raise VideoServiceError(f"FFmpeg concat failed: {stderr}") from e
        finally:
            Path(filelist.name).unlink(missing_ok=True)
