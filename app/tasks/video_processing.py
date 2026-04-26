"""Celery tasks for video generation pipeline."""

import asyncio
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

from celery import Task

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.schemas.video import VideoScene
from app.services.storage import GCSStorage, LocalStorage, StorageService
from app.services.video_service import VideoService


class VideoProcessingTask(Task):  # type: ignore[misc]
    def update_progress(
        self,
        clips_done: int,
        clips_total: int,
        clips_failed: list[int],
    ) -> None:
        self.update_state(
            state="GENERATING_VIDEO",
            meta={
                "clips_done": clips_done,
                "clips_total": clips_total,
                "clips_failed": clips_failed,
            },
        )


def _get_storage(image_path: str = "") -> StorageService:
    settings = get_settings()
    if settings.storage_type == "gcs" or image_path.startswith("gs://"):
        return GCSStorage(bucket_name=settings.gcs_bucket_name)
    return LocalStorage(base_dir=Path(settings.local_storage_path))


async def _process_video_async(
    task: VideoProcessingTask,
    task_id: str,
    image_path: str,
    scenes_data: list[dict[str, Any]],
) -> dict[str, Any]:
    settings = get_settings()
    storage = _get_storage(image_path)

    temp_dir = Path(tempfile.mkdtemp(prefix=f"video_{task_id}_"))
    local_image = temp_dir / Path(image_path).name

    try:
        if image_path.startswith("gs://") or image_path.startswith("http"):
            gcs_key = image_path.replace(f"gs://{settings.gcs_bucket_name}/", "")
            await storage.download(gcs_key, str(local_image))
        else:
            shutil.copy2(image_path, local_image)

        scenes = [VideoScene(**s) for s in scenes_data]
        clips_dir = temp_dir / "clips"
        clips_dir.mkdir()

        video_service = VideoService(
            replicate_token=settings.replicate_api_token,
            video_model=settings.replicate_video_model,
        )

        task.update_progress(0, len(scenes), [])

        successful_clips: list[str] = []
        failed_indices: list[int] = []
        for i, scene in enumerate(scenes):
            try:
                clip_path = await video_service.generate_clip(
                    image_path=str(local_image),
                    scene=scene,
                    output_dir=str(clips_dir),
                    clip_index=i,
                )
                successful_clips.append(clip_path)
            except Exception:
                failed_indices.append(i)
            task.update_progress(len(successful_clips), len(scenes), failed_indices)

        if not successful_clips:
            raise RuntimeError("All clips failed to generate")

        final_local = str(temp_dir / "final.mp4")
        video_service.concatenate_clips(successful_clips, final_local)

        result_key = f"videos/{task_id}/final.mp4"
        result_url = await storage.upload(final_local, result_key)

        status = "COMPLETED_PARTIAL" if failed_indices else "COMPLETED"
        return {
            "task_id": task_id,
            "status": status,
            "result_url": result_url,
            "clips_failed": failed_indices,
        }

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


@celery_app.task(
    bind=True,
    base=VideoProcessingTask,
    name="app.tasks.video_processing.process_video",
    time_limit=1800,
    soft_time_limit=1700,
)
def process_video(
    self: VideoProcessingTask,
    task_id: str,
    image_path: str,
    scenes_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate video clips in parallel then concatenate into a final MP4.

    Note: exceptions are caught and returned as {"status": "FAILED"} dicts rather than
    re-raised, so Celery records task state as SUCCESS. Task status endpoint must
    check result["status"] in addition to AsyncResult.state to detect failures.
    """
    try:
        return asyncio.run(_process_video_async(self, task_id, image_path, scenes_data))
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        self.update_state(state="FAILURE", meta={"error": error_msg})
        return {
            "task_id": task_id,
            "status": "FAILED",
            "error": error_msg,
        }
