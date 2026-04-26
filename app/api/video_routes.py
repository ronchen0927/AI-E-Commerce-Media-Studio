"""API routes for video generation."""

import os
import tempfile
import uuid
from typing import Any

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.routes import get_storage_service
from app.core.celery_app import celery_app
from app.core.config import Settings, get_settings
from app.schemas.video import (
    StoryboardResponse,
    VideoGenerateRequest,
    VideoGenerateResponse,
    VideoStatusResponse,
)
from app.services.storage import StorageService
from app.services.storyboard_service import StoryboardService

router = APIRouter(prefix="/api/v1/video", tags=["video"])

video_task_store: dict[str, dict[str, Any]] = {}


@router.post("/storyboard", response_model=StoryboardResponse)
async def generate_storyboard(
    image: UploadFile = File(...),
    style: str = Form("cinematic"),
    num_scenes: int = Form(3),
    storage: StorageService = Depends(get_storage_service),
    settings: Settings = Depends(get_settings),
) -> StoryboardResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    num_scenes = max(1, min(num_scenes, 5))

    file_ext = image.filename.split(".")[-1] if image.filename else "jpg"
    upload_id = str(uuid.uuid4())
    destination_path = f"uploads/{upload_id}/original.{file_ext}"

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{file_ext}", mode="wb"
    ) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        stored_path = await storage.upload(tmp_path, destination_path)
        service = StoryboardService(api_key=settings.openai_api_key)
        storyboard = await service.generate(
            image_path=tmp_path,
            image_storage_path=stored_path,
            style=style,
            num_scenes=num_scenes,
        )
    finally:
        os.unlink(tmp_path)

    return storyboard


@router.post("/generate", response_model=VideoGenerateResponse, status_code=202)
async def generate_video(
    request: VideoGenerateRequest,
) -> VideoGenerateResponse:
    task_id = str(uuid.uuid4())
    video_task_store[task_id] = {
        "task_id": task_id,
        "clips_total": len(request.scenes),
    }

    celery_app.send_task(
        "app.tasks.video_processing.process_video",
        args=[task_id, request.image_path, [s.model_dump() for s in request.scenes]],
        task_id=task_id,
    )

    return VideoGenerateResponse(
        task_id=task_id,
        message=f"Video generation started. {len(request.scenes)} clips queued.",
    )


@router.get("/status/{task_id}", response_model=VideoStatusResponse)
async def get_video_status(task_id: str) -> VideoStatusResponse:
    metadata = video_task_store.get(task_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Task not found")

    celery_result = AsyncResult(task_id, app=celery_app)
    state = celery_result.state

    if state == "SUCCESS" and celery_result.successful():
        info = celery_result.result or {}
        task_status = info.get("status", "COMPLETED")
        if task_status == "FAILED":
            return VideoStatusResponse(
                task_id=task_id,
                status="FAILED",
                clips_total=metadata["clips_total"],
                error=info.get("error"),
            )
        return VideoStatusResponse(
            task_id=task_id,
            status=task_status,
            clips_done=metadata["clips_total"] - len(info.get("clips_failed", [])),
            clips_total=metadata["clips_total"],
            clips_failed=info.get("clips_failed", []),
            result_url=info.get("result_url"),
        )

    if state == "FAILURE":
        return VideoStatusResponse(
            task_id=task_id,
            status="FAILED",
            clips_total=metadata["clips_total"],
            error=str(celery_result.result),
        )

    meta = celery_result.info or {}
    return VideoStatusResponse(
        task_id=task_id,
        status=state,
        clips_done=meta.get("clips_done", 0) if isinstance(meta, dict) else 0,
        clips_total=metadata["clips_total"],
        clips_failed=meta.get("clips_failed", []) if isinstance(meta, dict) else [],
    )
