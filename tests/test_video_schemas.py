"""Tests for video Pydantic schemas."""
from app.schemas.video import (
    StoryboardResponse,
    VideoGenerateRequest,
    VideoGenerateResponse,
    VideoScene,
    VideoStatusResponse,
)


def test_video_scene_defaults() -> None:
    scene = VideoScene(
        id=1,
        shot_type="Extreme macro close-up",
        camera_motion="Slow dolly forward",
        visual_description="Sneaker mesh texture detail",
        duration_seconds=5,
        prompt="Extreme macro close-up, slow dolly forward...",
    )
    assert scene.id == 1
    assert scene.duration_seconds == 5


def test_storyboard_response_source_default() -> None:
    response = StoryboardResponse(
        image_path="storage/abc/original.png",
        product_summary="White sneaker",
        style="cinematic",
        scenes=[],
    )
    assert response.source == "ai"


def test_video_generate_request() -> None:
    req = VideoGenerateRequest(
        image_path="storage/abc/original.png",
        scenes=[
            VideoScene(
                id=1,
                shot_type="Close-up",
                camera_motion="Static shot",
                visual_description="Product on white surface",
                duration_seconds=5,
                prompt="Close-up, static shot...",
            )
        ],
    )
    assert len(req.scenes) == 1


def test_video_status_response_partial() -> None:
    status = VideoStatusResponse(
        task_id="abc",
        status="completed_partial",
        clips_done=2,
        clips_total=3,
        clips_failed=[2],
        result_url="storage/abc/final.mp4",
    )
    assert status.clips_failed == [2]


def test_video_generate_response() -> None:
    response = VideoGenerateResponse(
        task_id="video-task-123",
        message="Video generation started",
    )
    assert response.task_id == "video-task-123"
    assert response.message == "Video generation started"
