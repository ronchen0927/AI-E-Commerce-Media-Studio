"""Pydantic schemas for video generation endpoints."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class VideoScene(BaseModel):
    """One scene in a video storyboard."""

    id: int
    shot_type: str
    camera_motion: str
    visual_description: str
    duration_seconds: int = Field(default=5, ge=3, le=10)
    prompt: str


class StoryboardResponse(BaseModel):
    """Response from POST /api/v1/video/storyboard."""

    image_path: str
    product_summary: str
    style: str
    source: Literal["ai", "template"] = "ai"
    scenes: list[VideoScene]


class VideoGenerateRequest(BaseModel):
    """Request body for POST /api/v1/video/generate."""

    image_path: str
    scenes: list[VideoScene]

    @field_validator("scenes")
    @classmethod
    def scenes_not_empty(cls, v: list[VideoScene]) -> list[VideoScene]:
        if not v:
            raise ValueError("scenes must not be empty")
        return v


class VideoGenerateResponse(BaseModel):
    """Response from POST /api/v1/video/generate."""

    task_id: str
    message: str


class VideoStatusResponse(BaseModel):
    """Response from GET /api/v1/video/status/{task_id}."""

    task_id: str
    status: str
    clips_done: int = 0
    clips_total: int = 0
    clips_failed: list[int] = Field(default_factory=list)
    result_url: Optional[str] = None
    error: Optional[str] = None
