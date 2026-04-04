"""Unit tests for task schemas."""

from datetime import datetime, timezone

from app.schemas.task import (
    TaskCreate,
    TaskMode,
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
    UploadResponse,
)


class TestTaskMode:
    """Tests for TaskMode enum."""

    def test_mode_values(self) -> None:
        """Test all mode values exist."""
        assert TaskMode.REMOVE_BG == "remove_bg"
        assert TaskMode.EDIT == "edit"

    def test_mode_count(self) -> None:
        """Test correct number of modes."""
        assert len(TaskMode) == 2


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_status_values(self) -> None:
        """Test all status values exist."""
        assert TaskStatus.PENDING == "PENDING"
        assert TaskStatus.REMOVING_BG == "REMOVING_BG"
        assert TaskStatus.EDITING == "EDITING"
        assert TaskStatus.COMPLETED == "COMPLETED"
        assert TaskStatus.FAILED == "FAILED"

    def test_status_count(self) -> None:
        """Test correct number of statuses."""
        assert len(TaskStatus) == 5


class TestTaskCreate:
    """Tests for TaskCreate schema."""

    def test_create_with_instruction(self) -> None:
        """Test creating task with instruction."""
        task = TaskCreate(
            mode=TaskMode.EDIT,
            instruction="Place on marble table",
        )
        assert task.instruction == "Place on marble table"
        assert task.mode == TaskMode.EDIT

    def test_create_remove_bg_mode(self) -> None:
        """Test creating task with remove_bg mode."""
        task = TaskCreate(mode=TaskMode.REMOVE_BG)
        assert task.mode == TaskMode.REMOVE_BG
        assert task.instruction is None

    def test_create_defaults(self) -> None:
        """Test creating task with defaults."""
        task = TaskCreate()
        assert task.mode == TaskMode.EDIT
        assert task.instruction is None


class TestTaskResponse:
    """Tests for TaskResponse schema."""

    def test_task_response_fields(self) -> None:
        """Test task response contains all required fields."""
        now = datetime.now(timezone.utc)
        response = TaskResponse(
            task_id="test-123",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        assert response.task_id == "test-123"
        assert response.status == TaskStatus.PENDING
        assert response.result_url is None
        assert response.error is None

    def test_task_response_with_result(self) -> None:
        """Test task response with result URL."""
        now = datetime.now(timezone.utc)
        response = TaskResponse(
            task_id="test-123",
            status=TaskStatus.COMPLETED,
            created_at=now,
            updated_at=now,
            result_url="/storage/final.png",
        )
        assert response.result_url == "/storage/final.png"


class TestTaskStatusResponse:
    """Tests for TaskStatusResponse schema."""

    def test_status_response(self) -> None:
        """Test status response fields."""
        response = TaskStatusResponse(
            task_id="test-123",
            status=TaskStatus.REMOVING_BG,
            progress="Removing background...",
        )
        assert response.task_id == "test-123"
        assert response.status == TaskStatus.REMOVING_BG
        assert response.progress == "Removing background..."


class TestUploadResponse:
    """Tests for UploadResponse schema."""

    def test_upload_response(self) -> None:
        """Test upload response fields."""
        response = UploadResponse(
            task_id="test-123",
            message="Image uploaded",
            status=TaskStatus.PENDING,
        )
        assert response.task_id == "test-123"
        assert response.message == "Image uploaded"
        assert response.status == TaskStatus.PENDING
