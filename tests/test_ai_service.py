"""Unit tests for AI services."""

from pathlib import Path

import pytest
from PIL import Image

from app.services.ai_service import (
    AIServiceFactory,
    BackgroundRemovalService,
    FireRedEditService,
)


class TestBackgroundRemovalService:
    """Tests for BackgroundRemovalService."""

    @pytest.fixture
    def service(self) -> BackgroundRemovalService:
        """Create service instance without API URL (local mode)."""
        return BackgroundRemovalService(api_url=None)

    @pytest.fixture
    def sample_image(self, tmp_path: Path) -> str:
        """Create a sample image for testing."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)
        return str(img_path)

    @pytest.mark.asyncio
    async def test_local_processing(
        self, service: BackgroundRemovalService, sample_image: str
    ) -> None:
        """Test local background removal creates output."""
        result = await service.process(sample_image)

        assert Path(result).exists()
        assert result.endswith("bg_removed.png")

        # Check output is RGBA
        output_img = Image.open(result)
        assert output_img.mode == "RGBA"


class TestFireRedEditService:
    """Tests for FireRedEditService."""

    @pytest.fixture
    def service(self) -> FireRedEditService:
        """Create service instance without API URL."""
        return FireRedEditService(api_url=None, use_local_model=False)

    def test_service_creation(self, service: FireRedEditService) -> None:
        """Test FireRedEditService can be instantiated."""
        assert not service.api_url  # Empty string or None
        assert service.use_local_model is False


class TestAIServiceFactory:
    """Tests for AIServiceFactory."""

    def test_get_background_removal_service(self) -> None:
        """Test factory returns BackgroundRemovalService."""
        service = AIServiceFactory.get_background_removal_service()
        assert isinstance(service, BackgroundRemovalService)

    def test_get_firered_edit_service(self) -> None:
        """Test factory returns FireRedEditService."""
        service = AIServiceFactory.get_firered_edit_service()
        assert isinstance(service, FireRedEditService)
