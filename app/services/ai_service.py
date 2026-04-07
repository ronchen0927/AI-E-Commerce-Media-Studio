"""
AI Service module for image processing pipeline.

This module provides async interfaces to AI models:
- RMBG-1.4 for background removal (API first, local fallback)
- FireRed-Image-Edit-1.1 for instruction-based editing
  (API first, local GGUF fallback)
"""

# Standard library imports
import base64
import logging
from pathlib import Path
from typing import Any, Optional

# Third-party imports
import httpx
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, ConfigDict
from skimage import io as skimage_io
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation

# Local imports
from app.core.config import get_settings

# Module-level state for lazy model loading
_rmbg_model: Any = None
_torch_device: Any = None

# FireRed-Image-Edit module-level state
_firered_pipe: Any = None
_firered_device: Any = None

logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    """Base exception for AI service errors."""

    pass


class AIService(BaseModel):
    """Base class for AI services using Pydantic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def process(self, image_path: str, **kwargs: str) -> str:
        """
        Process an image and return the path to the result.

        Args:
            image_path: Path to the input image.
            **kwargs: Additional parameters for the specific service.

        Returns:
            Path to the processed image.
        """
        raise NotImplementedError


def _load_rmbg_model() -> tuple[Any, Any]:
    """
    Lazy load RMBG-1.4 model and get device.

    Returns:
        Tuple of (model, device)
    """
    global _rmbg_model, _torch_device

    if _rmbg_model is not None:
        return _rmbg_model, _torch_device

    _torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading RMBG-1.4 model on {_torch_device}...")

    # WHY THIS IS NEEDED
    # ------------------
    # transformers 5.x added `all_tied_weights_keys` to PreTrainedModel.
    # BriaRMBG (loaded via trust_remote_code) predates it and doesn't define it.
    # BriaRMBG also uses PyTorch's custom class registry, so its __getattr__
    # raises a C++ torch::class_ error (not AttributeError) for unknown attrs.
    # This means hasattr() and sys.modules scanning both fail.
    #
    # HOW WE FIX IT
    # -------------
    # auto_factory.py does:
    #   model_class = get_class_from_dynamic_module(...)   # fresh class object
    #   model_class = add_generation_mixin_to_remote_model(model_class)
    #   return model_class.from_pretrained(...)            # calls all_tied_weights_keys
    #
    # We temporarily replace `get_class_from_dynamic_module` in the auto_factory
    # module so we intercept the *exact* class object that from_pretrained will
    # use, add the missing property, then restore the original.

    import transformers.models.auto.auto_factory as _af  # noqa: PLC0415

    _orig_gcfdm = _af.get_class_from_dynamic_module  # type: ignore[attr-defined]

    def _patched_gcfdm(
        class_reference: Any,
        pretrained_model_name_or_path: Any,
        **kwargs: Any,
    ) -> Any:
        cls = _orig_gcfdm(class_reference, pretrained_model_name_or_path, **kwargs)
        try:
            if getattr(cls, "__name__", None) == "BriaRMBG":
                if "all_tied_weights_keys" not in cls.__dict__:
                    cls.all_tied_weights_keys = property(lambda self: {})  # type: ignore[attr-defined]
                    logger.debug(
                        "Patched BriaRMBG.all_tied_weights_keys for transformers 5.x"
                    )
        except Exception as _patch_err:
            logger.warning(f"BriaRMBG patch failed inside loader: {_patch_err}")
        return cls

    _af.get_class_from_dynamic_module = _patched_gcfdm  # type: ignore[attr-defined, assignment]
    try:
        _rmbg_model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-1.4", trust_remote_code=True
        )
    finally:
        _af.get_class_from_dynamic_module = _orig_gcfdm  # type: ignore[attr-defined]  # always restore

    # `all_tied_weights_keys` is an *instance attribute* (a {target: source} dict)
    # set by PreTrainedModel.__init__ via get_expanded_tied_weights_keys().
    # BriaRMBG skips that, so we ensure it exists on the instance now.
    if "all_tied_weights_keys" not in _rmbg_model.__dict__:
        try:
            _rmbg_model.__dict__["all_tied_weights_keys"] = {}
        except Exception:
            pass

    _rmbg_model.to(_torch_device)
    _rmbg_model.eval()
    logger.info("RMBG-1.4 model loaded successfully")

    return _rmbg_model, _torch_device


def _load_firered_model() -> tuple[Any, Any]:
    """
    Lazy load FireRed-Image-Edit-1.1 model pipeline (GGUF or standard).

    Uses diffusers with GGUFQuantizationConfig for GGUF models,
    or loads the standard model if no GGUF path is configured.

    Returns:
        Tuple of (pipeline, device)
    """
    global _firered_pipe, _firered_device

    if _firered_pipe is not None:
        return _firered_pipe, _firered_device

    from diffusers import DiffusionPipeline

    _firered_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = get_settings()

    model_path = settings.firered_model_path

    if model_path and Path(model_path).exists() and model_path.endswith(".gguf"):
        # Load GGUF quantized model via diffusers
        # FireRed-Image-Edit-1.1 uses QwenImageTransformer2DModel (60 double
        # blocks, no single blocks).
        from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel

        logger.info(
            f"Loading FireRed-Image-Edit-1.1 GGUF model from {model_path} "
            f"on {_firered_device}..."
        )

        model_id = "FireRedTeam/FireRed-Image-Edit-1.1"

        # 1. Load GGUF Transformer
        gguf_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
        transformer = QwenImageTransformer2DModel.from_single_file(
            model_path,
            quantization_config=gguf_config,
            torch_dtype=torch.bfloat16,
            config=model_id,
            subfolder="transformer",
        )

        # 2. Assemble Pipeline
        _firered_pipe = DiffusionPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )

    else:
        # Load standard (non-GGUF) model from HuggingFace
        model_id = model_path if model_path else "FireRedTeam/FireRed-Image-Edit-1.1"
        logger.info(
            f"Loading FireRed-Image-Edit-1.1 model from {model_id} "
            f"on {_firered_device}..."
        )

        # Assemble Pipeline
        _firered_pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )

    # Memory optimization for consumer GPUs
    _firered_pipe.enable_model_cpu_offload(device=_firered_device)
    _firered_pipe.enable_attention_slicing()
    logger.info("FireRed-Image-Edit-1.1 model loaded successfully")

    return _firered_pipe, _firered_device


# --- Services ---


class BackgroundRemovalService(AIService):
    """Background removal service using RMBG-1.4.

    Strategy: API first -> local model fallback
    """

    api_url: Optional[str] = None
    use_local_model: bool = True

    def model_post_init(self, __context: object) -> None:
        """Initialize API URL from settings if not provided."""
        if self.api_url is None:
            settings = get_settings()
            self.api_url = settings.rmbg_api_url

    async def process(self, image_path: str, **kwargs: str) -> str:
        """Remove background from an image. API first, local fallback."""
        settings = get_settings()

        if settings.replicate_api_token:
            try:
                return await self._process_replicate(image_path)
            except Exception as e:
                import traceback

                logger.warning(f"Replicate RMBG failed: {e}")
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")

        elif self.api_url:
            try:
                return await self._process_api(image_path)
            except Exception as e:
                import traceback

                logger.warning(f"API RMBG failed: {e}")
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")

        if self.use_local_model:
            try:
                return await self._process_local_model(image_path)
            except Exception as e:
                logger.error(f"Local model fallback also failed: {e}")
                raise AIServiceError(f"Background removal failed: {e}") from e

        logger.warning("No API configured and local model disabled, using placeholder")
        return await self._process_placeholder(image_path)

    async def _process_local_model(self, image_path: str) -> str:
        """Process using local RMBG-1.4 model on GPU."""
        model, device = _load_rmbg_model()

        orig_im = skimage_io.imread(image_path)
        orig_im_size = orig_im.shape[0:2]
        model_input_size = [1024, 1024]

        image_tensor = self._preprocess_image(orig_im, model_input_size)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            result = model(image_tensor)

        mask = self._postprocess_mask(result[0][0], orig_im_size)

        output_path = Path(image_path).parent / "bg_removed.png"
        pil_mask = Image.fromarray(mask)
        orig_image = Image.open(image_path).convert("RGBA")

        no_bg_image = orig_image.copy()
        no_bg_image.putalpha(pil_mask)
        no_bg_image.save(output_path, "PNG")

        logger.info(f"Background removed via local model: {output_path}")
        return str(output_path)

    def _preprocess_image(
        self,
        im: np.ndarray,
        model_input_size: list[int],
    ) -> torch.Tensor:
        """Preprocess image for RMBG-1.4 model."""
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(
            torch.unsqueeze(im_tensor, 0), size=model_input_size, mode="bilinear"
        )
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        return torch.as_tensor(image)

    def _postprocess_mask(
        self,
        result: torch.Tensor,
        im_size: tuple[int, int],
    ) -> np.ndarray:
        """Postprocess model output to get mask."""
        if result.dim() == 3:
            result = result.unsqueeze(0)
        result = F.interpolate(
            result, size=im_size, mode="bilinear", align_corners=False
        )
        result = result.squeeze()
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        im_array = (result * 255).cpu().data.numpy().astype(np.uint8)
        return np.asarray(im_array)

    async def _process_api(self, image_path: str) -> str:
        """Process using remote API."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(image_path, "rb") as f:
                image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            headers = {"Content-Type": "application/json"}
            settings = get_settings()
            if settings.rmbg_api_key:
                headers["Authorization"] = f"Bearer {settings.rmbg_api_key}"

            response = await client.post(
                self.api_url,  # type: ignore[arg-type]
                json={"image": image_b64},
                headers=headers,
            )
            if response.status_code != 200:
                raise AIServiceError(
                    f"RMBG API error: {response.status_code} - {response.text}"
                )
            result = response.json()
            result_b64 = result.get("result", result.get("image", ""))
            output_path = Path(image_path).parent / "bg_removed.png"
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(result_b64))
            return str(output_path)

    async def _process_replicate(self, image_path: str) -> str:
        """Process using Replicate API."""
        import os

        import httpx
        import replicate

        settings = get_settings()
        os.environ["REPLICATE_API_TOKEN"] = settings.replicate_api_token

        logger.info("Sending image to Replicate RMBG API...")
        with open(image_path, "rb") as file_obj:
            output_url = await replicate.async_run(
                "bria/remove-background", input={"image": file_obj}
            )

        if isinstance(output_url, list) and len(output_url) > 0:
            output_url = str(output_url[0])
        else:
            output_url = str(output_url)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(output_url)
            response.raise_for_status()

            output_path = Path(image_path).parent / "bg_removed.png"
            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Replicate RMBG completed: {output_path}")
            return str(output_path)

    async def _process_placeholder(self, image_path: str) -> str:
        """Placeholder: simply converts image to RGBA."""
        output_path = Path(image_path).parent / "bg_removed.png"
        img = Image.open(image_path)
        output_img = img.convert("RGBA") if img.mode != "RGBA" else img
        output_img.save(output_path, "PNG")
        return str(output_path)


class FireRedEditService(AIService):
    """Instruction-based image editing service using FireRed-Image-Edit-1.1.

    Strategy: API first -> local GGUF model fallback
    """

    api_url: Optional[str] = None
    use_local_model: bool = True

    def model_post_init(self, __context: object) -> None:
        """Initialize API URL from settings if not provided."""
        if self.api_url is None:
            settings = get_settings()
            self.api_url = settings.firered_api_url

    async def process(self, image_path: str, **kwargs: str) -> str:
        """
        Edit an image based on an instruction. API first, local fallback.

        Args:
            image_path: Path to the input image.
            instruction: Text instruction for the edit
                (e.g. "Place this product on a marble table with warm lighting").
        """
        instruction = kwargs.get(
            "instruction",
            "professional product photography, clean white studio background, "
            "soft studio lighting, sharp focus, photorealistic",
        )

        settings = get_settings()

        if settings.replicate_api_token:
            try:
                return await self._process_replicate(image_path, instruction)
            except Exception as e:
                import traceback

                logger.warning(f"Replicate API failed: {e}")
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")

        elif self.api_url:
            try:
                return await self._process_api(image_path, instruction)
            except Exception as e:
                import traceback

                logger.warning(f"FireRed API failed: {e}")
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")

        if self.use_local_model:
            try:
                return await self._process_local_model(image_path, instruction)
            except Exception as e:
                logger.error(f"Local FireRed fallback also failed: {e}")
                raise AIServiceError(f"Image editing failed: {e}") from e

        logger.warning("No API configured and local model disabled")
        raise AIServiceError(
            "FireRed image editing unavailable: no API and local model disabled"
        )

    @torch.inference_mode()
    async def _process_local_model(
        self,
        image_path: str,
        instruction: str,
    ) -> str:
        """
        Process using local FireRed-Image-Edit-1.1 model (GGUF or standard).

        FireRed uses a FluxFillPipeline that takes:
        - image: the source image
        - mask_image: an all-white mask (edit the entire image)
        - prompt: the editing instruction
        """
        pipe, device = _load_firered_model()

        # Load and prepare input image
        source_img = Image.open(image_path).convert("RGB")

        # Resize to a working resolution while preserving aspect ratio
        max_side = 1024
        w, h = source_img.size
        scale = min(max_side / w, max_side / h, 1.0)
        new_w = int(w * scale) // 64 * 64  # Must be divisible by 64 for diffusers
        new_h = int(h * scale) // 64 * 64
        if new_w == 0:
            new_w = 64
        if new_h == 0:
            new_h = 64
        source_img = source_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Generate seed for reproducibility
        seed = int(torch.randint(0, 2147483647, (1,)).item())
        generator = torch.Generator(device=device).manual_seed(seed)

        logger.info(
            f"FireRed local inference: {new_w}x{new_h}, seed={seed}, "
            f"instruction='{instruction[:80]}...'"
        )

        # Run the pipeline
        result = pipe(
            image=source_img,
            prompt=instruction,
            height=new_h,
            width=new_w,
            num_inference_steps=28,
            guidance_scale=30.0,
            generator=generator,
        )

        output_image = result.images[0]

        # Save result
        output_path = Path(image_path).parent / "edited.png"
        output_image.save(output_path, "PNG")

        torch.cuda.empty_cache()

        logger.info(f"FireRed editing completed via local model: {output_path}")
        return str(output_path)

    async def _process_api(self, image_path: str, instruction: str) -> str:
        """Process using remote FireRed API."""
        async with httpx.AsyncClient(timeout=180.0) as client:
            with open(image_path, "rb") as f:
                image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode("utf-8")

            headers = {"Content-Type": "application/json"}
            settings = get_settings()
            if settings.firered_api_key:
                headers["Authorization"] = f"Bearer {settings.firered_api_key}"

            payload = {
                "image": image_b64,
                "instruction": instruction,
            }

            response = await client.post(
                self.api_url,  # type: ignore[arg-type]
                json=payload,
                headers=headers,
            )
            if response.status_code != 200:
                raise AIServiceError(
                    f"FireRed API error: {response.status_code} - {response.text}"
                )
            result = response.json()
            result_b64 = result.get("result", result.get("image", ""))
            output_path = Path(image_path).parent / "edited.png"
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(result_b64))
            return str(output_path)

    async def _process_replicate(self, image_path: str, instruction: str) -> str:
        """Process using Replicate API."""
        import io
        import os

        import httpx
        import replicate
        from PIL import Image

        settings = get_settings()
        # Set REPLICATE_API_TOKEN environment variable required by the replicate SDK
        os.environ["REPLICATE_API_TOKEN"] = settings.replicate_api_token

        # 1. Resize image to max 1024 to save bandwidth and prevent API overflow
        source_img = Image.open(image_path).convert("RGB")
        max_side = 1024
        w, h = source_img.size
        scale = min(max_side / w, max_side / h, 1.0)
        new_w = int(w * scale) // 64 * 64
        new_h = int(h * scale) // 64 * 64
        if new_w == 0:
            new_w = 64
        if new_h == 0:
            new_h = 64
        source_img = source_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Save to memory buffer
        img_byte_arr = io.BytesIO()
        source_img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        # 2. Call Replicate API
        logger.info(f"Sending image ({new_w}x{new_h}) to Replicate API...")
        output_url = await replicate.async_run(
            "prunaai/firered-image-edit-1.1:2275e825ae9ed8a17168e0ea82ae6722fe60ca25652bb9e61b98887eb0ad5bcc",
            input={"image": img_byte_arr, "prompt": instruction},
        )

        # Output from this model is typically a single URL string
        # or a list containing one URL
        if isinstance(output_url, list) and len(output_url) > 0:
            output_url = str(output_url[0])
        else:
            output_url = str(output_url)

        # 3. Download the result
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(output_url)
            response.raise_for_status()

            output_path = Path(image_path).parent / "edited.png"
            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Replicate API editing completed: {output_path}")
            return str(output_path)


class AIServiceFactory:
    """Factory for creating AI service instances."""

    @staticmethod
    def get_background_removal_service(
        use_local: bool = True,
    ) -> BackgroundRemovalService:
        """Get background removal service instance."""
        return BackgroundRemovalService(use_local_model=use_local)

    @staticmethod
    def get_firered_edit_service(
        use_local: bool = True,
    ) -> FireRedEditService:
        """Get FireRed image editing service instance."""
        return FireRedEditService(use_local_model=use_local)
