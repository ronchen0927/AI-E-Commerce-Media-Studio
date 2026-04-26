"""Storyboard generation service using GPT-5.4-mini Vision."""

import asyncio
import base64
import json
import logging
import mimetypes

from openai import OpenAI

from app.schemas.video import StoryboardResponse, VideoScene

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert e-commerce video director. Analyze the product image and"
    " generate a storyboard for a short marketing video.\n\n"
    "Output ONLY valid JSON (no markdown fences, no commentary) matching this schema:\n"
    "{\n"
    '  "product_summary": "<one concise sentence describing the product>",\n'
    '  "scenes": [\n'
    "    {\n"
    '      "id": <integer starting at 1>,\n'
    '      "shot_type": "<specific shot, e.g. Extreme macro close-up>",\n'
    '      "camera_motion": "<specific motion, e.g. Slow dolly forward — ALWAYS'
    ' required>",\n'
    '      "visual_description": "<concrete visual details only, no abstract'
    ' intent>",\n'
    '      "duration_seconds": <integer 4 to 8>,\n'
    '      "prompt": "<AI video prompt, 40-80 words, English only>"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Prompt structure: [Camera angle + motion]. [Subject detail]. [Action]."
    " [Scene/environment]. [Light source and quality]. [Lens/optics]."
    " [Visual style]. [Color palette].\n"
    "Rules: ALWAYS specify camera motion. ALWAYS specify light source."
    " NEVER use intent language (no implying, representing, symbolizing)."
    " Describe only what is SEEN. 40-80 words, English only."
)

_TEMPLATE_SCENES = [
    VideoScene(
        id=1,
        shot_type="Extreme macro close-up",
        camera_motion="Slow dolly forward",
        visual_description="Product surface texture in extreme detail",
        duration_seconds=5,
        prompt=(
            "Extreme macro close-up, slow dolly forward. Product surface with fine"
            " texture detail. Camera drifts slowly toward surface. Neutral studio"
            " environment. Soft diffused overhead light, no harsh shadows. Shallow"
            " depth of field, sharp center focus. Cinematic product photography"
            " aesthetic. Clean white and neutral tones."
        ),
    ),
    VideoScene(
        id=2,
        shot_type="Medium shot, three-quarter angle",
        camera_motion="Slow 360-degree orbit",
        visual_description="Full product rotating on a clean white surface",
        duration_seconds=6,
        prompt=(
            "Medium shot, slow 360-degree orbit around product. Full product centered"
            " on clean white surface. Product rotates slowly to reveal all sides. Soft"
            " even studio lighting from above and both sides. No shadows. Standard"
            " lens, neutral perspective. Clean commercial product photography style."
            " Bright white palette."
        ),
    ),
    VideoScene(
        id=3,
        shot_type="Low angle hero shot",
        camera_motion="Slow pedestal up",
        visual_description="Product from below, dramatic upward perspective",
        duration_seconds=5,
        prompt=(
            "Low angle hero shot, slow pedestal up. Product viewed from below against"
            " bright background. Camera rises from ground level to eye level. Rim"
            " lighting from behind creating soft halo around product edges. Wide angle"
            " lens, dramatic perspective. Premium commercial aesthetic."
            " High-contrast bright tones."
        ),
    ),
    VideoScene(
        id=4,
        shot_type="Close-up, eye-level",
        camera_motion="Slow rack focus",
        visual_description="Product label or logo in sharp focus, background blurred",
        duration_seconds=5,
        prompt=(
            "Close-up, eye-level, slow rack focus from background to foreground."
            " Product label or logo transitions from blurred to tack-sharp. Soft"
            " background light, foreground lit by single soft box from left. Shallow"
            " depth of field, anamorphic lens. Luxury commercial aesthetic. Warm amber"
            " and white tones."
        ),
    ),
    VideoScene(
        id=5,
        shot_type="Wide establishing shot",
        camera_motion="Slow pan left to right",
        visual_description="Product displayed in lifestyle setting",
        duration_seconds=6,
        prompt=(
            "Wide establishing shot, slow pan left to right. Product displayed on"
            " natural wood surface with minimal lifestyle props. Camera drifts slowly"
            " across the scene. Warm natural window light from left, soft fill from"
            " right. Standard lens. Warm lifestyle commercial aesthetic. Warm white"
            " and natural wood tones."
        ),
    ),
]


class StoryboardService:
    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

    async def generate(
        self,
        image_path: str,
        image_storage_path: str,
        style: str,
        num_scenes: int,
    ) -> StoryboardResponse:
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None,
                self._generate_sync,
                image_path,
                image_storage_path,
                style,
                num_scenes,
            )
        except Exception as e:
            logger.warning(
                f"GPT-5.4-mini storyboard generation failed, using template: {e}"
            )
            return self._template_storyboard(image_storage_path, style, num_scenes)

    def _generate_sync(
        self,
        image_path: str,
        image_storage_path: str,
        style: str,
        num_scenes: int,
    ) -> StoryboardResponse:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        mime, _ = mimetypes.guess_type(image_path)
        mime = mime or "image/png"

        response = self._client.chat.completions.create(
            model="gpt-5.4-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Generate {num_scenes} scenes in '{style}' style."
                            ),
                        },
                    ],
                },
            ],
        )

        raw = response.choices[0].message.content or ""
        data = json.loads(raw)

        scenes = [VideoScene(**s) for s in data["scenes"][:num_scenes]]
        return StoryboardResponse(
            image_path=image_storage_path,
            product_summary=data.get("product_summary", ""),
            style=style,
            source="ai",
            scenes=scenes,
        )

    def _template_storyboard(
        self,
        image_storage_path: str,
        style: str,
        num_scenes: int,
    ) -> StoryboardResponse:
        scenes = [
            VideoScene(
                id=i + 1,
                shot_type=s.shot_type,
                camera_motion=s.camera_motion,
                visual_description=s.visual_description,
                duration_seconds=s.duration_seconds,
                prompt=s.prompt,
            )
            for i, s in enumerate(_TEMPLATE_SCENES[:num_scenes])
        ]
        return StoryboardResponse(
            image_path=image_storage_path,
            product_summary="",
            style=style,
            source="template",
            scenes=scenes,
        )
