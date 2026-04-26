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
    "You are an expert e-commerce video director creating prompts for an"
    " image-to-video AI model (Wan i2v).\n\n"
    "CRITICAL RULE: The uploaded image already defines ALL visual content — shape,"
    " color, material, environment. Your prompt controls ONLY how things MOVE."
    " NEVER describe what the product looks like. Only describe motion.\n\n"
    "Output ONLY valid JSON (no markdown fences, no commentary) matching this schema:\n"
    "{\n"
    '  "product_summary": "<one concise sentence describing the product>",\n'
    '  "scenes": [\n'
    "    {\n"
    '      "id": <integer starting at 1>,\n'
    '      "shot_type": "<specific shot, e.g. Extreme macro close-up>",\n'
    '      "camera_motion": "<specific motion, e.g. Slow counterclockwise orbit>",\n'
    '      "visual_description": "<what part of the product is featured in this'
    ' shot>",\n'
    '      "duration_seconds": <integer 4 to 8>,\n'
    '      "prompt": "<motion-only prompt, 40-80 words, English only>"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Prompt structure: [Subject motion] + [Camera movement] + [Subtle natural"
    " effect (optional)] + [Speed modifier].\n\n"
    "Subject motion — how the product itself moves: rotates slowly, tilts"
    " slightly, stays still while camera moves.\n"
    "Camera movement — ONE move only: slow dolly in, slow orbit, slow pedestal"
    " up, slow pan, rack focus pull.\n"
    "Subtle natural effect (optional, only when grounded in reality) — things"
    " that happen naturally on set: highlight glides across a glossy surface,"
    " shadow shifts as the camera angle changes, a nearby prop sways gently,"
    " shallow depth of field breathes slightly. AVOID fantasy effects such as"
    " floating particles, magical mist, or pulsing bokeh.\n"
    "Speed modifier: slow and fluid / smooth and continuous / gentle.\n\n"
    "Rules: ONE camera move per scene. Keep effects realistic and subtle."
    " NEVER describe color, shape, or appearance. 40-80 words, English only."
)

_TEMPLATE_SCENES = [
    VideoScene(
        id=1,
        shot_type="Extreme macro close-up",
        camera_motion="Slow dolly forward",
        visual_description="Product surface texture in extreme detail",
        duration_seconds=5,
        prompt=(
            "The product surface slowly rotates, bringing fine texture detail into"
            " frame. Camera drifts forward at a glacial pace."
            " A highlight glides slowly across the glossy surface as the angle shifts."
            " Smooth and fluid motion."
        ),
    ),
    VideoScene(
        id=2,
        shot_type="Medium shot, three-quarter angle",
        camera_motion="Slow counterclockwise orbit",
        visual_description="Full product on clean surface, all sides revealed",
        duration_seconds=6,
        prompt=(
            "The product rotates clockwise on its base, slowly revealing all sides."
            " Camera performs a slow counterclockwise orbit."
            " Surface reflections shift naturally as the viewing angle changes."
            " Smooth and continuous motion."
        ),
    ),
    VideoScene(
        id=3,
        shot_type="Low angle hero shot",
        camera_motion="Slow pedestal up",
        visual_description="Product viewed from below, rising upward perspective",
        duration_seconds=5,
        prompt=(
            "The product stands still as the camera slowly rises from ground level"
            " to eye level. The shadow cast by the product shrinks as the camera"
            " climbs. Slow and steady motion."
        ),
    ),
    VideoScene(
        id=4,
        shot_type="Close-up, eye-level",
        camera_motion="Slow push-in with rack focus",
        visual_description="Product label or logo area, transitioning to sharp focus",
        duration_seconds=5,
        prompt=(
            "Camera slowly pushes in while rack focus pulls from background to the"
            " product label, which transitions from blurred to tack-sharp."
            " The background gradually softens into smooth bokeh."
            " Slow and intimate motion."
        ),
    ),
    VideoScene(
        id=5,
        shot_type="Wide establishing shot",
        camera_motion="Slow pan left to right",
        visual_description="Product in lifestyle setting with surrounding props",
        duration_seconds=6,
        prompt=(
            "Camera drifts slowly from left to right across the scene."
            " A nearby fabric prop sways gently as if caught in a soft breeze."
            " Warm light shifts slightly, moving the shadow edges across the surface."
            " Gentle and natural motion."
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
