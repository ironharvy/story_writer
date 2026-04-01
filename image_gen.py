import os
import re
import base64
import urllib.request
from pathlib import Path
from typing import Optional


ANIMAGINE_MODEL = "aisha-ai-official/animagine-xl-4.0"
FLUX_KONTEXT_MODEL = "black-forest-labs/flux-kontext-pro"

ANIME_STYLE_SUFFIX = (
    "anime style, detailed anime illustration, clean lineart, "
    "vibrant colors, high quality, masterpiece"
)

NEGATIVE_PROMPT = (
    "photorealistic, 3d render, low quality, blurry, deformed, "
    "extra limbs, bad anatomy, watermark, text, signature"
)


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip()).strip("_")


def _save_image_from_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))
    return dest


def _image_to_data_uri(path: Path) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    suffix = path.suffix.lstrip(".").lower()
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return f"data:{mime};base64,{data}"


def _get_replicate_module():
    try:
        import replicate
    except ImportError as exc:
        raise RuntimeError(
            "Replicate support is not installed. Install the optional image "
            "dependencies to enable image generation."
        ) from exc
    return replicate


class ImageGenerator:
    """Wraps Replicate API for anime character portraits and scene illustrations."""

    def __init__(self, api_token: Optional[str] = None, output_dir: str = ".tmp/images"):
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN", "")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.api_token:
            os.environ["REPLICATE_API_TOKEN"] = self.api_token

    def generate_character_portrait(self, prompt: str, character_name: str) -> str:
        """Generate an anime portrait via Animagine XL 4.0.

        Returns the path to the saved image file.
        """
        full_prompt = f"1girl/1boy, portrait, upper body, {prompt}, {ANIME_STYLE_SUFFIX}"
        replicate = _get_replicate_module()

        output = replicate.run(
            ANIMAGINE_MODEL,
            input={
                "prompt": full_prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "num_inference_steps": 28,
                "guidance_scale": 7.0,
                "width": 768,
                "height": 1024,
            },
        )

        image_url = output[0] if isinstance(output, list) else str(output)
        filename = f"char_portrait_{_sanitize_filename(character_name)}.png"
        dest = self.output_dir / filename
        _save_image_from_url(image_url, dest)
        return str(dest)

    def generate_scene_illustration(
        self,
        prompt: str,
        reference_image_path: Optional[str],
        chapter_index: int,
    ) -> str:
        """Generate a scene illustration via FLUX Kontext, using a character
        reference image to preserve identity.

        Returns the path to the saved image file.
        """
        replicate = _get_replicate_module()
        input_params: dict = {
            "prompt": f"{prompt}, {ANIME_STYLE_SUFFIX}",
            "aspect_ratio": "16:9",
        }

        if reference_image_path:
            input_params["image"] = _image_to_data_uri(Path(reference_image_path))

        output = replicate.run(FLUX_KONTEXT_MODEL, input=input_params)

        image_url = output[0] if isinstance(output, list) else str(output)
        filename = f"chapter_{chapter_index}_scene.png"
        dest = self.output_dir / filename
        _save_image_from_url(image_url, dest)
        return str(dest)
