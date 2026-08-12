"""Deterministic local mock for Qwen Image Edit."""

from __future__ import annotations

import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from PIL import Image, ImageDraw, ImageEnhance, ImageOps


class MockQwenImageEditService:
    """Create valid edited PNG artifacts without API access."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(
            output_dir or Path(tempfile.gettempdir()) / "spagent_qwen_image_edit_mock"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def edit_image(
        self,
        image_path: str,
        prompt: str,
        reference_image_paths: Optional[Sequence[str]] = None,
        negative_prompt: Optional[str] = None,
        size: Optional[str] = None,
        n: int = 1,
        seed: Optional[int] = None,
        prompt_extend: bool = True,
        watermark: bool = False,
    ) -> Dict[str, Any]:
        del negative_prompt, prompt_extend
        try:
            sources = [image_path, *(reference_image_paths or [])]
            if not prompt or not prompt.strip():
                raise ValueError("Prompt must be a non-empty string.")
            if not 1 <= len(sources) <= 3:
                raise ValueError("Qwen Image Edit accepts one to three input images.")
            if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 6:
                raise ValueError("n must be an integer between 1 and 6.")

            images = [self._load_local_image(path) for path in sources]
            width, height = self._target_size(images[-1].size, size)
            request_id = uuid.uuid4().hex
            image_paths = []
            for index in range(n):
                output = self._compose_mock(
                    images=images,
                    size=(width, height),
                    prompt=prompt.strip(),
                    variant_seed=(seed or 0) + index,
                    watermark=watermark,
                )
                output_path = self.output_dir / (
                    f"qwen_image_edit_mock_{request_id}_{index + 1}.png"
                )
                output.save(output_path, format="PNG")
                image_paths.append(str(output_path.resolve()))

            file_sizes = [Path(path).stat().st_size for path in image_paths]
            return {
                "success": True,
                "output_path": image_paths[0],
                "image_paths": image_paths,
                "file_size_bytes": file_sizes[0],
                "file_sizes_bytes": file_sizes,
                "model": "mock-qwen-image-edit",
                "size": f"{width}*{height}",
                "seed": seed,
                "request_id": request_id,
                "usage": {"image_count": n, "width": width, "height": height},
                "mock": True,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _load_local_image(self, image_path: str) -> Image.Image:
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("Each image path must be a non-empty string.")
        if image_path.startswith(("http://", "https://", "oss://")):
            raise ValueError("Mock mode requires local image paths.")
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Image file not found: {image_path}")
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except Exception as exc:
            raise ValueError(f"Invalid image file: {image_path}") from exc

    def _target_size(
        self, fallback: tuple[int, int], size: Optional[str]
    ) -> tuple[int, int]:
        if size is None:
            return fallback
        try:
            width, height = (int(value) for value in size.split("*", 1))
        except Exception as exc:
            raise ValueError("size must use the 'width*height' format.") from exc
        if width <= 0 or height <= 0:
            raise ValueError("size dimensions must be positive.")
        return width, height

    def _compose_mock(
        self,
        images: Sequence[Image.Image],
        size: tuple[int, int],
        prompt: str,
        variant_seed: int,
        watermark: bool,
    ) -> Image.Image:
        base = ImageOps.fit(images[0], size, method=Image.Resampling.LANCZOS)
        base = ImageEnhance.Color(base).enhance(0.92 + (variant_seed % 5) * 0.03)
        canvas = base.copy()

        if len(images) > 1:
            thumb_width = max(48, size[0] // 4)
            thumb_height = max(48, size[1] // 4)
            margin = max(6, min(size) // 50)
            x = size[0] - thumb_width - margin
            y = margin
            for reference in images[1:]:
                thumb = ImageOps.fit(
                    reference,
                    (thumb_width, thumb_height),
                    method=Image.Resampling.LANCZOS,
                )
                canvas.paste(thumb, (x, y))
                y += thumb_height + margin

        digest = hashlib.sha256(f"{prompt}:{variant_seed}".encode("utf-8")).digest()
        accent = tuple(48 + value // 2 for value in digest[:3])
        draw = ImageDraw.Draw(canvas)
        border = max(2, min(size) // 100)
        draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=accent, width=border)
        label_height = max(22, min(56, size[1] // 8))
        draw.rectangle((0, size[1] - label_height, size[0], size[1]), fill=(18, 18, 20))
        prompt_label = (
            prompt[:80] if prompt.isascii() else f"instruction {digest.hex()[:12]}"
        )
        label = f"Qwen Image Edit mock | {prompt_label}"
        draw.text(
            (border * 2, size[1] - label_height + border * 2),
            label,
            fill=(245, 245, 245),
        )
        if watermark:
            draw.text((border * 2, border * 2), "Qwen-Image", fill=accent)
        return canvas
