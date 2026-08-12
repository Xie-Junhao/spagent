"""Qwen Image Edit tool for instruction-based image editing and fusion."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool
from core.tool_result import IMAGE_GENERATION, MediaPayload, ToolResult

logger = logging.getLogger(__name__)


class QwenImageEditTool(Tool):
    """Edit one image or fuse multiple images with Qwen Image."""

    def __init__(
        self,
        use_mock: bool = True,
        api_key: Optional[str] = None,
        model: str = "qwen-image-2.0",
        base_url: Optional[str] = None,
        output_dir: Optional[str] = None,
        timeout: float = 300.0,
    ):
        super().__init__(
            name="qwen_image_edit_tool",
            description=(
                "Edit an existing image from a natural-language instruction using Qwen Image. "
                "Use it for adding, removing, replacing, restyling, relighting, changing text, "
                "or fusing content from up to two reference images. The returned image is newly "
                "generated and must not be treated as evidence about the original scene."
            ),
        )
        self.use_mock = use_mock
        self.model_name = model
        if use_mock:
            from external_experts.QwenImageEdit.mock_qwen_image_edit_service import (
                MockQwenImageEditService,
            )

            self._client = MockQwenImageEditService(output_dir=output_dir)
        else:
            from external_experts.QwenImageEdit.qwen_image_edit_client import (
                QwenImageEditClient,
            )

            self._client = QwenImageEditClient(
                api_key=api_key,
                model=model,
                base_url=base_url,
                output_dir=output_dir,
                timeout=timeout,
            )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Local path or public URL of the base image to edit.",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Precise editing instruction describing what to change and what to preserve."
                    ),
                },
                "reference_image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 2,
                    "description": (
                        "Optional one or two local image paths or URLs whose subjects or style "
                        "should be fused into the base image. Refer to them as Image 2 and Image 3."
                    ),
                },
                "negative_prompt": {
                    "type": "string",
                    "maxLength": 500,
                    "description": "Optional content or visual defects to avoid.",
                },
                "size": {
                    "type": "string",
                    "pattern": "^[0-9]{3,4}\\*[0-9]{3,4}$",
                    "description": (
                        "Optional output size in 'width*height' format, such as '1024*1024'."
                    ),
                },
                "n": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 6,
                    "default": 1,
                    "description": "Number of edited images to return.",
                },
                "seed": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 2147483647,
                    "description": "Optional random seed for relative reproducibility.",
                },
                "prompt_extend": {
                    "type": "boolean",
                    "default": True,
                    "description": "Let Qwen improve short editing instructions.",
                },
                "watermark": {
                    "type": "boolean",
                    "default": False,
                    "description": "Add the Qwen-Image watermark to generated images.",
                },
            },
            "required": ["image_path", "prompt"],
            "additionalProperties": False,
        }

    def call(
        self,
        image_path: str,
        prompt: str,
        reference_image_paths: Optional[List[str]] = None,
        negative_prompt: Optional[str] = None,
        size: Optional[str] = None,
        n: int = 1,
        seed: Optional[int] = None,
        prompt_extend: bool = True,
        watermark: bool = False,
    ) -> Dict[str, Any]:
        try:
            error = self._validate_inputs(
                image_path=image_path,
                prompt=prompt,
                reference_image_paths=reference_image_paths,
                negative_prompt=negative_prompt,
                size=size,
                n=n,
                seed=seed,
                prompt_extend=prompt_extend,
                watermark=watermark,
            )
            if error:
                return ToolResult(success=False, error=error, description=error)

            source_paths = [image_path, *(reference_image_paths or [])]
            result = self._client.edit_image(
                image_path=image_path,
                prompt=prompt,
                reference_image_paths=reference_image_paths,
                negative_prompt=negative_prompt,
                size=size,
                n=n,
                seed=seed,
                prompt_extend=prompt_extend,
                watermark=watermark,
            )
            if not result or not result.get("success"):
                error_msg = (
                    result.get("error", "No result returned")
                    if result
                    else "No result returned"
                )
                return ToolResult(
                    success=False,
                    error=f"Qwen Image Edit failed: {error_msg}",
                    description=f"Qwen Image Edit failed: {error_msg}",
                )

            output_path = result.get("output_path")
            image_paths = result.get("image_paths") or (
                [output_path] if output_path else []
            )
            if (
                not isinstance(output_path, str)
                or not output_path
                or not isinstance(image_paths, (list, tuple))
                or not image_paths
            ):
                return ToolResult(
                    success=False,
                    error="Qwen Image Edit returned no local output image.",
                    description="Qwen Image Edit returned no local output image.",
                )
            image_paths = list(image_paths)
            invalid_outputs = [
                path
                for path in image_paths
                if not isinstance(path, str) or not Path(path).is_file()
            ]
            if not Path(output_path).is_file() or invalid_outputs:
                error_msg = (
                    "Qwen Image Edit returned an unavailable local output image."
                )
                return ToolResult(
                    success=False,
                    error=error_msg,
                    description=error_msg,
                )

            payload = MediaPayload(
                category=IMAGE_GENERATION,
                output_path=output_path,
                image_paths=image_paths,
                metadata={
                    "model": result.get("model", self.model_name),
                    "size": result.get("size", size),
                    "seed": result.get("seed", seed),
                    "file_size_bytes": result.get("file_size_bytes"),
                },
            )
            return ToolResult(
                success=True,
                payload=payload,
                description=(
                    f"Qwen Image Edit created {len(image_paths)} edited image(s) "
                    f"from {len(source_paths)} source image(s)."
                ),
                output_path=output_path,
                image_paths=image_paths,
                source_image_paths=source_paths,
                prompt=prompt.strip(),
                request_id=result.get("request_id"),
                result=result,
            )
        except Exception as exc:
            logger.exception("Qwen Image Edit tool failed")
            return ToolResult(
                success=False,
                error=f"Qwen Image Edit failed: {exc}",
                description=f"Qwen Image Edit failed: {exc}",
            )

    def _validate_inputs(
        self,
        image_path: str,
        prompt: str,
        reference_image_paths: Optional[List[str]],
        negative_prompt: Optional[str],
        size: Optional[str],
        n: int,
        seed: Optional[int],
        prompt_extend: bool,
        watermark: bool,
    ) -> Optional[str]:
        if not isinstance(image_path, str) or not image_path.strip():
            return "image_path must be a non-empty string."
        if not isinstance(prompt, str) or not prompt.strip():
            return "Prompt must be a non-empty string."
        if reference_image_paths is not None and not isinstance(
            reference_image_paths, list
        ):
            return "reference_image_paths must be a list."
        if reference_image_paths and len(reference_image_paths) > 2:
            return "At most two reference images are supported."
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 6:
            return "n must be an integer between 1 and 6."
        if self.model_name == "qwen-image-edit" and n != 1:
            return "The qwen-image-edit model only supports n=1."
        if negative_prompt is not None and (
            not isinstance(negative_prompt, str) or len(negative_prompt) > 500
        ):
            return "negative_prompt must be a string of at most 500 characters."
        if seed is not None and (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not 0 <= seed <= 2147483647
        ):
            return "seed must be an integer between 0 and 2147483647."
        if not isinstance(prompt_extend, bool) or not isinstance(watermark, bool):
            return "prompt_extend and watermark must be booleans."
        if size is not None:
            if self.model_name == "qwen-image-edit":
                return "The qwen-image-edit model does not support custom size."
            match = (
                re.fullmatch(r"(\d{3,4})\*(\d{3,4})", size)
                if isinstance(size, str)
                else None
            )
            if not match:
                return (
                    "size must use the 'width*height' format, for example '1024*1024'."
                )
            width, height = (int(value) for value in match.groups())
            if width * height < 512 * 512 or width * height > 2048 * 2048:
                return "size must contain between 512*512 and 2048*2048 total pixels."
            if width > 2048 or height > 2048:
                return "size width and height must not exceed 2048 pixels."
            if self.model_name.startswith(
                ("qwen-image-edit-plus", "qwen-image-edit-max")
            ) and (width < 512 or height < 512):
                return "qwen-image-edit-plus/max require width and height between 512 and 2048."

        paths = [image_path, *(reference_image_paths or [])]
        for path in paths:
            if not isinstance(path, str) or not path.strip():
                return "Each image path must be a non-empty string."
            if path.startswith(("http://", "https://", "oss://")):
                continue
            if not Path(path).expanduser().is_file():
                return f"Image file not found: {path}"
        return None
