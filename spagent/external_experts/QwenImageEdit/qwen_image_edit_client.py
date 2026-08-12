"""Client for Qwen Image editing through the DashScope native API."""

from __future__ import annotations

import base64
import io
import logging
import math
import mimetypes
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
GENERATE_ENDPOINT = "/services/aigc/multimodal-generation/generation"
SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP", "GIF"}
MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_OUTPUT_BYTES = 50 * 1024 * 1024
MAX_SEED = 2_147_483_647
SIZE_PATTERN = re.compile(r"^(\d{3,4})\*(\d{3,4})$")


class QwenImageEditClient:
    """Synchronous Qwen Image Edit client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-image-2.0",
        base_url: Optional[str] = None,
        output_dir: Optional[str] = None,
        timeout: float = 300.0,
        http_client: Optional[httpx.Client] = None,
    ):
        resolved_api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not isinstance(resolved_api_key, str) or not resolved_api_key.strip():
            raise ValueError(
                "DashScope API key is required. Set DASHSCOPE_API_KEY or pass api_key."
            )
        self.api_key = resolved_api_key.strip()
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")

        self.model = model.strip()
        base_url_value = (
            base_url or os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL
        )
        if not isinstance(base_url_value, str) or not base_url_value.strip():
            raise ValueError("base_url must be a non-empty HTTP(S) URL.")
        resolved_base_url = base_url_value.strip().rstrip("/")
        parsed_base_url = urlparse(resolved_base_url)
        if (
            parsed_base_url.scheme not in ("http", "https")
            or not parsed_base_url.netloc
        ):
            raise ValueError("base_url must be an HTTP(S) URL.")
        if parsed_base_url.username or parsed_base_url.password:
            raise ValueError("base_url must not contain credentials.")
        if parsed_base_url.query or parsed_base_url.fragment:
            raise ValueError("base_url must not contain a query string or fragment.")
        if not resolved_base_url.endswith("/api/v1"):
            resolved_base_url += "/api/v1"
        self.base_url = resolved_base_url
        self.output_dir = Path(
            output_dir or Path(tempfile.gettempdir()) / "spagent_qwen_image_edit"
        ).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number.")
        self._http_client = http_client

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
        """Edit one image or fuse up to three images using a text instruction."""
        try:
            if isinstance(reference_image_paths, (str, bytes)):
                raise ValueError(
                    "reference_image_paths must be a sequence of image paths."
                )
            image_paths = [image_path, *(reference_image_paths or [])]
            self._validate_options(
                image_paths=image_paths,
                prompt=prompt,
                negative_prompt=negative_prompt,
                size=size,
                n=n,
                seed=seed,
                prompt_extend=prompt_extend,
                watermark=watermark,
            )

            content = [{"image": self._resolve_image(path)} for path in image_paths]
            content.append({"text": prompt.strip()})
            parameters: Dict[str, Any] = {
                "n": n,
                "watermark": bool(watermark),
            }
            if self.model != "qwen-image-edit":
                parameters["prompt_extend"] = bool(prompt_extend)
            if negative_prompt is not None:
                parameters["negative_prompt"] = negative_prompt
            if size is not None:
                parameters["size"] = size
            if seed is not None:
                parameters["seed"] = seed

            payload = {
                "model": self.model,
                "input": {
                    "messages": [
                        {"role": "user", "content": content},
                    ]
                },
                "parameters": parameters,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            client = self._http_client or httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
            )
            close_client = self._http_client is None
            try:
                response = client.post(
                    f"{self.base_url}{GENERATE_ENDPOINT}",
                    headers=headers,
                    json=payload,
                )
                data = self._response_json(response)
                if response.status_code < 200 or response.status_code >= 300:
                    return self._api_error(response.status_code, data)
                if data.get("code") and not data.get("output"):
                    return self._api_error(response.status_code, data)

                image_urls = self._extract_image_urls(data)[:n]
                if not image_urls:
                    return {
                        "success": False,
                        "error": "Qwen Image Edit returned no output image URL.",
                        "request_id": data.get("request_id"),
                    }

                request_id = data.get("request_id") or uuid.uuid4().hex
                image_paths_out: List[str] = []
                try:
                    for index, url in enumerate(image_urls, start=1):
                        image_paths_out.append(
                            self._download_image(client, url, request_id, index)
                        )
                except Exception:
                    for partial_path in image_paths_out:
                        Path(partial_path).unlink(missing_ok=True)
                    raise
            finally:
                if close_client:
                    client.close()

            usage_value = data.get("usage")
            usage: Dict[str, Any] = usage_value if isinstance(usage_value, dict) else {}
            file_sizes = [Path(path).stat().st_size for path in image_paths_out]
            return {
                "success": True,
                "output_path": image_paths_out[0],
                "image_paths": image_paths_out,
                "file_size_bytes": file_sizes[0],
                "file_sizes_bytes": file_sizes,
                "model": self.model,
                "size": self._usage_size(usage) or size,
                "seed": seed,
                "request_id": request_id,
                "usage": usage,
            }
        except (ValueError, OSError) as exc:
            return {"success": False, "error": str(exc)}
        except httpx.HTTPError as exc:
            # Download errors can contain signed result URLs. Do not expose
            # their query strings in logs or tool results.
            if isinstance(exc, httpx.HTTPStatusError):
                detail = f"HTTP {exc.response.status_code}"
            else:
                detail = exc.__class__.__name__
            logger.error("Qwen Image Edit HTTP request failed (%s)", detail)
            return {
                "success": False,
                "error": f"Qwen Image Edit request failed ({detail}).",
            }
        except Exception as exc:
            logger.exception("Unexpected Qwen Image Edit client error")
            return {"success": False, "error": f"Qwen Image Edit failed: {exc}"}

    def _validate_options(
        self,
        image_paths: Sequence[str],
        prompt: str,
        negative_prompt: Optional[str],
        size: Optional[str],
        n: int,
        seed: Optional[int],
        prompt_extend: bool,
        watermark: bool,
    ) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        if not 1 <= len(image_paths) <= 3:
            raise ValueError("Qwen Image Edit accepts one to three input images.")
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 6:
            raise ValueError("n must be an integer between 1 and 6.")
        if self.model == "qwen-image-edit" and n != 1:
            raise ValueError("The qwen-image-edit model only supports n=1.")
        if negative_prompt is not None and (
            not isinstance(negative_prompt, str) or len(negative_prompt) > 500
        ):
            raise ValueError(
                "negative_prompt must be a string of at most 500 characters."
            )
        if seed is not None and (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not 0 <= seed <= MAX_SEED
        ):
            raise ValueError(f"seed must be an integer between 0 and {MAX_SEED}.")
        if not isinstance(prompt_extend, bool) or not isinstance(watermark, bool):
            raise ValueError("prompt_extend and watermark must be booleans.")
        if size is not None:
            if not isinstance(size, str):
                raise ValueError("size must be a string in 'width*height' format.")
            if self.model == "qwen-image-edit":
                raise ValueError(
                    "The qwen-image-edit model does not support custom size."
                )
            self._validate_size(size)

    def _validate_size(self, size: str) -> None:
        match = SIZE_PATTERN.fullmatch(size)
        if not match:
            raise ValueError(
                "size must use the 'width*height' format, for example '1024*1024'."
            )
        width, height = (int(value) for value in match.groups())
        pixels = width * height
        if pixels < 512 * 512 or pixels > 2048 * 2048:
            raise ValueError(
                "size must contain between 512*512 and 2048*2048 total pixels."
            )
        if self.model.startswith(("qwen-image-edit-plus", "qwen-image-edit-max")) and (
            not 512 <= width <= 2048 or not 512 <= height <= 2048
        ):
            raise ValueError(
                "qwen-image-edit-plus/max require width and height between 512 and 2048."
            )

    def _resolve_image(self, image_path: str) -> str:
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("Each image path must be a non-empty string.")
        value = image_path.strip()
        if value.startswith(("http://", "https://", "oss://")):
            return value

        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"Image file not found: {image_path}")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Could not read image file: {image_path}") from exc
        if len(content) > MAX_INPUT_BYTES:
            raise ValueError(f"Input image exceeds the 10 MB API limit: {image_path}")

        try:
            with Image.open(io.BytesIO(content)) as image:
                image_format = (image.format or "").upper()
                image.verify()
        except Exception as exc:
            raise ValueError(f"Invalid image file: {image_path}") from exc
        if image_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format {image_format or 'unknown'}: {image_path}"
            )

        # Prefer the decoded format over the filename extension.
        mime_type = Image.MIME.get(image_format) or mimetypes.guess_type(path.name)[0]
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"Could not determine image MIME type: {image_path}")
        encoded = base64.b64encode(content).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _response_json(self, response: httpx.Response) -> Dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError(
                f"DashScope returned a non-JSON response (HTTP {response.status_code})."
            ) from exc
        if not isinstance(data, dict):
            raise ValueError("DashScope returned an invalid JSON response.")
        return data

    def _api_error(self, status_code: int, data: Dict[str, Any]) -> Dict[str, Any]:
        code = self._redact_api_value(data.get("code") or "APIError")
        message = self._redact_api_value(data.get("message") or f"HTTP {status_code}")
        raw_request_id = data.get("request_id")
        request_id = (
            self._redact_api_value(raw_request_id)
            if raw_request_id is not None
            else None
        )
        suffix = f" (request_id={request_id})" if request_id else ""
        return {
            "success": False,
            "error": f"Qwen Image Edit API error {code}: {message}{suffix}",
            "request_id": request_id,
        }

    def _redact_api_value(self, value: Any) -> str:
        return str(value).replace(self.api_key, "[REDACTED]")

    def _extract_image_urls(self, data: Dict[str, Any]) -> List[str]:
        urls: List[str] = []
        output = data.get("output")
        if not isinstance(output, dict):
            return urls

        choices = output.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            url = item.get("image") or item.get("url")
                            if isinstance(url, str) and url:
                                urls.append(url)

        results = output.get("results")
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    url = item.get("url") or item.get("image")
                    if isinstance(url, str) and url:
                        urls.append(url)

        return list(dict.fromkeys(urls))

    def _download_image(
        self,
        client: httpx.Client,
        image_url: str,
        request_id: str,
        index: int,
    ) -> str:
        if not image_url.startswith(("http://", "https://")):
            raise ValueError("Qwen Image Edit returned an unsupported image URL.")
        content_buffer = io.BytesIO()
        with client.stream("GET", image_url) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                if content_buffer.tell() + len(chunk) > MAX_OUTPUT_BYTES:
                    raise ValueError(
                        "Qwen Image Edit output exceeds the 50 MB safety limit."
                    )
                content_buffer.write(chunk)
        content = content_buffer.getvalue()
        if not content:
            raise ValueError("Qwen Image Edit returned an empty image file.")

        try:
            with Image.open(io.BytesIO(content)) as image:
                image.load()
                has_alpha = "A" in image.getbands() or "transparency" in image.info
                output_image = image.convert("RGBA" if has_alpha else "RGB")
        except Exception as exc:
            raise ValueError("Qwen Image Edit output is not a valid image.") from exc

        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(request_id))[:80]
        final_path = self.output_dir / f"qwen_image_edit_{safe_id}_{index}.png"
        temp_path = self.output_dir / f".{final_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            output_image.save(temp_path, format="PNG")
            temp_path.replace(final_path)
        finally:
            temp_path.unlink(missing_ok=True)
        return str(final_path.resolve())

    @staticmethod
    def _usage_size(usage: Dict[str, Any]) -> Optional[str]:
        width, height = usage.get("width"), usage.get("height")
        if isinstance(width, int) and isinstance(height, int):
            return f"{width}*{height}"
        return None
