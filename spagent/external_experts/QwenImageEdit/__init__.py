"""Qwen Image Edit API client and mock service.

Imports are lazy so mock-only catalog builds do not require the HTTP stack.
"""

from typing import Any

__all__ = ["MockQwenImageEditService", "QwenImageEditClient"]


def __getattr__(name: str) -> Any:
    if name == "MockQwenImageEditService":
        from .mock_qwen_image_edit_service import MockQwenImageEditService

        return MockQwenImageEditService
    if name == "QwenImageEditClient":
        from .qwen_image_edit_client import QwenImageEditClient

        return QwenImageEditClient
    raise AttributeError(name)
