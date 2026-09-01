"""Tests for QwenImageEditTool and its DashScope HTTP client."""

from __future__ import annotations

import importlib.util
import json
import os
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from spagent.tools import QwenImageEditTool
from spagent.tools.catalog import build_tools, resolve_tool_keys
from spagent.core.prompts import build_tool_selection_guide
from spagent.core.render import render
from spagent.core.tool import Tool
from spagent.core.tool_result import ToolResult, validate_payload


@pytest.fixture
def source_images(tmp_path: Path) -> list[str]:
    paths = []
    for index, color in enumerate(((210, 55, 45), (40, 125, 205), (45, 170, 95))):
        path = tmp_path / f"source_{index}.png"
        Image.new("RGB", (96 + index * 8, 72 + index * 8), color).save(path)
        paths.append(str(path))
    return paths


def _png_bytes(color=(120, 80, 220), size=(80, 64)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


requires_httpx = pytest.mark.skipif(
    importlib.util.find_spec("httpx") is None,
    reason="httpx is not installed in this test environment",
)
TEST_BASE_URL = "https://workspace.test/api/v1"


def test_tool_is_exported_and_schema_is_valid():
    tool = QwenImageEditTool(use_mock=True)
    schema = tool.to_function_schema()

    assert isinstance(tool, Tool)
    assert tool.name == "qwen_image_edit_tool"
    assert schema["function"]["parameters"]["required"] == ["image_path", "prompt"]
    properties = schema["function"]["parameters"]["properties"]
    assert properties["reference_image_paths"]["maxItems"] == 2
    assert properties["n"]["minimum"] == 1
    assert properties["n"]["maximum"] == 6
    json.dumps(schema)
    if importlib.util.find_spec("jsonschema") is not None:
        from jsonschema import Draft7Validator

        Draft7Validator.check_schema(schema["function"]["parameters"])


def test_mock_single_image_returns_standardized_renderable_result(
    source_images, tmp_path
):
    tool = QwenImageEditTool(use_mock=True, output_dir=str(tmp_path / "outputs"))
    result = tool.call(
        image_path=source_images[0],
        prompt="Add a small blue ceramic cup while preserving the red background.",
        seed=17,
    )

    assert isinstance(result, ToolResult)
    assert result["success"] is True
    assert result["category"] == "image_generation"
    assert validate_payload(result, result["category"])[0]
    json.dumps(result)
    assert Path(result["output_path"]).is_file()
    with Image.open(result["output_path"]) as output:
        assert output.format == "PNG"
        assert output.size == (96, 72)

    rendered = render(result, tool_name=tool.name)
    assert rendered.images == [result["output_path"]]
    assert "edited image" in rendered.text


def test_mock_multi_image_fusion_and_multiple_outputs(source_images, tmp_path):
    tool = QwenImageEditTool(use_mock=True, output_dir=str(tmp_path / "outputs"))
    result = tool.call(
        image_path=source_images[0],
        reference_image_paths=source_images[1:],
        prompt="Use Image 1 as the scene and place the colors of Images 2 and 3 into it.",
        size="512*512",
        n=3,
        seed=5,
        watermark=True,
    )

    assert result["success"] is True
    assert len(result["source_image_paths"]) == 3
    assert len(result["image_paths"]) == 3
    assert len(set(result["image_paths"])) == 3
    for output_path in result["image_paths"]:
        with Image.open(output_path) as output:
            assert output.size == (512, 512)
    assert render(result, tool_name=tool.name).images == [result["output_path"]]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt": " "}, "Prompt must be"),
        ({"image_path": "missing.png"}, "Image file not found"),
        ({"reference_image_paths": ["a", "b", "c"]}, "At most two"),
        ({"n": 0}, "n must be"),
        ({"seed": -1}, "seed must be"),
        ({"size": "200*200"}, "between 512*512"),
        ({"negative_prompt": "x" * 501}, "at most 500"),
        ({"prompt_extend": 1}, "must be booleans"),
    ],
)
def test_tool_rejects_invalid_inputs(source_images, kwargs, message):
    call_kwargs = {"image_path": source_images[0], "prompt": "Make a careful edit."}
    call_kwargs.update(kwargs)
    result = QwenImageEditTool(use_mock=True).call(**call_kwargs)
    assert result["success"] is False
    assert message in result["error"]


def test_legacy_model_constraints_are_checked(source_images):
    tool = QwenImageEditTool(use_mock=True, model="qwen-image-edit")
    result = tool.call(image_path=source_images[0], prompt="Edit it", n=2)
    assert result["success"] is False
    assert "only supports n=1" in result["error"]


def test_model_name_is_normalized_and_validated(source_images):
    tool = QwenImageEditTool(use_mock=True, model="  qwen-image-edit  ")
    result = tool.call(image_path=source_images[0], prompt="Edit it", n=2)
    assert result["success"] is False
    assert "only supports n=1" in result["error"]

    with pytest.raises(ValueError, match="non-empty"):
        QwenImageEditTool(use_mock=True, model="   ")


def test_plus_model_dimensions_are_checked(source_images):
    tool = QwenImageEditTool(use_mock=True, model="qwen-image-edit-plus")
    result = tool.call(
        image_path=source_images[0],
        prompt="Edit it",
        size="400*800",
    )
    assert result["success"] is False
    assert "require width and height" in result["error"]


def test_qwen_image_2_size_uses_total_pixel_limit(source_images):
    tool = QwenImageEditTool(use_mock=True, model="qwen-image-2.0")
    error = tool._validate_inputs(
        image_path=source_images[0],
        prompt="Create a wide edit.",
        reference_image_paths=None,
        negative_prompt=None,
        size="2688*1536",
        n=1,
        seed=None,
        prompt_extend=True,
        watermark=False,
    )
    assert error is None


@requires_httpx
def test_http_client_applies_model_specific_size_limits(tmp_path):
    from spagent.external_experts.QwenImageEdit import QwenImageEditClient

    client = QwenImageEditClient(
        api_key="test-key",
        model="qwen-image-2.0",
        base_url=TEST_BASE_URL,
        output_dir=str(tmp_path),
    )
    client._validate_size("2688*1536")

    plus_client = QwenImageEditClient(
        api_key="test-key",
        model="qwen-image-edit-plus",
        output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="width and height"):
        plus_client._validate_size("2688*1536")


@requires_httpx
def test_qwen_image_2_requires_workspace_endpoint(monkeypatch):
    from spagent.external_experts.QwenImageEdit import QwenImageEditClient

    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="workspace endpoint"):
        QwenImageEditClient(api_key="test-key", model="qwen-image-2.0")

    legacy = QwenImageEditClient(api_key="test-key", model="qwen-image-edit")
    assert legacy.base_url == "https://dashscope.aliyuncs.com/api/v1"


def test_catalog_builds_tool_and_resolves_function_name():
    assert resolve_tool_keys(["qwen_image_edit_tool"]) == (["qwen_image_edit"], [])
    tools, skipped = build_tools(["qwen_image_edit"], use_mock=True, strict=True)
    assert skipped == []
    assert [tool.name for tool in tools] == ["qwen_image_edit_tool"]


def test_real_workflows_expose_dashscope_configuration():
    workflow_dir = Path(__file__).parents[1] / ".github" / "workflows"
    for workflow_name in ("tool-real-smoke.yml", "tool-agent-e2e.yml"):
        workflow = (workflow_dir / workflow_name).read_text(encoding="utf-8")
        assert "DASHSCOPE_API_KEY: ${{ secrets.DASHSCOPE_API_KEY }}" in workflow
        assert "DASHSCOPE_BASE_URL: ${{ secrets.DASHSCOPE_BASE_URL }}" in workflow


def test_agent_selects_generation_workflow_and_tool_guide():
    from spagent import SPAgent

    class MockModel:
        model_name = "mock"

    tool = QwenImageEditTool(use_mock=True)
    agent = SPAgent(model=MockModel(), tools=[tool], workflow_mode="auto")
    assert (
        agent._select_workflow("Edit this image and replace the background.", ["a.png"])
        == "generation"
    )
    assert agent._select_workflow("请编辑图像并更换背景。", ["a.png"]) == "generation"
    guide = build_tool_selection_guide({tool.name})
    assert tool.name in guide
    assert "what to preserve" in guide


def test_spagent_executes_image_edit_end_to_end(source_images, tmp_path):
    from spagent import SPAgent
    from spagent.core.model import Model

    class ScriptedModel(Model):
        def __init__(self, responses):
            super().__init__(model_name="scripted")
            self.responses = list(responses)
            self.seen_image_counts = []

        def _next(self, image_count):
            self.seen_image_counts.append(image_count)
            return self.responses.pop(0)

        def single_image_inference(self, image_path, prompt, **kwargs):
            del image_path, prompt, kwargs
            return self._next(1)

        def multiple_images_inference(self, image_paths, prompt, **kwargs):
            del prompt, kwargs
            return self._next(len(image_paths))

        def text_only_inference(self, prompt, **kwargs):
            del prompt, kwargs
            return self._next(0)

    arguments = {
        "image_path": source_images[0],
        "prompt": "Replace the background with a clean blue studio wall.",
        "seed": 7,
    }
    tool_call = (
        "<tool_call>"
        + json.dumps({"name": "qwen_image_edit_tool", "arguments": arguments})
        + "</tool_call>"
    )
    model = ScriptedModel([tool_call, "<answer>Edited image created.</answer>"])
    tool = QwenImageEditTool(use_mock=True, output_dir=str(tmp_path / "agent_outputs"))
    agent = SPAgent(model=model, tools=[tool], workflow_mode="auto")

    result = agent.step(
        content="Edit this image and replace the background.",
        images=source_images[0],
        max_tool_iterations=2,
    )

    assert result.prompts["workflow"] == "generation"
    assert len(result.tool_results) == 1
    tool_result = next(iter(result.tool_results.values()))
    assert tool_result["success"] is True
    assert result.additional_images == [tool_result["output_path"]]
    assert Path(result.additional_images[0]).is_file()
    assert model.seen_image_counts == [1, 2]


def test_tool_rejects_success_with_missing_output(source_images, tmp_path):
    tool = QwenImageEditTool(use_mock=True)

    class BrokenClient:
        def edit_image(self, **kwargs):
            del kwargs
            missing = str(tmp_path / "missing.png")
            return {
                "success": True,
                "output_path": missing,
                "image_paths": [missing],
            }

    tool._client = BrokenClient()
    result = tool.call(source_images[0], "Edit it carefully.")
    assert result["success"] is False
    assert "unavailable local output" in result["error"]


def test_tool_preserves_failed_api_request_id(source_images):
    tool = QwenImageEditTool(use_mock=True)

    class FailedClient:
        def edit_image(self, **kwargs):
            del kwargs
            return {
                "success": False,
                "error": "API request failed.",
                "request_id": "req-debug-1",
            }

    tool._client = FailedClient()
    result = tool.call(source_images[0], "Edit it carefully.")
    assert result["success"] is False
    assert result["request_id"] == "req-debug-1"


@requires_httpx
def test_http_client_matches_official_request_and_downloads_png(
    source_images, tmp_path
):
    import httpx

    from spagent.external_experts.QwenImageEdit import QwenImageEditClient

    generated = _png_bytes()
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers["authorization"]
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "output": {
                        "choices": [
                            {
                                "message": {
                                    "content": [
                                        {"image": "https://result.test/edit-1.png"},
                                        {"image": "https://result.test/edit-2.png"},
                                    ]
                                }
                            }
                        ]
                    },
                    "usage": {"image_count": 2, "width": 80, "height": 64},
                    "requestId": "request/unsafe id",
                },
            )
        assert str(request.url) in (
            "https://result.test/edit-1.png",
            "https://result.test/edit-2.png",
        )
        return httpx.Response(
            200, content=generated, headers={"content-type": "image/png"}
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = QwenImageEditClient(
        api_key="test-key",
        model="qwen-image-2.0",
        base_url="https://workspace.test/api/v1/",
        output_dir=str(tmp_path / "outputs"),
        http_client=http_client,
    )
    try:
        result = client.edit_image(
            image_path=source_images[0],
            prompt="Replace the background with a clean studio wall.",
            reference_image_paths=[source_images[1]],
            negative_prompt="blur",
            size="1024*1024",
            n=2,
            seed=9,
            prompt_extend=False,
        )
    finally:
        http_client.close()

    assert result["success"] is True
    assert result["size"] == "80*64"
    assert len(result["image_paths"]) == 2
    assert Path(result["output_path"]).name == "qwen_image_edit_request_unsafe_id_1.png"
    with Image.open(result["output_path"]) as output:
        assert output.size == (80, 64)

    payload = captured["payload"]
    assert captured["url"].endswith("/services/aigc/multimodal-generation/generation")
    assert captured["authorization"] == "Bearer test-key"
    assert payload["model"] == "qwen-image-2.0"
    content = payload["input"]["messages"][0]["content"]
    assert len(content) == 3
    assert content[0]["image"].startswith("data:image/png;base64,")
    assert content[1]["image"].startswith("data:image/png;base64,")
    assert content[2] == {"text": "Replace the background with a clean studio wall."}
    assert payload["parameters"] == {
        "n": 2,
        "watermark": False,
        "prompt_extend": False,
        "negative_prompt": "blur",
        "size": "1024*1024",
        "seed": 9,
    }


@requires_httpx
def test_http_client_uses_decoded_image_mime_type(tmp_path):
    import httpx

    from spagent.external_experts.QwenImageEdit import QwenImageEditClient

    disguised_png = tmp_path / "actually_png.jpg"
    disguised_png.write_bytes(_png_bytes())
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "output": {
                        "choices": [
                            {
                                "message": {
                                    "content": [
                                        {"image": "https://result.test/edit.png"}
                                    ]
                                }
                            }
                        ]
                    },
                    "request_id": "mime-test",
                },
            )
        return httpx.Response(200, content=_png_bytes())

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = QwenImageEditClient(
        api_key="test-key",
        output_dir=str(tmp_path / "out"),
        base_url=TEST_BASE_URL,
        http_client=http_client,
    )
    try:
        result = client.edit_image(str(disguised_png), "Edit it")
    finally:
        http_client.close()
    assert result["success"] is True
    image_value = captured["payload"]["input"]["messages"][0]["content"][0]["image"]
    assert image_value.startswith("data:image/png;base64,")


@requires_httpx
def test_http_client_hides_signed_output_url_on_download_error(source_images, tmp_path):
    import httpx

    from spagent.external_experts.QwenImageEdit import QwenImageEditClient

    signed_url = "https://result.test/edit.png?Signature=secret-token"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "choices": [{"message": {"content": [{"image": signed_url}]}}]
                    },
                    "request_id": "signed-url-test",
                },
            )
        return httpx.Response(503, content=b"unavailable")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = QwenImageEditClient(
        api_key="test-key",
        output_dir=str(tmp_path),
        base_url=TEST_BASE_URL,
        http_client=http_client,
    )
    try:
        result = client.edit_image(source_images[0], "Edit it")
    finally:
        http_client.close()
    assert result["success"] is False
    assert "HTTP 503" in result["error"]
    assert "secret-token" not in result["error"]


@requires_httpx
def test_http_client_rejects_invalid_timeout():
    from spagent.external_experts.QwenImageEdit import QwenImageEditClient

    with pytest.raises(ValueError, match="positive finite"):
        QwenImageEditClient(api_key="test-key", base_url=TEST_BASE_URL, timeout=0)


@requires_httpx
def test_http_client_rejects_invalid_constructor_strings():
    from spagent.external_experts.QwenImageEdit import QwenImageEditClient

    with pytest.raises(ValueError, match="model must be"):
        QwenImageEditClient(api_key="test-key", model=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="base_url must be"):
        QwenImageEditClient(api_key="test-key", base_url=123)  # type: ignore[arg-type]


@requires_httpx
def test_http_client_surfaces_api_error_without_key(source_images, tmp_path):
    import httpx

    from spagent.external_experts.QwenImageEdit import QwenImageEditClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert "test-secret" not in request.content.decode("utf-8")
        return httpx.Response(
            401,
            json={
                "code": "InvalidApiKey",
                "message": "invalid key test-secret",
                "requestId": "req-1",
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = QwenImageEditClient(
        api_key="test-secret",
        output_dir=str(tmp_path),
        base_url=TEST_BASE_URL,
        http_client=http_client,
    )
    try:
        result = client.edit_image(source_images[0], "Edit it")
    finally:
        http_client.close()
    assert result["success"] is False
    assert "InvalidApiKey" in result["error"]
    assert "req-1" in result["error"]
    assert "test-secret" not in result["error"]
    assert "[REDACTED]" in result["error"]


@requires_httpx
def test_http_client_rejects_non_image_download(source_images, tmp_path):
    import httpx

    from spagent.external_experts.QwenImageEdit import QwenImageEditClient

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "choices": [
                            {
                                "message": {
                                    "content": [{"image": "https://result.test/bad"}]
                                }
                            }
                        ]
                    },
                    "request_id": "req-bad",
                },
            )
        return httpx.Response(200, content=b"not an image")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = QwenImageEditClient(
        api_key="test-key",
        output_dir=str(tmp_path),
        base_url=TEST_BASE_URL,
        http_client=http_client,
    )
    try:
        result = client.edit_image(source_images[0], "Edit it")
    finally:
        http_client.close()
    assert result["success"] is False
    assert "not a valid image" in result["error"]


@requires_httpx
def test_http_client_removes_partial_multi_image_download(source_images, tmp_path):
    import httpx

    from spagent.external_experts.QwenImageEdit import QwenImageEditClient

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "choices": [
                            {
                                "message": {
                                    "content": [
                                        {"image": "https://result.test/good.png"},
                                        {"image": "https://result.test/bad.png"},
                                    ]
                                }
                            }
                        ]
                    },
                    "request_id": "req-partial",
                },
            )
        if str(request.url).endswith("good.png"):
            return httpx.Response(200, content=_png_bytes())
        return httpx.Response(200, content=b"not an image")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = QwenImageEditClient(
        api_key="test-key",
        output_dir=str(tmp_path),
        base_url=TEST_BASE_URL,
        http_client=http_client,
    )
    try:
        result = client.edit_image(source_images[0], "Edit it", n=2)
    finally:
        http_client.close()
    assert result["success"] is False
    assert list(tmp_path.glob("qwen_image_edit_req-partial_*.png")) == []


@pytest.mark.skipif(
    os.environ.get("QWEN_IMAGE_EDIT_REAL_TEST") != "1",
    reason="Set QWEN_IMAGE_EDIT_REAL_TEST=1 to call the DashScope API.",
)
def test_optional_real_dashscope_edit(source_images, tmp_path):
    api_input = tmp_path / "api_input.png"
    with Image.open(source_images[0]) as source:
        source.resize((512, 512)).save(api_input)
    tool = QwenImageEditTool(use_mock=False, output_dir=str(tmp_path))
    result = tool.call(
        image_path=str(api_input),
        prompt="Add a small blue circle in the center and preserve everything else.",
        size="512*512",
        n=1,
        seed=7,
    )
    assert result["success"] is True, result
    with Image.open(result["output_path"]) as output:
        assert output.format == "PNG"
