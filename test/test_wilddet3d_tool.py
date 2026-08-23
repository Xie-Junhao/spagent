"""
Tests for WildDet3DTool.

Mock tests run without a WildDet3D server. Set WILDDET3D_REAL_TEST=1 to run the
optional live-server smoke test.
"""

import os
import sys
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "spagent"))

from spagent.tools import WildDet3DTool
from core.tool_result import ToolResult, validate_payload


@pytest.fixture
def sample_image_path(tmp_path):
    path = tmp_path / "wilddet3d_sample.jpg"
    Image.new("RGB", (160, 120), color=(80, 120, 170)).save(path, format="JPEG")
    return str(path)


def test_wilddet3d_tool_is_exported():
    assert WildDet3DTool is not None


def test_wilddet3d_schema_contains_prompt_inputs():
    tool = WildDet3DTool(use_mock=True)
    params = tool.parameters

    assert "image_path" in params["required"]
    assert "text_prompt" in params["properties"]
    assert "boxes" in params["properties"]
    assert "points" in params["properties"]
    assert "anyOf" in params


def test_wilddet3d_mock_text_prompt(tmp_path, sample_image_path):
    tool = WildDet3DTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(
        image_path=sample_image_path,
        text_prompt="chair",
        score_threshold=0.3,
        save_visualization=True,
    )

    assert result["success"] is True
    assert isinstance(result, ToolResult)
    assert validate_payload(result, "detection")[0]
    assert result["boxes_2d"]
    assert result["boxes_3d"]
    assert result["scores"]
    assert result["class_names"] == ["chair"]
    assert result["output_path"] is not None
    assert os.path.exists(result["output_path"])
    assert result["depth_path"] is not None
    assert os.path.exists(result["depth_path"])


def test_wilddet3d_mock_box_prompt(tmp_path, sample_image_path):
    tool = WildDet3DTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(
        image_path=sample_image_path,
        boxes=[[10, 15, 80, 95]],
        save_visualization=True,
    )

    assert result["success"] is True
    assert result["boxes_2d"] == [[10.0, 15.0, 80.0, 95.0]]
    assert len(result["boxes_3d"][0]) == 7
    assert result["class_names"] == ["object"]


def test_wilddet3d_mock_point_prompt(tmp_path, sample_image_path):
    tool = WildDet3DTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(
        image_path=sample_image_path,
        points=[[60, 50, 1]],
        save_visualization=False,
    )

    assert result["success"] is True
    assert result["boxes_3d"]
    assert result["output_path"] is None


def test_wilddet3d_rejects_missing_image(tmp_path):
    tool = WildDet3DTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(image_path=str(tmp_path / "missing.jpg"), text_prompt="chair")

    assert result["success"] is False
    assert "Image file not found" in result["error"]


def test_wilddet3d_rejects_empty_prompt(sample_image_path):
    tool = WildDet3DTool(use_mock=True)

    result = tool.call(image_path=sample_image_path, text_prompt=" ")

    assert result["success"] is False
    assert "Provide at least one prompt" in result["error"]


def test_wilddet3d_server_filters_on_combined_score_and_keeps_3d_scores_aligned():
    from spagent.external_experts.WildDet3D.wilddet3d_server import _filter_by_score

    filtered = _filter_by_score(
        boxes_2d=np.array([[1, 2, 10, 12], [3, 4, 20, 24]], dtype=np.float32),
        boxes_3d=np.arange(20, dtype=np.float32).reshape(2, 10),
        scores=np.array([0.2, 0.8], dtype=np.float32),
        scores_2d=np.array([0.9, 0.9], dtype=np.float32),
        scores_3d=np.array([0.7, 0.6], dtype=np.float32),
        class_ids=np.array([0, 1]),
        threshold=0.3,
    )

    boxes_2d, boxes_3d, scores, scores_2d, scores_3d, class_ids = filtered
    assert boxes_2d.shape == (1, 4)
    assert boxes_3d.shape == (1, 10)
    assert scores.tolist() == pytest.approx([0.8])
    assert scores_2d.tolist() == pytest.approx([0.9])
    assert scores_3d.tolist() == pytest.approx([0.6])
    assert class_ids.tolist() == [1]


def test_wilddet3d_server_maps_class_ids_to_detection_labels():
    from spagent.external_experts.WildDet3D.wilddet3d_server import _detection_class_names

    labels = _detection_class_names(["chair", "table"], np.array([1, 0, 1]), 3)
    assert labels == ["table", "chair", "table"]


@pytest.mark.skipif(
    os.environ.get("WILDDET3D_REAL_TEST") != "1",
    reason="Set WILDDET3D_REAL_TEST=1 to run against a live WildDet3D server.",
)
def test_wilddet3d_real_service(sample_image_path):
    server_url = os.environ.get("WILDDET3D_SERVER_URL", "http://127.0.0.1:20027")
    tool = WildDet3DTool(use_mock=False, server_url=server_url)

    result = tool.call(
        image_path=sample_image_path,
        text_prompt="chair",
        score_threshold=0.3,
        save_visualization=True,
    )

    assert result["success"] is True
    assert "boxes_3d" in result
