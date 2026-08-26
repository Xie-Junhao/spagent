# How to Add New Tools to SPAgent

This tutorial explains how to create and integrate new tools into the SPAgent system. SPAgent uses external expert tools to extend its capabilities for spatial intelligence tasks such as depth estimation, object detection, segmentation, and 3D reconstruction.

## Table of Contents

1. [Overview](#overview)
2. [Tool Architecture](#tool-architecture)
3. [Step-by-Step: Creating a New Tool](#step-by-step-creating-a-new-tool)
4. [Parameter Schema Format](#parameter-schema-format)
5. [Return Format](#return-format)
6. [Registering Tools with SPAgent](#registering-tools-with-spagent)
7. [Integration with Training (Plugin)](#integration-with-training-plugin)
8. [Optional: Server/Client Architecture](#optional-serverclient-architecture)
9. [Best Practices](#best-practices)
10. [Complete Example](#complete-example)

---

## Overview

SPAgent tools follow a simple interface:

- **Tool base class** (`spagent.core.tool.Tool`): Abstract base with `name`, `description`, `parameters`, and `call()`
- **Output contract** (`spagent.core.tool_result`): `ToolResult` envelope with a category-specific typed payload
- **ToolRegistry**: Manages available tools; the agent uses it to look up and execute tools by name
- **Tool call format**: The model emits `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` in its response; SPAgent parses these and executes the corresponding tools

When you add a new tool, you need to:

1. Subclass `Tool` and implement `parameters` and `call()`
2. Put the tool class in `spagent/tools/` (or your own module)
3. Register it with SPAgent (or the training scheduler) when creating the agent

---

## Tool Architecture

```
spagent/
├── core/
│   └── tool.py          # Tool base class and ToolRegistry
├── tools/
│   ├── __init__.py      # Exports all tools
│   ├── depth_tool.py
│   ├── segmentation_tool.py
│   ├── yoloe_tool.py
│   └── ...              # Your new tool goes here
```

The `Tool` base class requires:

| Member | Type | Description |
|--------|------|-------------|
| `name` | str | Unique tool identifier (used in `<tool_call>`) |
| `description` | str | Human-readable description for the model |
| `parameters` | property → dict | JSON Schema for parameters (OpenAI function format) |
| `call(**kwargs)` | method | Execute the tool and return a `ToolResult` |

---

## Step-by-Step: Creating a New Tool

### Step 1: Create a new file in `spagent/tools/`

Create `spagent/tools/my_custom_tool.py`:

```python
"""
My Custom Tool

Description of what this tool does.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from spagent.core.tool import Tool
from spagent.core.tool_result import DetectionPayload, ToolResult

logger = logging.getLogger(__name__)


class MyCustomTool(Tool):
    """Tool for [your functionality]"""

    def __init__(self, use_mock: bool = True, server_url: str = "http://localhost:8000"):
        super().__init__(
            name="my_custom_tool",
            description="Clear description of what this tool does and when to use it."
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize client (mock or real)"""
        if self.use_mock:
            self._client = MockMyService()
        else:
            self._client = RealMyClient(server_url=self.server_url)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image."
                },
                "option": {
                    "type": "string",
                    "enum": ["mode_a", "mode_b"],
                    "description": "Processing mode.",
                    "default": "mode_a"
                }
            },
            "required": ["image_path"]
        }

    def call(
        self,
        image_path: str,
        option: str = "mode_a"
    ) -> ToolResult:
        try:
            if not Path(image_path).exists():
                return ToolResult.fail(
                    f"Image not found: {image_path}",
                    category="detection",
                )

            result = self._client.process(image_path, option=option)

            if result and result.get("success"):
                payload = DetectionPayload(
                    boxes=result["boxes"],
                    labels=result["labels"],
                    box_format=result.get("box_format", "xyxy_pixel"),
                    confidence=result.get("confidence"),
                )
                return ToolResult(
                    success=True,
                    payload=payload,
                    description=result.get("description", ""),
                    output_path=result.get("output_path"),
                )

            error = result.get("error", "Unknown error") if result else "No result"
            return ToolResult.fail(error, category="detection")
        except Exception as e:
            logger.error(f"MyCustomTool error: {e}")
            return ToolResult.fail(str(e), category="detection")
```

### Step 2: Export the tool in `spagent/tools/__init__.py`

Add your tool to the module exports:

```python
from .my_custom_tool import MyCustomTool

__all__ = [
    # ... existing tools ...
    'MyCustomTool'
]
```

---

## Parameter Schema Format

Use **JSON Schema** in OpenAI function-calling format:

```python
{
    "type": "object",
    "properties": {
        "param_name": {
            "type": "string" | "number" | "boolean" | "array" | "object",
            "description": "What this parameter does",
            "enum": ["a", "b"],           # optional: restrict to values
            "default": "a",               # optional: default value
            "minimum": 0, "maximum": 1,   # optional: for numbers
            "items": {"type": "string"}   # for array elements
        }
    },
    "required": ["param_name"]   # list of required parameters
}
```

Common types:

- `string`: paths, text, IDs
- `number`: thresholds, angles, counts
- `boolean`: flags
- `array`: lists of paths, class names, coordinates
- `object`: nested structures

The model uses these schemas to decide when and how to call your tool, so keep descriptions clear and concise.

---

## Return Format

`call()` must return a `ToolResult`. Successful results include a typed payload
for one of the supported categories; failures use `ToolResult.fail(...)`.
See the normative [Tool Output Contract](Tool/TOOL_CONFIGURATIONS.md) for
required payload carriers by category.

```python
payload = DetectionPayload(
    boxes=boxes,
    labels=labels,
    box_format="xyxy_pixel",
    confidence=scores,
)
return ToolResult(
    success=True,
    payload=payload,
    description="Detected objects.",
    output_path="/path/to/annotated.png",
)

# Failure:
return ToolResult.fail("Image file not found", category="detection")
```

---

## Registering Tools with SPAgent

### Option A: Pass tools at initialization

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import DepthEstimationTool, MyCustomTool

model = GPTModel(model_name="gpt-4o-mini")
tools = [
    DepthEstimationTool(use_mock=True),
    MyCustomTool(use_mock=False, server_url="http://localhost:8000"),
]

agent = SPAgent(model=model, tools=tools)
```

### Option B: Add tools after initialization

```python
agent = SPAgent(model=model)
agent.add_tool(MyCustomTool())
agent.remove_tool("my_custom_tool")  # if needed
```

### Verify registration

```python
print(agent.list_tools())  # ['depth_estimation_tool', 'my_custom_tool', ...]
```

---

## Integration with Training (Plugin)

When using SPAgent with the training plugin (`SPAgentToolCallingScheduler`), tools can be registered in two ways.

### 1. Auto-registration

The scheduler tries to auto-register a fixed set of tools in `plugin/plugin.py`:

```python
tool_classes = [
    ('DepthEstimationTool', 'depth_estimation_tool'),
    ('SegmentationTool', 'segmentation_tool'),
    ('ObjectDetectionTool', 'object_detection_tool'),
    ('Pi3Tool', 'pi3_tool'),
]
```

To include your tool in auto-registration, add it to this list and ensure it is exported from `spagent.tools`.

### 2. Manual registration

Register your tool explicitly when configuring the scheduler:

```python
from spagent.tools import MyCustomTool

scheduler = SPAgentToolCallingScheduler(max_turns=3)
scheduler.register_tool(MyCustomTool(use_mock=False, server_url="http://..."))
# or
scheduler.register_tools([MyCustomTool(), OtherTool()])
```

---

## Optional: Server/Client Architecture

Many SPAgent tools use a server/client setup:

- **Server**: Runs the heavy model (e.g., depth, segmentation) and exposes an HTTP API
- **Client**: Called from the tool’s `call()` to send requests and parse responses
- **Mock**: Lightweight implementation for testing without the real server

Example layout:

```
spagent/external_experts/
└── MyExpert/
    ├── my_server.py       # Flask/FastAPI server
    ├── my_client.py       # Client for the tool
    └── mock_my_service.py # Mock for testing
```

The tool switches between mock and real client based on `use_mock`:

```python
def _init_client(self):
    if self.use_mock:
        from external_experts.MyExpert.mock_my_service import MockMyService
        self._client = MockMyService()
    else:
        from external_experts.MyExpert.my_client import MyClient
        self._client = MyClient(server_url=self.server_url)
```

---

## Best Practices

1. **Naming**: Use snake_case for tool names (e.g. `my_custom_tool`).
2. **Descriptions**: Write clear, model-friendly descriptions so the LLM knows when to use the tool.
3. **Parameters**: Prefer explicit `required` and `enum` where it helps.
4. **Validation**: Check inputs (e.g. file existence) before calling external services.
5. **Errors**: Always return `success: False` with a clear `error` message on failure.
6. **Logging**: Use `logger.info` / `logger.error` for debugging and monitoring.
7. **Mock mode**: Provide a mock implementation for development and CI.
8. **Output paths**: If the tool produces images/files, return paths so the agent can pass them to the model.

---

## Complete Example

See the minimal `ToolResult` example and required CI commands in
[Contributing Tools](Tool/CONTRIBUTING_TOOLS.md). For a complete implementation,
use `spagent/tools/face_detection_tool.py` as the lightweight reference tool.

---

## Related Documentation

- [Contributing Tools](Tool/CONTRIBUTING_TOOLS.md) – Minimal contract example, checklist, and CI commands
- [Tool Output Contract](Tool/TOOL_CONFIGURATIONS.md) – Detailed envelope, payload, and rendering specification
- [Tool Usage Guide](Tool/TOOL_USING.md) – Overview of built-in tools and how to run their servers
- [Advanced Examples](Examples/ADVANCED_EXAMPLES.md) – More usage patterns
- [Tool Definition Examples](../../spagent/tool_definition_examples.py) – Additional custom tool examples
