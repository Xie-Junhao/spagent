# Contributing a new tool — what the auto-tests expect

Every tool must follow the standardized `ToolResult` contract. See
[Tool Output Contract](TOOL_CONFIGURATIONS.md) for the detailed envelope,
category payload, and rendering specification. This guide only covers the
minimum contribution steps and automated checks.

## Checklist for a new tool

1. Implement the tool in `spagent/tools/<name>_tool.py`, returning a
   `ToolResult` with a typed payload on success and `ToolResult.fail(...)`
   on failure. Never raise on bad input; validate image paths in mock mode too.
2. Register it in `spagent/tools/catalog.py` (`TOOL_CATALOG`, and
   `DEFAULT_SERVER_URLS` if server-backed). Unregistered tool files fail CI.
3. Provide a mock client so the tool runs with `use_mock=True` and zero
   VRAM, format-faithful to the real backend (see
   `external_experts/*/mock_*_service.py` for examples).
4. Add a minimal mock-call row to `CALL_KW` in `test/tool_ci_report.py`.
   A missing row fails the gate with "no CALL_KW entry".
5. Document the tool: add a row to the tool table in
   `docs/Tool/TOOL_USING.md` (and `docs/Tool/EXTERNAL_EXPERTS.md` plus its
   port table if the tool is server-backed). Lane 1's `docs` check fails
   undocumented tools.
6. Run the lanes locally before opening the PR (below).

Minimal detection example:

```python
from spagent.core.tool_result import DetectionPayload, ToolResult

payload = DetectionPayload(
    boxes=boxes,
    labels=labels,
    box_format="xyxy_pixel",
    confidence=scores,
    image_width=width,
    image_height=height,
)
return ToolResult(
    success=True,
    payload=payload,
    description=f"Detected {len(boxes)} objects.",
    output_path=annotated_path,
)

# Failure path:
return ToolResult.fail("Image not found", category="detection")
```

## Lane 1 — no-compute contract gate (`tool-ci.yml`, every PR)

```bash
python test/tool_ci_report.py                      # full catalog report
python test/tool_ci_report.py --tools <your-key>   # gate just your tool
```

Per tool it checks: **build** (mock), **schema** (`parameters` has
type/properties), **call** (mock succeeds), **toolresult** (returns
`ToolResult`, not a plain dict), **contract** (`validate_payload`),
**render** (non-empty projection under `default` and `all` presets),
**boxes** (pixel projection is non-degenerate: `x2>x1`, `y2>y1`), and
**failpath** (missing input image → `success=False`, no exception).

Only tools changed by the PR gate the merge; the full catalog is always
reported. Tools whose imports need unavailable heavy deps show as
`⏭ dep` and never block (mirroring `test/verify_all_tools.py`).

## Lane 2 — with-compute real-backend smoke (`tool-real-smoke.yml`, opt-in)

```bash
# start your backend first (see TOOL_USING.md), then:
python test/tool_real_smoke.py --tool <your-key> --url <your-key>=http://localhost:<port>
```

This lane **measures the artifact, not the label** — the lesson of the
PR #230 verification campaign:

- backend reachability is probed out-of-band first, because several tools
  convert a dead backend into a `success=True` zero-finding result;
- detection: pixel-projected boxes must be in-bounds and non-degenerate,
  confidence aligned with boxes (0 detections is a *valid* finding);
- segmentation: mask dims must equal the image dims;
- depth: `shape` must equal the image dims;
- 3D reconstruction: `points_count` must be a real positive count, and a
  `cached: true` result fails (stale-cache placeholder, not a measurement);
- optical flow: raw `.npy` must be `(H, W, 2)`; with `--flow-shift N` a
  synthetic N-px shift must be recovered;
- OCR: `text` must exist (empty text is a valid finding);
- generation: the output file must exist and decode.

In CI it runs via `workflow_dispatch` on a self-hosted GPU runner with the
backends already up; running it locally is equivalent and is the expected
path while no such runner is registered.

## Lane 3 — full-agent e2e (`tool-agent-e2e.yml`, opt-in)

One real `SPAgent.step()` episode per tool: a live VLM receives your tool's
schema, must *choose* to invoke it against its live backend, and must produce
a grounded final answer. This verifies the whole chain — tool description →
model tool call → backend → `ToolResult` → `render()` projection → answer —
including whether your tool *description* is good enough for a model to
actually pick the tool up.

```bash
# API keys via env vars only (never CLI args). Pick any one provider:
GEMINI_API_KEY=...    python test/tool_agent_e2e.py --tool <key> --provider gemini
OPENAI_API_KEY=...    python test/tool_agent_e2e.py --tool <key> --provider openai
ANTHROPIC_API_KEY=... python test/tool_agent_e2e.py --tool <key> --provider anthropic
# or a local OpenAI-compatible server (vLLM, llama.cpp):
python test/tool_agent_e2e.py --tool <key> --provider local \
    --local-base-url http://localhost:8000/v1 --model <served-model>
```

Server-backed tools take the same `--url key=http://...` override as lane 2,
and reuse its dead-backend reachability guard. Episode prompts are
category-appropriate defaults (override with `--prompt`). Pass criteria: the
agent loop completes, the model invoked *your* tool, at least one invocation
succeeded, and the final answer is non-empty. A failure of the form "the
model never called <tool>" usually means the tool description needs work —
that is a real finding, not a flaky test.
