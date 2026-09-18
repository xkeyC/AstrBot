"""Expose an AstrBot ToolSet to Codex as dynamic tools and run Codex tool calls."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)

from astrbot.core import logger
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.permission_rules import EVENT_EXTRA_KEY as POLICY_EXTRA_KEY
from astrbot.core.permission_rules import PermissionPolicy, tool_mcp_server

from .constants import CODEX_TOOL_NAMESPACE

_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_MAX_TOOL_NAME = 128
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

JsonObject = dict[str, Any]


def _codex_name(name: str, taken: set[str]) -> str:
    base = _INVALID_NAME_CHARS.sub("_", name) or "tool"
    # "mcp" and "mcp__*" are reserved by Codex.
    if base == "mcp" or base.startswith("mcp__"):
        base = f"ext_{base}"
    base = base[:_MAX_TOOL_NAME]
    candidate, n = base, 1
    while candidate in taken:
        n += 1
        suffix = f"_{n}"
        candidate = base[: _MAX_TOOL_NAME - len(suffix)] + suffix
    taken.add(candidate)
    return candidate


def _input_schema(tool: FunctionTool) -> JsonObject:
    params = tool.parameters
    if not isinstance(params, dict) or not params:
        return dict(_EMPTY_SCHEMA)
    schema = dict(params)
    schema.setdefault("type", "object")
    if schema.get("type") == "object":
        schema.setdefault("properties", {})
    return schema


class CodexToolBridge:
    """Name mapping between an AstrBot ToolSet and one Codex thread."""

    def __init__(self, tool_set: ToolSet | None, *, defer: bool = False) -> None:
        self.defer = defer
        self.tools: dict[str, FunctionTool] = {}
        self.specs: list[JsonObject] = []
        taken: set[str] = set()
        for tool in sorted((tool_set.tools if tool_set else []), key=lambda t: t.name):
            if not getattr(tool, "active", True):
                continue
            name = _codex_name(tool.name, taken)
            self.tools[name] = tool
            spec = {
                "type": "function",
                "name": name,
                "description": (tool.description or tool.name)[:4000],
                "inputSchema": _input_schema(tool),
            }
            if defer:
                # Deferred tools never enter the prompt prefix; code mode
                # discovers them through ALL_TOOLS.
                spec["deferLoading"] = True
            self.specs.append(spec)
        self.fingerprint = hashlib.sha256(
            json.dumps(self.specs, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]

    def dynamic_tools(self) -> list[JsonObject]:
        if not self.specs:
            return []
        return [
            {
                "type": "namespace",
                "name": CODEX_TOOL_NAMESPACE,
                "description": "AstrBot plugin tools for the current chat session.",
                "tools": self.specs,
            }
        ]

    async def call(
        self,
        params: JsonObject,
        run_context: ContextWrapper,
        agent_hooks: BaseAgentRunHooks,
    ) -> JsonObject:
        """Answer an ``item/tool/call`` request."""
        from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor

        name = params.get("tool", "")
        raw_args = params.get("arguments")
        args: JsonObject = raw_args if isinstance(raw_args, dict) else {}
        tool = self.tools.get(name)
        if tool is None or params.get("namespace") not in (None, CODEX_TOOL_NAMESPACE):
            return _text_result(
                f"error: tool {params.get('namespace')}.{name} is not available.",
                success=False,
            )

        if tool.handler and tool.parameters and tool.parameters.get("properties"):
            expected = set(tool.parameters["properties"].keys())
            ignored = set(args) - expected
            if ignored:
                logger.warning(
                    "Codex tool %s: ignoring unexpected args %s", name, ignored
                )
            args = {k: v for k, v in args.items() if k in expected}

        event = getattr(getattr(run_context, "context", None), "event", None)
        get_extra = getattr(event, "get_extra", None)
        policy = get_extra(POLICY_EXTRA_KEY) if callable(get_extra) else None
        if isinstance(policy, PermissionPolicy) and not policy.allows_tool(
            tool.name, tool_mcp_server(tool)
        ):
            logger.info("Codex tool %s denied by rule %r", tool.name, policy.rule_name)
            return _text_result(
                f"error: permission denied — this user may not use {tool.name}.",
                success=False,
            )

        logger.info("Codex -> AstrBot tool %s(%s)", tool.name, args)
        try:
            await agent_hooks.on_tool_start(run_context, tool, args)
        except Exception as e:  # noqa: BLE001
            logger.error("Error in on_tool_start hook: %s", e, exc_info=True)

        items: list[JsonObject] = []
        final: CallToolResult | None = None
        success = True
        try:
            async for resp in FunctionToolExecutor.execute(
                tool=tool, run_context=run_context, **args
            ):
                if isinstance(resp, CallToolResult):
                    final = resp
                    items.extend(_content_items(resp))
                    if resp.isError:
                        success = False
                elif resp is None:
                    items.append(
                        {
                            "type": "inputText",
                            "text": "The tool has no return value, or has sent "
                            "the result directly to the user.",
                        }
                    )
                else:
                    items.append({"type": "inputText", "text": str(resp)})
        except Exception as e:  # noqa: BLE001
            logger.warning("Codex tool %s failed: %s", name, e, exc_info=True)
            items = [{"type": "inputText", "text": f"error: {e!s}"}]
            success = False
        finally:
            try:
                await agent_hooks.on_tool_end(run_context, tool, args, final)
            except Exception as e:  # noqa: BLE001
                logger.error("Error in on_tool_end hook: %s", e, exc_info=True)

        if not items:
            items = [{"type": "inputText", "text": "The tool returned no content."}]
        return {"contentItems": items, "success": success}


def _text_result(text: str, *, success: bool) -> JsonObject:
    return {"contentItems": [{"type": "inputText", "text": text}], "success": success}


def _content_items(result: CallToolResult) -> list[JsonObject]:
    items: list[JsonObject] = []
    for content in result.content or []:
        if isinstance(content, TextContent):
            items.append({"type": "inputText", "text": content.text})
        elif isinstance(content, ImageContent):
            mime = content.mimeType or "image/png"
            items.append(
                {"type": "inputImage", "imageUrl": f"data:{mime};base64,{content.data}"}
            )
        elif isinstance(content, EmbeddedResource):
            res = content.resource
            if isinstance(res, TextResourceContents):
                items.append({"type": "inputText", "text": res.text})
            elif (
                isinstance(res, BlobResourceContents)
                and res.mimeType
                and res.mimeType.startswith("image/")
            ):
                items.append(
                    {
                        "type": "inputImage",
                        "imageUrl": f"data:{res.mimeType};base64,{res.blob}",
                    }
                )
            else:
                items.append(
                    {"type": "inputText", "text": "[unsupported embedded resource]"}
                )
        else:
            items.append(
                {"type": "inputText", "text": f"[unsupported content: {content.type}]"}
            )
    return items
