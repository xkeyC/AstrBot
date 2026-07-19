import json
from typing import Any

from astrbot.core.agent.tool import FunctionTool
from astrbot.core.agent.tool_registry import (
    TOOL_SEARCH_DEFAULT_LIMIT,
    TOOL_SEARCH_NAME,
    ToolSearchIndex,
)

TOOL_SEARCH_HISTORY_MARKER_KEY = "astrbot_internal_tool_type"
TOOL_SEARCH_HISTORY_MARKER_VALUE = "responses_tool_search"


def _responses_function_schema(tool: FunctionTool, *, deferred: bool) -> dict[str, Any]:
    """Build a native Responses API function declaration."""
    schema = {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters or {"type": "object", "properties": {}},
        "strict": False,
    }
    if deferred:
        schema["defer_loading"] = True
    return schema


def create_responses_tool_search(
    deferred_tools: list[FunctionTool],
) -> tuple[FunctionTool, dict[str, Any]]:
    """Create the legacy client-executed Responses API tool search tool."""
    index = ToolSearchIndex(deferred_tools)

    async def search_tools(
        _event: Any,
        query: str,
        limit: int = TOOL_SEARCH_DEFAULT_LIMIT,
    ) -> str:
        try:
            result_limit = int(limit)
        except (TypeError, ValueError):
            result_limit = TOOL_SEARCH_DEFAULT_LIMIT
        if result_limit <= 0:
            return json.dumps({"tools": []}, ensure_ascii=False)

        matches = index.search(str(query))[: min(result_limit, len(deferred_tools))]
        result = [
            _responses_function_schema(match.tool, deferred=True) for match in matches
        ]
        return json.dumps({"tools": result}, ensure_ascii=False)

    function_tool = FunctionTool(
        name=TOOL_SEARCH_NAME,
        description="Search deferred tools and expose matching tools to the model.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for deferred tools.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of tools to return. "
                        f"Defaults to {TOOL_SEARCH_DEFAULT_LIMIT}."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=search_tools,
    )
    request_schema = {
        "type": TOOL_SEARCH_NAME,
        "execution": "client",
        "description": (
            "# Tool discovery\n\n"
            "Searches over deferred AstrBot tool metadata with BM25 and exposes "
            "matching tools for the next model call. Some plugin, MCP, and other "
            "non-core tools are not provided upfront; use `tool_search` when one "
            "is needed."
        ),
        "parameters": function_tool.parameters,
    }
    return function_tool, request_schema


def to_responses_function_schema(tool: FunctionTool) -> dict[str, Any]:
    """Convert an AstrBot function tool to a directly visible Responses tool."""
    return _responses_function_schema(tool, deferred=False)
