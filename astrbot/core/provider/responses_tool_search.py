import json
import math
import re
from collections import Counter
from typing import Any

from astrbot.core.agent.tool import FunctionTool

TOOL_SEARCH_DEFAULT_LIMIT = 8
TOOL_SEARCH_HISTORY_MARKER_KEY = "astrbot_internal_tool_type"
TOOL_SEARCH_HISTORY_MARKER_VALUE = "responses_tool_search"
TOOL_SEARCH_NAME = "tool_search"


def _tokenize(text: str) -> list[str]:
    """Tokenize Latin words, tool identifiers, and individual CJK characters.

    Args:
        text: Searchable tool metadata or a model query.

    Returns:
        Normalized BM25 tokens.
    """
    return re.findall(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]", text.lower())


def _responses_function_schema(tool: FunctionTool, *, deferred: bool) -> dict[str, Any]:
    """Build a native Responses API function declaration.

    Args:
        tool: AstrBot function tool.
        deferred: Whether the declaration is loaded through tool search.

    Returns:
        Responses API function schema.
    """
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
    """Create the client-executed Responses API tool search tool.

    Args:
        deferred_tools: Non-core tools hidden from the initial model request.

    Returns:
        The executable AstrBot tool and its native Responses API declaration.
    """
    documents = [
        _tokenize(
            " ".join(
                (
                    tool.name,
                    tool.name.replace("_", " "),
                    tool.description,
                    json.dumps(tool.parameters, ensure_ascii=False, default=str),
                )
            )
        )
        for tool in deferred_tools
    ]
    term_frequencies = [Counter(document) for document in documents]
    document_frequencies: Counter[str] = Counter()
    for frequencies in term_frequencies:
        document_frequencies.update(frequencies.keys())
    average_document_length = sum(map(len, documents)) / len(documents)

    async def search_tools(
        _event: Any,
        query: str,
        limit: int = TOOL_SEARCH_DEFAULT_LIMIT,
    ) -> str:
        """Search deferred tool metadata and return loadable declarations.

        Args:
            _event: Unused AstrBot event injected by the local tool executor.
            query: Model-provided tool search query.
            limit: Maximum number of declarations to return.

        Returns:
            JSON-encoded tool_search_output payload body.
        """
        query_tokens = _tokenize(str(query))
        if not query_tokens:
            return json.dumps({"tools": []}, ensure_ascii=False)

        try:
            result_limit = int(limit)
        except (TypeError, ValueError):
            result_limit = TOOL_SEARCH_DEFAULT_LIMIT
        if result_limit <= 0:
            return json.dumps({"tools": []}, ensure_ascii=False)

        query_frequencies = Counter(query_tokens)
        scores = []
        document_count = len(documents)
        for index, frequencies in enumerate(term_frequencies):
            document_length = len(documents[index])
            score = 0.0
            for token, query_frequency in query_frequencies.items():
                term_frequency = frequencies.get(token, 0)
                if term_frequency == 0:
                    continue
                document_frequency = document_frequencies[token]
                inverse_document_frequency = math.log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                normalization = term_frequency + 1.5 * (
                    1 - 0.75 + 0.75 * document_length / average_document_length
                )
                score += (
                    inverse_document_frequency
                    * term_frequency
                    * 2.5
                    / normalization
                    * min(query_frequency, 2)
                )
            scores.append(score)
        ranked_indices = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda index: (-scores[index], index),
        )[: min(result_limit, len(deferred_tools))]
        result = [
            _responses_function_schema(deferred_tools[index], deferred=True)
            for index in ranked_indices
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
    """Convert an AstrBot function tool to a directly visible Responses tool.

    Args:
        tool: Tool to expose directly.

    Returns:
        Native Responses API function tool schema.
    """
    return _responses_function_schema(tool, deferred=False)
