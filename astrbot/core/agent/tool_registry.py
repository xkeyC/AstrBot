import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from astrbot.core.agent.tool import FunctionTool, ToolSet

TOOL_SEARCH_DEFAULT_LIMIT = 8
TOOL_SEARCH_MAX_LIMIT = 20
TOOL_SEARCH_NAME = "tool_search"
TOOL_INVOKE_NAME = "tool_invoke"
TOOL_REGISTRY_RESERVED_NAMES = frozenset({TOOL_SEARCH_NAME, TOOL_INVOKE_NAME})

_BM25_FIELD_WEIGHTS = (4.0, 1.0, 0.35)
_NAME_SEQUENCE_BONUS = 1.5
_NAMESPACE_PREFIX_BONUS = 2.0
_KEYWORD_FIELD_WEIGHTS = (3.0, 2.0, 1.0)
_ENGLISH_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize Latin words, identifiers, and individual CJK characters."""
    return re.findall(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]", text.lower())


def _schema_search_text(schema: Any) -> str:
    """Extract meaningful names and descriptions from a JSON schema."""
    parts: list[str] = []

    def collect(node: Any) -> None:
        if not isinstance(node, dict):
            return

        description = node.get("description")
        if isinstance(description, str):
            parts.append(description)

        properties = node.get("properties")
        if isinstance(properties, dict):
            for name, child in properties.items():
                property_name = str(name)
                parts.extend((property_name, property_name.replace("_", " ")))
                collect(child)

        collect(node.get("items"))
        for keyword in ("anyOf", "oneOf", "allOf"):
            variants = node.get(keyword)
            if isinstance(variants, list):
                for variant in variants:
                    collect(variant)

    collect(schema)
    return " ".join(parts)


def _longest_contiguous_match(left: list[str], right: list[str]) -> int:
    """Return the longest shared contiguous token sequence length."""
    longest = 0
    for left_start, left_token in enumerate(left):
        for right_start, right_token in enumerate(right):
            if left_token != right_token:
                continue
            length = 1
            while (
                left_start + length < len(left)
                and right_start + length < len(right)
                and left[left_start + length] == right[right_start + length]
            ):
                length += 1
            longest = max(longest, length)
    return longest


@dataclass(frozen=True, slots=True)
class ToolSearchMatch:
    """A normalized tool registry search result."""

    tool: FunctionTool
    score: float
    matched_terms: tuple[str, ...]


class ToolSearchIndex:
    """In-memory weighted BM25 index over an effective AstrBot tool set."""

    def __init__(self, tools: list[FunctionTool]) -> None:
        self.tools = list(tools)
        self._field_texts = (
            [f"{tool.name} {tool.name.replace('_', ' ')}" for tool in self.tools],
            [tool.description for tool in self.tools],
            [_schema_search_text(tool.parameters) for tool in self.tools],
        )
        self._field_documents = tuple(
            [_tokenize(text) for text in texts] for texts in self._field_texts
        )
        self._field_statistics = []
        for documents in self._field_documents:
            term_frequencies = [Counter(document) for document in documents]
            document_frequencies: Counter[str] = Counter()
            for frequencies in term_frequencies:
                document_frequencies.update(frequencies.keys())
            average_length = (
                sum(map(len, documents)) / len(documents) if documents else 1.0
            ) or 1.0
            self._field_statistics.append(
                (term_frequencies, document_frequencies, average_length)
            )
        self._split_name_tokens = [
            [
                token
                for token in _tokenize(tool.name.replace("_", " "))
                if _ENGLISH_TOKEN_PATTERN.fullmatch(token)
            ]
            for tool in self.tools
        ]

    def search(
        self,
        query: str,
        keywords: list[str] | None = None,
    ) -> list[ToolSearchMatch]:
        """Search and rank all matching tools.

        Args:
            query: Natural-language capability query.
            keywords: Optional exact phrases to boost.

        Returns:
            Matches sorted by descending normalized score.
        """
        if not self.tools:
            return []

        normalized_query = str(query).strip()
        normalized_keywords = [
            str(keyword).strip().lower()
            for keyword in keywords or []
            if str(keyword).strip()
        ]
        if normalized_query == "*":
            return [
                ToolSearchMatch(tool=tool, score=1.0, matched_terms=("*",))
                for tool in self.tools
            ]

        query_tokens = _tokenize(normalized_query)
        for keyword in normalized_keywords:
            query_tokens.extend(_tokenize(keyword))
        if not query_tokens:
            return []

        query_frequencies = Counter(query_tokens)
        query_english_tokens = [
            token for token in query_tokens if _ENGLISH_TOKEN_PATTERN.fullmatch(token)
        ]
        document_count = len(self.tools)
        scores: list[float] = []
        matched_terms_by_index: list[tuple[str, ...]] = []

        for index, tool in enumerate(self.tools):
            score = 0.0
            matched_terms: set[str] = set()
            for weight, documents, statistics in zip(
                _BM25_FIELD_WEIGHTS,
                self._field_documents,
                self._field_statistics,
                strict=True,
            ):
                term_frequencies, document_frequencies, average_length = statistics
                frequencies = term_frequencies[index]
                document_length = len(documents[index])
                for token, query_frequency in query_frequencies.items():
                    term_frequency = frequencies.get(token, 0)
                    if term_frequency == 0:
                        continue
                    matched_terms.add(token)
                    document_frequency = document_frequencies[token]
                    inverse_document_frequency = math.log(
                        1
                        + (document_count - document_frequency + 0.5)
                        / (document_frequency + 0.5)
                    )
                    normalization = term_frequency + 1.5 * (
                        1 - 0.75 + 0.75 * document_length / average_length
                    )
                    score += (
                        weight
                        * inverse_document_frequency
                        * term_frequency
                        * 2.5
                        / normalization
                        * min(query_frequency, 2)
                    )

            lower_field_texts = [
                field_texts[index].lower() for field_texts in self._field_texts
            ]
            for keyword in normalized_keywords:
                for weight, field_text in zip(
                    _KEYWORD_FIELD_WEIGHTS,
                    lower_field_texts,
                    strict=True,
                ):
                    if keyword in field_text:
                        score += weight
                        matched_terms.add(keyword)

            contiguous_match = _longest_contiguous_match(
                query_english_tokens,
                self._split_name_tokens[index],
            )
            if contiguous_match >= 2:
                score += _NAME_SEQUENCE_BONUS * contiguous_match
            if (
                "_" in tool.name
                and self._split_name_tokens[index]
                and self._split_name_tokens[index][0] in query_english_tokens
            ):
                score += _NAMESPACE_PREFIX_BONUS

            scores.append(score)
            matched_terms_by_index.append(tuple(sorted(matched_terms)))

        ranked_indices = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda index: (-scores[index], index),
        )
        if not ranked_indices:
            return []
        highest_score = scores[ranked_indices[0]]
        return [
            ToolSearchMatch(
                tool=self.tools[index],
                score=round(scores[index] / highest_score, 6),
                matched_terms=matched_terms_by_index[index],
            )
            for index in ranked_indices
        ]


def build_tool_prefix_index(tools: list[FunctionTool]) -> str:
    """Build a compact prompt inventory of searchable tool-name prefixes.

    Args:
        tools: Effective hidden tools available to the current request.

    Returns:
        A prompt section containing safe prefixes and compact suffix hints.
    """
    prefix_counts: Counter[str] = Counter()
    prefix_suffixes: dict[str, list[str]] = {}
    for tool in tools:
        prefix, separator, remainder = tool.name.partition("_")
        if separator and prefix:
            normalized_prefix = f"{prefix.lower()}_"
            prefix_counts[normalized_prefix] += 1
            safe_suffix = re.sub(r"[^a-zA-Z0-9_-]", "", remainder)[:64]
            if safe_suffix:
                prefix_suffixes.setdefault(normalized_prefix, []).append(safe_suffix)

    if not prefix_counts:
        return (
            "## Tool registry\n\n"
            "Hidden tools are available through `tool_search` and `tool_invoke`. "
            "No tool-name prefixes are available; use index `*` and search by "
            "capability keywords."
        )

    prefix_lines = []
    for prefix, count in sorted(prefix_counts.items()):
        suffixes = prefix_suffixes.get(prefix, [])
        displayed_suffixes = ", ".join(suffixes[:12])
        if len(suffixes) > 12:
            displayed_suffixes += f", +{len(suffixes) - 12} more"
        suffix_hint = f": {displayed_suffixes}" if displayed_suffixes else ""
        prefix_lines.append(
            f"- `{prefix}` ({count} tool{'s' if count != 1 else ''}){suffix_hint}"
        )
    prefix_block = "\n".join(prefix_lines)
    return (
        "## Tool registry\n\n"
        "Non-core tools are hidden behind `tool_search` and `tool_invoke`. "
        "Use the following complete prefix index to narrow searches precisely. "
        "Pass one of these exact values as `index`; do not guess a tool name.\n\n"
        "### Available tool prefixes\n\n"
        f"{prefix_block}\n\n"
        "Call `tool_search` with an exact index and capability keywords. Use `*` "
        "only for a cross-index fallback. Use the returned full parameter schema, "
        "then call `tool_invoke` with the exact `tool_id`. If `has_more` is true, "
        "continue from `next_offset`."
    )


def create_tool_registry_tools(
    deferred_tools: list[FunctionTool],
    context_tool_ids_provider: Callable[[], set[str]] | None = None,
) -> ToolSet:
    """Create provider-agnostic registry search and invocation meta-tools.

    Args:
        deferred_tools: Effective non-core tools hidden from provider schemas.
        context_tool_ids_provider: Returns tool IDs disclosed by search results that
            are still present in the live, uncompressed context.

    Returns:
        A stable ToolSet containing tool_search and tool_invoke.
    """
    available_indexes = {
        f"{prefix.lower()}_"
        for tool in deferred_tools
        for prefix, separator, _remainder in [tool.name.partition("_")]
        if separator and prefix
    }
    search_indexes = {
        "*": ToolSearchIndex(deferred_tools),
        **{
            prefix: ToolSearchIndex(
                [
                    tool
                    for tool in deferred_tools
                    if tool.name.lower().startswith(prefix)
                ]
            )
            for prefix in available_indexes
        },
    }
    registry_tool_ids = {tool.name for tool in deferred_tools}

    def get_context_tool_ids() -> set[str]:
        if context_tool_ids_provider is None:
            return set()
        try:
            return set(context_tool_ids_provider()).intersection(registry_tool_ids)
        except Exception:
            return set()

    async def search_tools(
        _event: Any,
        index: str,
        keywords: list[str],
        limit: int = TOOL_SEARCH_DEFAULT_LIMIT,
        offset: int = 0,
        min_score: float = 0.0,
    ) -> str:
        try:
            result_limit = max(1, min(int(limit), TOOL_SEARCH_MAX_LIMIT))
        except (TypeError, ValueError):
            result_limit = TOOL_SEARCH_DEFAULT_LIMIT
        try:
            result_offset = max(0, int(offset))
        except (TypeError, ValueError):
            result_offset = 0
        try:
            score_threshold = max(0.0, min(float(min_score), 1.0))
        except (TypeError, ValueError):
            score_threshold = 0.0

        normalized_index = str(index).strip().lower()
        raw_keywords = [keywords] if isinstance(keywords, str) else keywords or []
        normalized_keywords = [
            str(keyword).strip() for keyword in raw_keywords if str(keyword).strip()
        ]
        context_tool_ids = get_context_tool_ids()
        if normalized_index != "*" and normalized_index not in available_indexes:
            return json.dumps(
                {
                    "index": normalized_index,
                    "keywords": normalized_keywords,
                    "search_mode": "weighted_bm25",
                    "tools": [],
                    "total": 0,
                    "returned": 0,
                    "tools_in_context": len(context_tool_ids),
                    "filtered_in_context": 0,
                    "pagination_mode": (
                        "live_context_dedup"
                        if context_tool_ids_provider is not None
                        else "offset"
                    ),
                    "has_more": False,
                    "next_offset": None,
                    "error": "Unknown tool index. Use an exact value from the system prefix inventory.",
                },
                ensure_ascii=False,
            )
        query = " ".join(normalized_keywords) or "*"
        matches = search_indexes[normalized_index].search(query, normalized_keywords)
        matches_before_context_filter = len(matches)
        matches = [
            match for match in matches if match.tool.name not in context_tool_ids
        ]
        filtered_in_context = matches_before_context_filter - len(matches)
        filtered_matches = [
            match for match in matches if match.score >= score_threshold
        ]
        if matches and not filtered_matches:
            filtered_matches = matches[:1]

        page = filtered_matches[result_offset : result_offset + result_limit]
        page_end = result_offset + len(page)
        has_more = page_end < len(filtered_matches)
        next_offset = (
            0
            if has_more and context_tool_ids_provider is not None
            else page_end
            if has_more
            else None
        )
        payload = {
            "index": normalized_index,
            "keywords": normalized_keywords,
            "search_mode": "weighted_bm25",
            "tools": [
                {
                    "tool_id": match.tool.name,
                    "description": match.tool.description,
                    "parameters": match.tool.parameters,
                    "score": match.score,
                    "matched_terms": list(match.matched_terms),
                }
                for match in page
            ],
            "total": len(filtered_matches),
            "returned": len(page),
            "tools_in_context": len(context_tool_ids),
            "filtered_in_context": filtered_in_context,
            "pagination_mode": (
                "live_context_dedup"
                if context_tool_ids_provider is not None
                else "offset"
            ),
            "has_more": has_more,
            "next_offset": next_offset,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    async def invoke_guard(
        _event: Any,
        tool_id: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        del tool_id, arguments
        return "error: tool_invoke must be resolved by the AstrBot agent runner."

    search_tool = FunctionTool(
        name=TOOL_SEARCH_NAME,
        description=(
            "Search the hidden tool registry for capabilities. Results contain "
            "tool_id and the full parameter schema. If has_more is true, call "
            "again with next_offset. First select an exact index from the system "
            "prefix inventory, then provide keywords to match tool schemas. Use "
            "tool_invoke to execute a result. To enumerate all hidden tools, use "
            "index=* with empty keywords and follow pagination. Tools already "
            "disclosed in the live context are filtered until compression removes "
            "their search results. The response reports tools_in_context and "
            "filtered_in_context counts."
        ),
        parameters={
            "type": "object",
            "properties": {
                "index": {
                    "type": "string",
                    "description": (
                        "Exact namespace from the system prefix index, for example "
                        "qq_. Use * only as a cross-index fallback. Tool IDs are "
                        "filtered with startswith(index)."
                    ),
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Capability terms matched against tool names, descriptions, "
                        "and parameter schemas within the selected index."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": TOOL_SEARCH_MAX_LIMIT,
                    "default": TOOL_SEARCH_DEFAULT_LIMIT,
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": (
                        "Result offset used for pagination. Always use the returned "
                        "next_offset; live-context deduplication may reset it to 0."
                    ),
                },
                "min_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0,
                },
            },
            "required": ["index", "keywords"],
            "additionalProperties": False,
        },
        handler=search_tools,
    )
    invoke_tool = FunctionTool(
        name=TOOL_INVOKE_NAME,
        description=(
            "Invoke a hidden tool returned by tool_search. Pass its exact tool_id "
            "and arguments matching the returned parameter schema."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tool_id": {
                    "type": "string",
                    "description": "Exact tool_id returned by tool_search.",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments matching the selected tool schema.",
                    "default": {},
                    "additionalProperties": True,
                },
            },
            "required": ["tool_id"],
            "additionalProperties": False,
        },
        handler=invoke_guard,
    )
    return ToolSet(tools=[search_tool, invoke_tool])
