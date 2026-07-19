import json

import pytest

from astrbot.core.agent.tool import FunctionTool
from astrbot.core.agent.tool_registry import (
    TOOL_INVOKE_NAME,
    TOOL_SEARCH_NAME,
    build_tool_prefix_index,
    create_tool_registry_tools,
)


def _tool(name: str, description: str = "Tool description") -> FunctionTool:
    return FunctionTool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
        handler=None,
    )


@pytest.mark.asyncio
async def test_tool_registry_search_filters_prefix_and_paginates():
    registry = create_tool_registry_tools(
        [
            _tool("qq_group_members", "Get QQ group members."),
            _tool("qq_file", "List QQ group albums and files."),
            _tool("github_get_team_members", "Get GitHub team members."),
        ]
    )
    search_tool = registry.get_tool(TOOL_SEARCH_NAME)

    first_page = json.loads(
        await search_tool.handler(None, index="qq_", keywords=[], limit=1)
    )
    second_page = json.loads(
        await search_tool.handler(
            None,
            index="qq_",
            keywords=[],
            limit=1,
            offset=first_page["next_offset"],
        )
    )

    assert [tool["tool_id"] for tool in first_page["tools"]] == ["qq_group_members"]
    assert first_page["has_more"] is True
    assert first_page["total"] == 2
    assert first_page["returned"] == 1
    assert first_page["next_offset"] == 1
    assert [tool["tool_id"] for tool in second_page["tools"]] == ["qq_file"]
    assert second_page["has_more"] is False
    assert second_page["next_offset"] is None
    assert second_page["tools"][0]["parameters"]["properties"]["query"] == {
        "type": "string"
    }


@pytest.mark.asyncio
async def test_tool_registry_can_enumerate_all_tools_by_page():
    registry = create_tool_registry_tools(
        [_tool("qq_file"), _tool("github_team"), _tool("standalone")]
    )
    search_tool = registry.get_tool(TOOL_SEARCH_NAME)

    first_page = json.loads(
        await search_tool.handler(None, index="*", keywords=[], limit=2)
    )
    second_page = json.loads(
        await search_tool.handler(
            None,
            index="*",
            keywords=[],
            limit=2,
            offset=first_page["next_offset"],
        )
    )

    assert first_page["total"] == 3
    assert first_page["returned"] == 2
    assert first_page["has_more"] is True
    assert second_page["returned"] == 1
    assert second_page["has_more"] is False


@pytest.mark.asyncio
async def test_tool_registry_requires_an_exact_prefix_index():
    registry = create_tool_registry_tools([_tool("qq_file"), _tool("qq_members")])
    search_tool = registry.get_tool(TOOL_SEARCH_NAME)

    output = json.loads(await search_tool.handler(None, index="qq", keywords=["file"]))

    assert output["tools"] == []
    assert output["error"].startswith("Unknown tool index")


@pytest.mark.asyncio
async def test_tool_registry_keywords_match_parameter_schema_within_index():
    album_tool = FunctionTool(
        name="qq_lookup",
        description="QQ lookup.",
        parameters={
            "type": "object",
            "properties": {"album_id": {"type": "string"}},
        },
        handler=None,
    )
    member_tool = FunctionTool(
        name="qq_search",
        description="QQ search.",
        parameters={
            "type": "object",
            "properties": {"member_id": {"type": "string"}},
        },
        handler=None,
    )
    registry = create_tool_registry_tools([album_tool, member_tool])
    search_tool = registry.get_tool(TOOL_SEARCH_NAME)

    output = json.loads(
        await search_tool.handler(None, index="qq_", keywords=["album_id"])
    )

    assert [tool["tool_id"] for tool in output["tools"]] == ["qq_lookup"]


def test_tool_registry_exposes_only_stable_meta_tools():
    registry = create_tool_registry_tools([_tool("qq_group_members")])

    assert registry.names() == [TOOL_SEARCH_NAME, TOOL_INVOKE_NAME]


def test_tool_prefix_index_lists_effective_namespaces():
    prompt = build_tool_prefix_index(
        [
            _tool("qq_group_members"),
            _tool("qq_file"),
            _tool("github_get_team_members"),
            _tool("standalone"),
        ]
    )

    assert "- `qq_` (2 tools)" in prompt
    assert "group_members, file" in prompt
    assert "- `github_` (1 tool)" in prompt
    assert "standalone" not in prompt
    assert "Pass one of these exact values as `index`" in prompt
