import json

import pytest

from astrbot.core.agent.tool import FunctionTool
from astrbot.core.provider.responses_tool_search import create_responses_tool_search


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tools", "query", "expected_name"),
    [
        (
            [("weather_lookup", "Look up weather forecasts")],
            "weather",
            "weather_lookup",
        ),
        (
            [
                ("calendar_lookup", "Look up calendar events"),
                ("weather_lookup", "Look up weather forecasts"),
            ],
            "weather",
            "weather_lookup",
        ),
        (
            [
                ("calendar_lookup", "Look up calendar events"),
                ("weather_lookup", "Look up forecast data"),
                ("contact_lookup", "Look up contact records"),
            ],
            "forecast",
            "weather_lookup",
        ),
    ],
)
async def test_responses_tool_search_ranks_small_corpora(
    tools: list[tuple[str, str]],
    query: str,
    expected_name: str,
):
    deferred_tools = [
        FunctionTool(
            name=name,
            description=description,
            parameters={"type": "object", "properties": {}},
            handler=None,
        )
        for name, description in tools
    ]
    search_tool, _ = create_responses_tool_search(deferred_tools)

    output = json.loads(await search_tool.handler(None, query=query, limit=1))

    assert [tool["name"] for tool in output["tools"]] == [expected_name]


@pytest.mark.asyncio
async def test_responses_tool_search_returns_no_tools_without_a_match():
    deferred_tool = FunctionTool(
        name="weather_lookup",
        description="Look up weather forecasts",
        parameters={"type": "object", "properties": {}},
        handler=None,
    )
    search_tool, _ = create_responses_tool_search([deferred_tool])

    output = json.loads(await search_tool.handler(None, query="unrelated", limit=1))

    assert output == {"tools": []}


@pytest.mark.asyncio
async def test_responses_tool_search_prioritizes_english_name_matches():
    deferred_tools = [
        FunctionTool(
            name="generic_lookup",
            description="Search QQ group members and return member details.",
            parameters={"type": "object", "properties": {}},
            handler=None,
        ),
        FunctionTool(
            name="qq_group_members",
            description="获取群成员列表。",
            parameters={"type": "object", "properties": {}},
            handler=None,
        ),
    ]
    search_tool, _ = create_responses_tool_search(deferred_tools)

    output = json.loads(
        await search_tool.handler(None, query="qq group members", limit=2)
    )

    assert [tool["name"] for tool in output["tools"]] == [
        "qq_group_members",
        "generic_lookup",
    ]


@pytest.mark.asyncio
async def test_responses_tool_search_prioritizes_qq_tools_for_mixed_query():
    tool_specs = [
        ("qq_file", "管理 QQ 群文件、群相册、照片和视频列表。", {}),
        ("qq_group_members", "分页获取 QQ 群成员列表。", {}),
        ("qq_delete_files", "删除 QQ 群文件。", {}),
        ("get_wiki_summary", "获取 MediaWiki 页面摘要。", {}),
        ("get_wiki_section", "获取 MediaWiki 页面章节。", {}),
        ("github_get_team_members", "Get members of a GitHub team.", {}),
        (
            "ha_search",
            "Search Home Assistant entities.",
            {
                "group_members": {
                    "type": "string",
                    "description": "Optional group members search filter.",
                },
                "album": {
                    "type": "string",
                    "description": "Optional album-like media category.",
                },
            },
        ),
    ]
    deferred_tools = [
        FunctionTool(
            name=name,
            description=description,
            parameters={"type": "object", "properties": parameters},
            handler=None,
        )
        for name, description, parameters in tool_specs
    ]
    search_tool, _ = create_responses_tool_search(deferred_tools)

    output = json.loads(
        await search_tool.handler(
            None,
            query=(
                "QQ 获取群成员列表 查看群相册 相册列表 照片列表 qq group members album"
            ),
            limit=7,
        )
    )
    names = [tool["name"] for tool in output["tools"]]

    assert names[:2] == ["qq_group_members", "qq_file"]
    for distractor in (
        "get_wiki_summary",
        "get_wiki_section",
        "github_get_team_members",
        "ha_search",
    ):
        assert names.index("qq_group_members") < names.index(distractor)
        assert names.index("qq_file") < names.index(distractor)


@pytest.mark.asyncio
async def test_responses_tool_search_default_limit_is_eight():
    deferred_tools = [
        FunctionTool(
            name=f"qq_tool_{index}",
            description="QQ utility.",
            parameters={"type": "object", "properties": {}},
            handler=None,
        )
        for index in range(10)
    ]
    search_tool, _ = create_responses_tool_search(deferred_tools)

    output = json.loads(await search_tool.handler(None, query="qq"))

    assert len(output["tools"]) == 8
