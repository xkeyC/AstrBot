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
