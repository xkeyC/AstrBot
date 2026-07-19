import json
from types import SimpleNamespace

import pytest

import astrbot.core.message.components as Comp
import astrbot.core.provider.sources.openai_source as openai_source_module
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.exceptions import EmptyModelOutputError
from astrbot.core.provider.responses_tool_search import (
    TOOL_SEARCH_HISTORY_MARKER_KEY,
    TOOL_SEARCH_HISTORY_MARKER_VALUE,
)
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial


class _FakeResponses:
    def __init__(self, events):
        self.events = events
        self.payload = None

    async def create(self, **payload):
        self.payload = payload
        return _FakeStream(self.events)


class _FakeStream:
    def __init__(self, events):
        self.events = events

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self.events:
            yield event


def _make_provider(events) -> tuple[ProviderOpenAIOfficial, _FakeResponses]:
    fake_responses = _FakeResponses(events)
    provider = ProviderOpenAIOfficial.__new__(ProviderOpenAIOfficial)
    provider.provider_config = {
        "type": "openai_chat_completion",
        "api_mode": "responses",
        "custom_extra_body": {},
    }
    provider.provider_settings = {}
    provider.api_mode = "responses"
    provider.model_name = "gpt-4.1"
    provider.reasoning_key = "reasoning_content"
    provider.responses_default_params = {
        "model",
        "input",
        "tools",
        "tool_choice",
        "stream",
        "extra_body",
    }
    provider.client = SimpleNamespace(responses=fake_responses)
    return provider, fake_responses


def _completed_event(output=None):
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            id="resp_1",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=2,
                input_tokens_details=SimpleNamespace(cached_tokens=3),
            ),
            output=output or [],
        ),
    )


@pytest.mark.asyncio
async def test_responses_api_streaming_text_and_usage():
    provider, fake_responses = _make_provider(
        [
            SimpleNamespace(
                type="response.output_text.delta",
                delta="hel",
                item_id="msg_1",
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                delta="lo",
                item_id="msg_1",
            ),
            _completed_event(),
        ]
    )

    responses = [
        response
        async for response in provider._query_responses_stream(
            {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
            None,
        )
    ]

    assert [response.completion_text for response in responses] == [
        "hel",
        "lo",
        "hello",
    ]
    assert responses[-1].usage.input_other == 7
    assert responses[-1].usage.input_cached == 3
    assert responses[-1].usage.output == 2
    assert fake_responses.payload["input"] == [{"role": "user", "content": "hi"}]
    assert fake_responses.payload["stream"] is True


@pytest.mark.asyncio
async def test_responses_api_streaming_tool_call():
    tool = FunctionTool(
        name="lookup",
        description="Lookup data",
        parameters={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        handler=None,
    )
    tool_item = SimpleNamespace(
        type="function_call",
        call_id="call_1",
        name="lookup",
        arguments='{"q":"abc"}',
    )
    provider, fake_responses = _make_provider(
        [
            SimpleNamespace(type="response.output_item.done", item=tool_item),
            _completed_event(output=[tool_item]),
        ]
    )

    responses = [
        response
        async for response in provider._query_responses_stream(
            {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
            ToolSet([tool]),
        )
    ]

    final = responses[-1]
    assert final.role == "tool"
    assert final.tools_call_ids == ["call_1"]
    assert final.tools_call_name == ["lookup"]
    assert final.tools_call_args == [{"q": "abc"}]
    assert fake_responses.payload["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Lookup data",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
            "strict": None,
        }
    ]


@pytest.mark.asyncio
async def test_responses_api_orders_kb_tool_after_other_tools():
    kb_tool = FunctionTool(
        name="astr_kb_search",
        description="Search knowledge base",
        parameters={"type": "object", "properties": {}},
        handler=None,
    )
    shell_tool = FunctionTool(
        name="astrbot_execute_shell",
        description="Execute shell command",
        parameters={"type": "object", "properties": {}},
        handler=None,
    )
    provider, fake_responses = _make_provider([_completed_event(output=[])])

    with pytest.raises(EmptyModelOutputError):
        responses = [
            response
            async for response in provider._query_responses_stream(
                {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
                ToolSet([kb_tool, shell_tool]),
            )
        ]
        assert responses

    assert [tool["name"] for tool in fake_responses.payload["tools"]] == [
        "astrbot_execute_shell",
        "astr_kb_search",
    ]


@pytest.mark.asyncio
async def test_responses_api_merges_tool_item_id_and_call_id():
    tool_item = SimpleNamespace(
        type="function_call",
        id="fc_item_1",
        call_id="call_1",
        name="lookup",
        arguments='{"q":"abc"}',
    )
    provider, _ = _make_provider(
        [
            SimpleNamespace(
                type="response.function_call_arguments.done",
                item_id="fc_item_1",
                name="lookup",
                arguments='{"q":"abc"}',
            ),
            SimpleNamespace(type="response.output_item.done", item=tool_item),
            _completed_event(output=[tool_item]),
        ]
    )

    responses = [
        response
        async for response in provider._query_responses_stream(
            {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
            None,
        )
    ]

    final = responses[-1]
    assert final.tools_call_ids == ["call_1"]
    assert final.tools_call_name == ["lookup"]
    assert final.tools_call_args == [{"q": "abc"}]


@pytest.mark.asyncio
async def test_responses_api_accumulates_tool_argument_deltas():
    tool_item_added = SimpleNamespace(
        type="function_call",
        id="fc_item_1",
        call_id="call_1",
        name="lookup",
    )
    tool_item_done = SimpleNamespace(
        type="function_call",
        id="fc_item_1",
        call_id="call_1",
        name="lookup",
    )
    provider, _ = _make_provider(
        [
            SimpleNamespace(type="response.output_item.added", item=tool_item_added),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="fc_item_1",
                delta='{"q":',
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="fc_item_1",
                delta='"abc"}',
            ),
            SimpleNamespace(
                type="response.function_call_arguments.done",
                item_id="fc_item_1",
            ),
            SimpleNamespace(type="response.output_item.done", item=tool_item_done),
            _completed_event(output=[tool_item_done]),
        ]
    )

    responses = [
        response
        async for response in provider._query_responses_stream(
            {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
            None,
        )
    ]

    final = responses[-1]
    assert final.tools_call_ids == ["call_1"]
    assert final.tools_call_name == ["lookup"]
    assert final.tools_call_args == [{"q": "abc"}]


@pytest.mark.asyncio
async def test_responses_api_uses_completed_output_text_when_no_delta():
    output_message = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text="completed text")],
    )
    provider, _ = _make_provider([_completed_event(output=[output_message])])

    responses = [
        response
        async for response in provider._query_responses_stream(
            {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
            None,
        )
    ]

    assert len(responses) == 1
    assert responses[0].completion_text == "completed text"


@pytest.mark.asyncio
async def test_responses_api_returns_image_generation_result():
    image_item = SimpleNamespace(
        type="image_generation_call",
        result="iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB",
    )
    provider, _ = _make_provider([_completed_event(output=[image_item])])

    responses = [
        response
        async for response in provider._query_responses_stream(
            {
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "draw an image"}],
            },
            None,
        )
    ]

    assert len(responses) == 1
    assert responses[0].result_chain is not None
    assert len(responses[0].result_chain.chain) == 1
    image = responses[0].result_chain.chain[0]
    assert isinstance(image, Comp.Image)
    assert image.file == "base64://iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"


@pytest.mark.asyncio
async def test_responses_api_returns_streamed_image_generation_item():
    image_item = SimpleNamespace(
        type="image_generation_call",
        result="iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB",
    )
    provider, _ = _make_provider(
        [
            SimpleNamespace(type="response.output_item.done", item=image_item),
            _completed_event(output=[]),
        ]
    )

    responses = [
        response
        async for response in provider._query_responses_stream(
            {
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "draw an image"}],
            },
            None,
        )
    ]

    assert len(responses) == 1
    assert responses[0].result_chain is not None
    image = responses[0].result_chain.chain[0]
    assert isinstance(image, Comp.Image)
    assert image.file == "base64://iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"


@pytest.mark.asyncio
async def test_responses_api_ignores_builtin_search_call_when_text_is_returned():
    search_item = SimpleNamespace(type="web_search_call", status="completed")
    output_message = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text="search summary")],
    )
    provider, _ = _make_provider(
        [
            SimpleNamespace(type="response.output_item.done", item=search_item),
            _completed_event(output=[search_item, output_message]),
        ]
    )

    responses = [
        response
        async for response in provider._query_responses_stream(
            {
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "search the web"}],
            },
            None,
        )
    ]

    assert len(responses) == 1
    assert responses[0].completion_text == "search summary"


def test_responses_api_converts_tool_history_to_response_items():
    provider, _ = _make_provider([])

    converted = provider._convert_messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"abc"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
    )

    assert converted == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"abc"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "result"},
    ]


@pytest.mark.asyncio
async def test_responses_tools_search_exposes_only_core_tools(monkeypatch):
    class CoreFunctionTool(FunctionTool):
        pass

    core_tool = CoreFunctionTool(
        name="core_tool",
        description="Core operation",
        parameters={"type": "object", "properties": {}},
        handler=None,
    )
    plugin_tool = FunctionTool(
        name="plugin_lookup",
        description="Look up plugin data",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=None,
    )
    output_message = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text="done")],
    )
    provider, fake_responses = _make_provider(
        [_completed_event(output=[output_message])]
    )
    provider.provider_config["tools_search"] = True
    monkeypatch.setattr(
        openai_source_module,
        "get_builtin_tool_name",
        lambda tool_type: "core_tool" if tool_type is CoreFunctionTool else None,
    )

    responses = [
        response
        async for response in provider._query_responses_stream(
            {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
            ToolSet([core_tool, plugin_tool]),
        )
    ]

    assert responses[-1].completion_text == "done"
    assert fake_responses.payload["tools"] == [
        {
            "type": "function",
            "name": "core_tool",
            "description": "Core operation",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        },
        {
            "type": "tool_search",
            "execution": "client",
            "description": fake_responses.payload["tools"][1]["description"],
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for deferred tools.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of tools to return. Defaults to 8.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    ]


@pytest.mark.asyncio
async def test_responses_tools_search_call_returns_deferred_tool(monkeypatch):
    plugin_tool = FunctionTool(
        name="plugin_lookup",
        description="Look up plugin data",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=None,
    )
    search_item = SimpleNamespace(
        type="tool_search_call",
        call_id="search_1",
        execution="client",
        arguments={"query": "plugin lookup", "limit": 1},
    )
    provider, _ = _make_provider(
        [
            SimpleNamespace(type="response.output_item.done", item=search_item),
            _completed_event(output=[search_item]),
        ]
    )
    provider.provider_config["tools_search"] = True
    monkeypatch.setattr(
        openai_source_module,
        "get_builtin_tool_name",
        lambda _tool_type: None,
    )

    responses = [
        response
        async for response in provider._query_responses_stream(
            {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
            ToolSet([plugin_tool]),
        )
    ]

    final = responses[-1]
    assert final.tools_call_ids == ["search_1"]
    assert final.tools_call_name == ["tool_search"]
    assert final.tools_call_args == [{"query": "plugin lookup", "limit": 1}]
    assert final.tools_call_extra_content == {
        "search_1": {TOOL_SEARCH_HISTORY_MARKER_KEY: TOOL_SEARCH_HISTORY_MARKER_VALUE}
    }
    internal_tool = final.internal_tools["search_1"]
    output = json.loads(
        await internal_tool.handler(None, query="plugin lookup", limit=1)
    )
    assert output == {
        "tools": [
            {
                "type": "function",
                "name": "plugin_lookup",
                "description": "Look up plugin data",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "strict": False,
                "defer_loading": True,
            }
        ]
    }


@pytest.mark.asyncio
async def test_responses_tools_search_does_not_trust_builtin_tool_name(monkeypatch):
    class CoreFunctionTool(FunctionTool):
        pass

    plugin_tool = FunctionTool(
        name="core_tool",
        description="Plugin using a builtin tool name",
        parameters={"type": "object", "properties": {}},
        handler=None,
    )
    output_message = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text="done")],
    )
    provider, fake_responses = _make_provider(
        [_completed_event(output=[output_message])]
    )
    provider.provider_config["tools_search"] = True
    monkeypatch.setattr(
        openai_source_module,
        "get_builtin_tool_name",
        lambda tool_type: "core_tool" if tool_type is CoreFunctionTool else None,
    )

    responses = [
        response
        async for response in provider._query_responses_stream(
            {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
            ToolSet([plugin_tool]),
        )
    ]

    assert responses[-1].completion_text == "done"
    assert [tool["type"] for tool in fake_responses.payload["tools"]] == ["tool_search"]


@pytest.mark.asyncio
async def test_responses_tools_search_does_not_capture_same_name_function(
    monkeypatch,
):
    plugin_tool = FunctionTool(
        name="tool_search",
        description="An ordinary plugin function",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=None,
    )
    function_item = SimpleNamespace(
        type="function_call",
        call_id="ordinary_search_1",
        name="tool_search",
        arguments='{"query":"calendar"}',
    )
    provider, _ = _make_provider(
        [
            SimpleNamespace(type="response.output_item.done", item=function_item),
            _completed_event(output=[function_item]),
        ]
    )
    provider.provider_config["tools_search"] = True
    monkeypatch.setattr(
        openai_source_module,
        "get_builtin_tool_name",
        lambda _tool_type: None,
    )

    responses = [
        response
        async for response in provider._query_responses_stream(
            {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
            ToolSet([plugin_tool]),
        )
    ]

    final = responses[-1]
    assert final.tools_call_name == ["tool_search"]
    assert final.internal_tools == {}
    assert final.tools_call_extra_content == {}


@pytest.mark.asyncio
async def test_responses_tools_search_is_scoped_to_each_response(monkeypatch):
    first_tool = FunctionTool(
        name="first_lookup",
        description="Look up alpha data",
        parameters={"type": "object", "properties": {}},
        handler=None,
    )
    second_tool = FunctionTool(
        name="second_lookup",
        description="Look up beta data",
        parameters={"type": "object", "properties": {}},
        handler=None,
    )
    first_search_item = SimpleNamespace(
        type="tool_search_call",
        call_id="search_1",
        execution="client",
        arguments={"query": "alpha", "limit": 1},
    )
    provider, fake_responses = _make_provider(
        [
            SimpleNamespace(type="response.output_item.done", item=first_search_item),
            _completed_event(output=[first_search_item]),
        ]
    )
    provider.provider_config["tools_search"] = True
    monkeypatch.setattr(
        openai_source_module,
        "get_builtin_tool_name",
        lambda _tool_type: None,
    )

    first_responses = [
        response
        async for response in provider._query_responses_stream(
            {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
            ToolSet([first_tool]),
        )
    ]
    first_final = first_responses[-1]

    second_search_item = SimpleNamespace(
        type="tool_search_call",
        call_id="search_1",
        execution="client",
        arguments={"query": "beta", "limit": 1},
    )
    fake_responses.events = [
        SimpleNamespace(type="response.output_item.done", item=second_search_item),
        _completed_event(output=[second_search_item]),
    ]
    second_responses = [
        response
        async for response in provider._query_responses_stream(
            {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
            ToolSet([second_tool]),
        )
    ]
    second_final = second_responses[-1]

    first_output = json.loads(
        await first_final.internal_tools["search_1"].handler(
            None, query="alpha", limit=1
        )
    )
    second_output = json.loads(
        await second_final.internal_tools["search_1"].handler(
            None, query="beta", limit=1
        )
    )
    assert [tool["name"] for tool in first_output["tools"]] == ["first_lookup"]
    assert [tool["name"] for tool in second_output["tools"]] == ["second_lookup"]


def test_responses_api_converts_tool_search_history_to_native_items():
    provider, _ = _make_provider([])

    converted = provider._convert_messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "search_1",
                        "type": "function",
                        "function": {
                            "name": "tool_search",
                            "arguments": '{"query":"calendar"}',
                        },
                        "extra_content": {
                            TOOL_SEARCH_HISTORY_MARKER_KEY: TOOL_SEARCH_HISTORY_MARKER_VALUE
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "search_1",
                "content": '{"tools":[{"type":"function","name":"calendar"}]}',
            },
        ]
    )

    assert converted == [
        {
            "type": "tool_search_call",
            "call_id": "search_1",
            "execution": "client",
            "arguments": {"query": "calendar"},
        },
        {
            "type": "tool_search_output",
            "call_id": "search_1",
            "status": "completed",
            "execution": "client",
            "tools": [{"type": "function", "name": "calendar"}],
        },
    ]


def test_responses_api_keeps_unmarked_tool_search_as_function_call():
    provider, _ = _make_provider([])

    converted = provider._convert_messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "ordinary_search_1",
                        "type": "function",
                        "function": {
                            "name": "tool_search",
                            "arguments": '{"query":"calendar"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "ordinary_search_1",
                "content": "ordinary result",
            },
        ]
    )

    assert converted == [
        {
            "type": "function_call",
            "call_id": "ordinary_search_1",
            "name": "tool_search",
            "arguments": '{"query":"calendar"}',
        },
        {
            "type": "function_call_output",
            "call_id": "ordinary_search_1",
            "output": "ordinary result",
        },
    ]


@pytest.mark.asyncio
async def test_responses_api_non_streaming_query_raises():
    provider, _ = _make_provider([])

    with pytest.raises(RuntimeError, match="only supports streaming"):
        await provider._query({"model": "gpt-4.1", "messages": []}, None)


@pytest.mark.asyncio
async def test_responses_api_empty_stream_raises():
    provider, _ = _make_provider([_completed_event()])

    with pytest.raises(EmptyModelOutputError):
        responses = [
            response
            async for response in provider._query_responses_stream(
                {"model": "gpt-4.1", "messages": []},
                None,
            )
        ]
        assert responses
