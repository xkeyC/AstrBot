from types import SimpleNamespace

import pytest

from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.exceptions import EmptyModelOutputError
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

    assert [response.completion_text for response in responses] == ["hel", "lo", "hello"]
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
