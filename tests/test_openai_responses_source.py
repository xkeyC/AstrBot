import json
from types import SimpleNamespace

import pytest
from openai.types.responses import Response

import astrbot.core.message.components as Comp
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.config.default import CONFIG_METADATA_2
from astrbot.core.provider.sources.openai_responses_source import (
    ProviderOpenAIResponses,
)


def _make_provider(overrides: dict | None = None) -> ProviderOpenAIResponses:
    provider_config = {
        "id": "test-responses",
        "provider": "openai",
        "type": "openai_responses",
        "model": "gpt-test",
        "key": ["test-key"],
        "api_base": "https://api.openai.com/v1",
    }
    if overrides:
        provider_config.update(overrides)
    return ProviderOpenAIResponses(provider_config, {})


def _make_response(output: list[dict], **overrides) -> Response:
    payload = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-test",
        "output": output,
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {
                "cached_tokens": 3,
                "cache_write_tokens": 0,
            },
            "output_tokens": 4,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 14,
        },
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    payload.update(overrides)
    return Response.model_validate(payload)


def test_responses_provider_templates_are_independent_and_stateless():
    templates = CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
        "config_template"
    ]

    assert templates["OpenAI Responses"]["type"] == "openai_responses"
    assert templates["OpenAI Responses"]["api_base"] == "https://api.openai.com/v1"
    assert templates["OpenAI Responses"]["responses_web_search"] is False
    assert templates["OpenAI Responses"]["responses_tool_choice"] == "auto"
    assert templates["OpenAI Responses"]["responses_compact_threshold"] == 0
    assert templates["DeepSeek Responses"]["type"] == "openai_responses"
    assert templates["DeepSeek Responses"]["api_base"] == "https://api.deepseek.com/v1"
    assert templates["xAI"]["type"] == "openai_responses"
    assert templates["xAI"]["api_base"] == "https://api.x.ai/v1"
    assert "xai_native_search" not in templates["xAI"]


def test_convert_chat_history_preserves_response_items_and_function_calls():
    provider = _make_provider()
    reasoning_item = {
        "id": "rs_1",
        "type": "reasoning",
        "status": "completed",
        "summary": [],
        "encrypted_content": "encrypted-reasoning",
    }
    reasoning_state = json.dumps(
        {
            "type": provider._REASONING_STATE_TYPE,
            "scope": provider._replay_state_scope(),
            "items": [reasoning_item],
        }
    )

    response_input = provider._convert_chat_messages_to_response_input(
        [
            {"role": "system", "content": "system context"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,AAAA",
                            "detail": "high",
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "think",
                        "think": "hidden",
                        "encrypted": reasoning_state,
                    },
                    {"type": "text", "text": "calling"},
                ],
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"SZ"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
    )

    assert response_input == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look"},
                {
                    "type": "input_image",
                    "detail": "high",
                    "image_url": "data:image/png;base64,AAAA",
                },
            ],
        },
        reasoning_item,
        {"type": "message", "role": "assistant", "content": "calling"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "weather",
            "arguments": '{"city":"SZ"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "sunny",
        },
    ]


def test_deepseek_converts_plain_reasoning_history_to_reasoning_item():
    provider = _make_provider(
        {
            "provider": "deepseek",
            "api_base": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
        }
    )

    response_input = provider._convert_chat_messages_to_response_input(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "think", "think": "prior thought"},
                    {"type": "text", "text": "prior answer"},
                ],
            }
        ]
    )

    assert response_input == [
        {
            "type": "reasoning",
            "content": [
                {"type": "reasoning_text", "text": "prior thought"},
            ],
            "summary": [],
        },
        {"type": "message", "role": "assistant", "content": "prior answer"},
    ]


@pytest.mark.asyncio
async def test_prepare_payload_replays_full_history_without_server_state():
    provider = _make_provider()

    payloads, context = await provider._prepare_chat_payload(
        prompt="current",
        contexts=[
            {"role": "system", "content": "root instructions"},
            {"role": "developer", "content": "application instructions"},
            {"role": "user", "content": "previous"},
        ],
        system_prompt="request instructions",
    )

    assert context == [
        {"role": "user", "content": "previous"},
        {"role": "user", "content": "current"},
    ]
    assert payloads == {
        "model": "gpt-test",
        "store": False,
        "instructions": (
            "request instructions\n\nroot instructions\n\napplication instructions"
        ),
        "input": [
            {"type": "message", "role": "user", "content": "previous"},
            {"type": "message", "role": "user", "content": "current"},
        ],
    }
    assert "previous_response_id" not in payloads
    assert "conversation" not in payloads
    assert all(
        item.get("role") not in {"system", "developer"}
        for item in payloads["input"]
        if item.get("type") == "message"
    )


@pytest.mark.asyncio
async def test_query_flattens_tools_and_enforces_stateless_body(monkeypatch):
    provider = _make_provider(
        {
            "custom_extra_body": {
                "max_tokens": 321,
                "reasoning_effort": "low",
                "previous_response_id": "resp_previous",
                "conversation": "conv_1",
                "store": True,
            },
            "responses_compact_threshold": 64000,
        }
    )
    captured: dict = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _make_response(
            [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "weather",
                    "arguments": '{"city":"SZ"}',
                    "status": "completed",
                }
            ]
        )

    monkeypatch.setattr(provider.client.responses, "create", fake_create)
    tools = SimpleNamespace(
        openai_schema=lambda: [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
    )

    result = await provider._query(
        {
            "model": "gpt-test",
            "input": "weather",
            "store": True,
            "previous_response_id": "resp_direct",
            "conversation": "conv_direct",
        },
        tools,
    )

    assert captured["store"] is False
    assert captured["stream"] is False
    assert "previous_response_id" not in captured
    assert "conversation" not in captured
    assert captured["context_management"] == [
        {"type": "compaction", "compact_threshold": 64000}
    ]
    assert captured["tools"] == [
        {
            "type": "function",
            "name": "weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]
    assert captured["extra_body"] == {
        "max_output_tokens": 321,
        "reasoning": {"effort": "low"},
    }
    assert result.role == "tool"
    assert result.tools_call_name == ["weather"]
    assert result.tools_call_args == [{"city": "SZ"}]
    assert result.tools_call_ids == ["call_1"]


def test_compaction_uses_extra_body_with_older_openai_sdk_signatures():
    provider = _make_provider({"responses_compact_threshold": 32000})
    provider.default_params = {
        parameter
        for parameter in provider.default_params
        if parameter != "context_management"
    }
    payloads = {"model": "gpt-test", "input": "continue"}

    extra_body = provider._prepare_response_request(payloads, tools=None)

    assert "context_management" not in payloads
    assert extra_body["context_management"] == [
        {"type": "compaction", "compact_threshold": 32000}
    ]


@pytest.mark.asyncio
async def test_query_combines_astrbot_and_responses_native_tools(monkeypatch):
    provider = _make_provider(
        {
            "responses_web_search": True,
            "responses_web_search_context_size": "high",
            "responses_web_search_allowed_domains": ["example.com"],
            "responses_file_search_vector_store_ids": ["vs_1"],
            "responses_code_interpreter": True,
            "responses_image_generation": True,
            "responses_tool_choice": "required",
        }
    )
    captured: dict = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _make_response(
            [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "done", "annotations": []},
                    ],
                }
            ]
        )

    monkeypatch.setattr(provider.client.responses, "create", fake_create)
    tools = SimpleNamespace(
        openai_schema=lambda: [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            }
        ]
    )

    await provider._query({"model": "gpt-test", "input": "hi"}, tools)

    assert captured["tool_choice"] == "required"
    assert captured["tools"] == [
        {
            "type": "function",
            "name": "weather",
            "description": "Get weather",
            "parameters": {"type": "object"},
        },
        {
            "type": "web_search",
            "search_context_size": "high",
            "filters": {"allowed_domains": ["example.com"]},
        },
        {"type": "file_search", "vector_store_ids": ["vs_1"]},
        {"type": "code_interpreter", "container": {"type": "auto"}},
        {"type": "image_generation"},
    ]


def test_build_response_tools_deduplicates_configured_and_custom_tools():
    provider = _make_provider(
        {
            "responses_web_search": True,
            "responses_file_search_vector_store_ids": ["vs_1"],
            "responses_code_interpreter": True,
            "responses_image_generation": True,
        }
    )
    tools = SimpleNamespace(
        openai_schema=lambda: [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {"type": "object"},
                },
            }
        ]
    )

    response_tools = provider._build_response_tools(
        tools,
        [
            {"type": "function", "name": "weather", "parameters": {}},
            {"type": "web_search", "search_context_size": "high"},
            {"type": "file_search", "vector_store_ids": ["vs_custom"]},
            {"type": "code_interpreter", "container": {"type": "auto"}},
            {"type": "image_generation"},
            {"type": "computer_use", "display_width": 1024},
        ],
    )

    assert response_tools == [
        {"type": "function", "name": "weather", "parameters": {"type": "object"}},
        {"type": "web_search", "search_context_size": "medium"},
        {"type": "file_search", "vector_store_ids": ["vs_1"]},
        {"type": "code_interpreter", "container": {"type": "auto"}},
        {"type": "image_generation"},
        {"type": "computer_use", "display_width": 1024},
    ]


@pytest.mark.asyncio
async def test_parse_response_extracts_text_reasoning_usage_and_replay_state():
    provider = _make_provider()
    response = _make_response(
        [
            {
                "type": "reasoning",
                "id": "rs_1",
                "status": "completed",
                "summary": [],
                "encrypted_content": "encrypted-reasoning",
                "content": [
                    {"type": "reasoning_text", "text": "thinking"},
                ],
            },
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "answer", "annotations": []},
                ],
            },
        ]
    )

    result = await provider._parse_response(response, tools=None)

    assert result.completion_text == "answer"
    assert result.reasoning_content == "thinking"
    assert result.usage.input_other == 7
    assert result.usage.input_cached == 3
    assert result.usage.output == 4
    assert result.raw_completion is response
    state = json.loads(result.reasoning_signature)
    assert state["type"] == provider._REASONING_STATE_TYPE
    assert state["scope"] == provider._replay_state_scope()
    assert state["items"][0]["id"] == "rs_1"
    assert state["items"][0]["encrypted_content"] == "encrypted-reasoning"
    assert state["items"][0]["content"] == [
        {"text": "thinking", "type": "reasoning_text"}
    ]


def test_build_response_tools_keeps_knowledge_base_search_last():
    provider = _make_provider()
    tools = SimpleNamespace(
        openai_schema=lambda: [
            {
                "type": "function",
                "function": {"name": "astr_kb_search", "parameters": {}},
            },
            {
                "type": "function",
                "function": {"name": "tool_search", "parameters": {}},
            },
            {
                "type": "function",
                "function": {"name": "tool_invoke", "parameters": {}},
            },
        ]
    )

    response_tools = provider._build_response_tools(tools, custom_tools=None)

    assert [tool["name"] for tool in response_tools] == [
        "tool_search",
        "tool_invoke",
        "astr_kb_search",
    ]


@pytest.mark.asyncio
async def test_compaction_state_is_replayed_and_prunes_older_input():
    provider = _make_provider()
    response = _make_response(
        [
            {
                "type": "compaction",
                "id": "cmp_1",
                "encrypted_content": "encrypted-compaction",
            },
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "after", "annotations": []},
                ],
            },
        ]
    )

    result = await provider._parse_response(response, tools=None)
    response_input = provider._convert_chat_messages_to_response_input(
        [
            {"role": "user", "content": "discard me"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "think",
                        "think": "",
                        "encrypted": result.reasoning_signature,
                    },
                    {"type": "text", "text": "after"},
                ],
            },
            {"role": "user", "content": "continue"},
        ]
    )

    assert response_input == [
        {
            "type": "compaction",
            "id": "cmp_1",
            "encrypted_content": "encrypted-compaction",
        },
        {"type": "message", "role": "assistant", "content": "after"},
        {"type": "message", "role": "user", "content": "continue"},
    ]


def test_stale_compaction_state_is_discarded_without_pruning_history():
    provider = _make_provider()
    stale_state = json.dumps(
        {
            "type": provider._REASONING_STATE_TYPE,
            "scope": {
                **provider._replay_state_scope(),
                "provider_id": "another-provider",
            },
            "items": [
                {
                    "type": "compaction",
                    "id": "cmp_stale",
                    "encrypted_content": "encrypted-for-another-provider",
                }
            ],
        }
    )

    response_input = provider._convert_chat_messages_to_response_input(
        [
            {"role": "user", "content": "keep me"},
            {
                "role": "assistant",
                "content": [
                    {"type": "think", "think": "", "encrypted": stale_state},
                    {"type": "text", "text": "previous answer"},
                ],
            },
            {"role": "user", "content": "continue"},
        ]
    )

    assert response_input == [
        {"type": "message", "role": "user", "content": "keep me"},
        {"type": "message", "role": "assistant", "content": "previous answer"},
        {"type": "message", "role": "user", "content": "continue"},
    ]


@pytest.mark.asyncio
async def test_native_tool_output_items_are_preserved_for_stateless_replay():
    provider = _make_provider()
    native_tool_item = {
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"type": "search", "query": "AstrBot"},
    }
    response = SimpleNamespace(
        id="resp_1",
        status="completed",
        usage=None,
        output=[
            native_tool_item,
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "found", "annotations": []}
                ],
            },
        ],
    )

    result = await provider._parse_response(response, tools=None)
    state = json.loads(result.reasoning_signature)
    response_input = provider._convert_chat_messages_to_response_input(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "think",
                        "think": "",
                        "encrypted": result.reasoning_signature,
                    },
                    {"type": "text", "text": "found"},
                ],
            }
        ]
    )

    assert state["items"] == [native_tool_item]
    assert response_input == [
        native_tool_item,
        {"type": "message", "role": "assistant", "content": "found"},
    ]


@pytest.mark.asyncio
async def test_parse_response_accepts_compatible_output_image_payloads():
    provider = _make_provider()
    response = SimpleNamespace(
        id="resp_1",
        status="completed",
        usage=None,
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    {
                        "type": "output_image",
                        "image_url": "https://example.com/generated.png",
                    }
                ],
            )
        ],
    )

    result = await provider._parse_response(response, tools=None)

    assert result.completion_text == "[Image]"
    assert isinstance(result.result_chain.chain[1], Comp.Image)
    assert result.result_chain.chain[1].file == "https://example.com/generated.png"


@pytest.mark.asyncio
async def test_parse_response_keeps_web_citations_and_generated_images():
    provider = _make_provider()
    response = _make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The answer has a source.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "start_index": 18,
                                "end_index": 24,
                                "url": "https://example.com/source",
                                "title": "Example source",
                            },
                            {
                                "type": "file_citation",
                                "file_id": "file_1",
                                "filename": "notes.pdf",
                                "index": 0,
                            },
                        ],
                    }
                ],
            },
            {
                "type": "image_generation_call",
                "id": "img_1",
                "status": "completed",
                "result": "aGVsbG8=",
            },
        ]
    )

    result = await provider._parse_response(response, tools=None)

    assert "The answer has a source." in result.completion_text
    assert "Example source: https://example.com/source" in result.completion_text
    assert "notes.pdf (file_1)" in result.completion_text
    assert len(result.result_chain.chain) == 3
    assert result.result_chain.chain[1].type == "Image"


@pytest.mark.asyncio
async def test_parse_response_keeps_generated_images_as_non_empty_output():
    provider = _make_provider()
    response = _make_response(
        [
            {
                "type": "image_generation_call",
                "id": "img_1",
                "status": "completed",
                "result": "aGVsbG8=",
            }
        ]
    )

    result = await provider._parse_response(response, tools=None)

    assert result.completion_text == "[Image]"
    assert len(result.result_chain.chain) == 2
    assert result.result_chain.chain[0].type == "Plain"
    assert result.result_chain.chain[1].type == "Image"


@pytest.mark.asyncio
async def test_query_stream_yields_semantic_deltas_and_final_response(monkeypatch):
    provider = _make_provider()
    final_response = _make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "hello", "annotations": []},
                ],
            }
        ]
    )
    captured: dict = {}

    async def fake_stream():
        yield SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        )
        yield SimpleNamespace(type="response.reasoning_text.delta", delta="think")
        yield SimpleNamespace(type="response.output_text.delta", delta="hello")
        yield SimpleNamespace(type="response.completed", response=final_response)

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return fake_stream()

    monkeypatch.setattr(provider.client.responses, "create", fake_create)

    results = [
        result
        async for result in provider._query_stream(
            {"model": "gpt-test", "input": "hi", "store": False},
            tools=None,
        )
    ]

    assert captured["stream"] is True
    assert captured["store"] is False
    assert len(results) == 3
    assert results[0].is_chunk is True
    assert results[0].reasoning_content == "think"
    assert results[1].is_chunk is True
    assert results[1].completion_text == "hello"
    assert results[2].is_chunk is False
    assert results[2].completion_text == "hello"


@pytest.mark.asyncio
async def test_query_stream_merges_function_argument_deltas(monkeypatch):
    provider = _make_provider()
    final_response = _make_response(
        [
            {
                "type": "reasoning",
                "id": "rs_stream",
                "status": "completed",
                "summary": [],
                "encrypted_content": "encrypted-stream-reasoning",
            }
        ]
    )

    async def fake_stream():
        item = SimpleNamespace(
            type="function_call",
            id="fc_1",
            call_id="call_1",
            name="lookup",
        )
        yield SimpleNamespace(type="response.output_item.added", item=item)
        yield SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_1",
            delta='{"q":',
        )
        yield SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_1",
            delta='"abc"}',
        )
        yield SimpleNamespace(
            type="response.function_call_arguments.done",
            item_id="fc_1",
        )
        yield SimpleNamespace(type="response.output_item.done", item=item)
        yield SimpleNamespace(type="response.completed", response=final_response)

    async def fake_create(**_kwargs):
        return fake_stream()

    monkeypatch.setattr(provider.client.responses, "create", fake_create)

    results = [
        result
        async for result in provider._query_stream(
            {"model": "gpt-override", "input": "hi"},
            tools=None,
        )
    ]

    assert results[-1].role == "tool"
    assert results[-1].tools_call_ids == ["call_1"]
    assert results[-1].tools_call_name == ["lookup"]
    assert results[-1].tools_call_args == [{"q": "abc"}]
    replay_state = json.loads(results[-1].reasoning_signature)
    assert replay_state["scope"]["model"] == "gpt-override"
    assert replay_state["items"][0]["encrypted_content"] == (
        "encrypted-stream-reasoning"
    )


@pytest.mark.asyncio
async def test_query_stream_keeps_image_from_output_item_done(monkeypatch):
    provider = _make_provider()
    final_response = _make_response([])

    async def fake_stream():
        yield SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="image_generation_call",
                result="aGVsbG8=",
            ),
        )
        yield SimpleNamespace(type="response.completed", response=final_response)

    async def fake_create(**_kwargs):
        return fake_stream()

    monkeypatch.setattr(provider.client.responses, "create", fake_create)

    results = [
        result
        async for result in provider._query_stream(
            {"model": "gpt-test", "input": "draw"},
            tools=None,
        )
    ]

    assert results[-1].completion_text == "[Image]"
    assert isinstance(results[-1].result_chain.chain[1], Comp.Image)
    assert results[-1].result_chain.chain[1].file == "base64://aGVsbG8="


@pytest.mark.asyncio
async def test_parse_failed_response_raises_provider_error():
    provider = _make_provider()
    response = _make_response(
        [],
        status="failed",
        error={"code": "server_error", "message": "failed"},
        usage=None,
    )

    with pytest.raises(RuntimeError, match="server_error: failed"):
        await provider._parse_response(response, tools=None)


@pytest.mark.asyncio
async def test_parse_response_separates_independent_message_items():
    provider = _make_provider()
    response = _make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Let me look that up.",
                        "annotations": [],
                    }
                ],
            },
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "AstrBot"},
            },
            {
                "type": "message",
                "id": "msg_2",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "AstrBot is a bot.",
                        "annotations": [],
                    },
                ],
            },
        ]
    )

    result = await provider._parse_response(response, tools=None)

    assert result.completion_text == "Let me look that up.\n\nAstrBot is a bot."


@pytest.mark.asyncio
async def test_parse_response_keeps_one_message_item_unsplit():
    provider = _make_provider()
    response = _make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "first ", "annotations": []},
                    {"type": "output_text", "text": "second", "annotations": []},
                ],
            }
        ]
    )

    result = await provider._parse_response(response, tools=None)

    assert result.completion_text == "first second"


@pytest.mark.asyncio
async def test_query_stream_separates_message_items(monkeypatch):
    provider = _make_provider()
    final_response = _make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "searching", "annotations": []},
                ],
            },
            {
                "type": "message",
                "id": "msg_2",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "answer", "annotations": []},
                ],
            },
        ]
    )

    async def fake_stream():
        yield SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        )
        yield SimpleNamespace(type="response.output_text.delta", delta="searching")
        yield SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(type="message", id="msg_1"),
        )
        yield SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="message", id="msg_2"),
        )
        yield SimpleNamespace(type="response.output_text.delta", delta="answer")
        yield SimpleNamespace(type="response.completed", response=final_response)

    async def fake_create(**kwargs):
        return fake_stream()

    monkeypatch.setattr(provider.client.responses, "create", fake_create)

    results = [
        result
        async for result in provider._query_stream(
            {"model": "gpt-test", "input": "hi", "store": False},
            tools=None,
        )
    ]

    streamed = "".join(result.completion_text for result in results if result.is_chunk)
    assert streamed == "searching\n\nanswer"
    assert results[-1].completion_text == "searching\n\nanswer"


def test_hosted_tools_restate_conversation_rules_in_instructions():
    provider = _make_provider({"responses_web_search": True})
    payloads = {"model": "gpt-test", "input": [], "instructions": "Root prompt."}

    provider._prepare_response_request(payloads, tools=None)

    assert payloads["instructions"].startswith("Root prompt.\n\n")
    assert ProviderOpenAIResponses.HOSTED_TOOL_INSTRUCTION in payloads["instructions"]

    # A retried request must not stack the same reinforcement twice.
    provider._prepare_response_request(payloads, tools=None)
    assert (
        payloads["instructions"].count(ProviderOpenAIResponses.HOSTED_TOOL_INSTRUCTION)
        == 1
    )


def test_function_tools_only_requests_keep_instructions_untouched():
    provider = _make_provider()
    tools = ToolSet(
        [
            FunctionTool(
                name="ping",
                description="ping",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )
    payloads = {"model": "gpt-test", "input": [], "instructions": "Root prompt."}

    provider._prepare_response_request(payloads, tools=tools)

    assert payloads["instructions"] == "Root prompt."


@pytest.mark.asyncio
async def test_commentary_phase_is_isolated_from_the_reply():
    provider = _make_provider()
    response = _make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Let me search the web.",
                        "annotations": [],
                    }
                ],
                "phase": "commentary",
            },
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "AstrBot"},
            },
            {
                "type": "message",
                "id": "msg_2",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "AstrBot is a bot.",
                        "annotations": [],
                    },
                ],
                "phase": "final_answer",
            },
        ]
    )

    result = await provider._parse_response(response, tools=None)

    assert result.completion_text == "AstrBot is a bot."
    assert result.reasoning_content == "Let me search the web."


@pytest.mark.asyncio
async def test_phase_labelled_messages_are_replayed_verbatim():
    provider = _make_provider()
    response = _make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Working on it.", "annotations": []}
                ],
                "phase": "commentary",
            },
            {
                "type": "message",
                "id": "msg_2",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Done.", "annotations": []}
                ],
                "phase": "final_answer",
            },
        ]
    )

    result = await provider._parse_response(
        response, tools=None, request_model="gpt-test"
    )
    state = json.loads(result.reasoning_signature)
    response_input = provider._convert_chat_messages_to_response_input(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "think",
                        "think": "Working on it.",
                        "encrypted": result.reasoning_signature,
                    },
                    {"type": "text", "text": "Done."},
                ],
            }
        ],
        "gpt-test",
    )

    assert [item["phase"] for item in state["items"]] == ["commentary", "final_answer"]
    # The flattened history copy must not duplicate the replayed messages.
    assert response_input == state["items"]


@pytest.mark.asyncio
async def test_unlabelled_messages_keep_the_flattened_history_replay():
    provider = _make_provider()
    response = _make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Done.", "annotations": []}
                ],
            }
        ]
    )

    result = await provider._parse_response(response, tools=None)

    assert result.reasoning_signature is None


@pytest.mark.asyncio
async def test_query_stream_isolates_commentary_and_hosted_search(monkeypatch):
    provider = _make_provider()
    final_response = _make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "searching", "annotations": []},
                ],
                "phase": "commentary",
            },
            {
                "type": "message",
                "id": "msg_2",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "answer", "annotations": []},
                ],
                "phase": "final_answer",
            },
        ]
    )

    async def fake_stream():
        yield SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        )
        yield SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="message", id="msg_1", phase="commentary"),
        )
        yield SimpleNamespace(type="response.output_text.delta", delta="searching")
        yield SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(type="message", id="msg_1", phase="commentary"),
        )
        yield SimpleNamespace(type="response.web_search_call.in_progress")
        yield SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="message", id="msg_2", phase="final_answer"),
        )
        yield SimpleNamespace(type="response.output_text.delta", delta="answer")
        yield SimpleNamespace(type="response.completed", response=final_response)

    async def fake_create(**kwargs):
        return fake_stream()

    monkeypatch.setattr(provider.client.responses, "create", fake_create)

    results = [
        result
        async for result in provider._query_stream(
            {"model": "gpt-test", "input": "hi", "store": False},
            tools=None,
        )
    ]

    chunks = [result for result in results if result.is_chunk]
    assert [chunk.reasoning_content for chunk in chunks] == ["searching", None]
    assert chunks[1].completion_text == "answer"
    assert results[-1].completion_text == "answer"
    assert results[-1].reasoning_content == "searching"


@pytest.mark.asyncio
async def test_message_items_become_separate_reply_messages():
    provider = _make_provider()
    response = _make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Let me check.", "annotations": []}
                ],
            },
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "AstrBot"},
            },
            {
                "type": "message",
                "id": "msg_2",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "AstrBot is a bot.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "start_index": 0,
                                "end_index": 7,
                                "url": "https://example.com/a",
                                "title": "Example",
                            }
                        ],
                    }
                ],
            },
        ]
    )

    result = await provider._parse_response(response, tools=None)

    assert [chain.get_plain_text() for chain in result.reply_segments] == [
        "Let me check.",
        "AstrBot is a bot. \n\nSources:\n- Example: https://example.com/a",
    ]
    # The merged chain still carries the whole turn for history.
    assert result.completion_text.startswith("Let me check.\n\nAstrBot is a bot.")


@pytest.mark.asyncio
async def test_single_message_item_has_no_reply_segments():
    provider = _make_provider()
    response = _make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Done.", "annotations": []}
                ],
            }
        ]
    )

    result = await provider._parse_response(response, tools=None)

    assert result.reply_segments == []
    assert result.completion_text == "Done."


@pytest.mark.asyncio
async def test_query_stream_breaks_the_message_at_a_hosted_call(monkeypatch):
    provider = _make_provider()
    final_response = _make_response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "checking", "annotations": []}
                ],
            },
            {
                "type": "message",
                "id": "msg_2",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "answer", "annotations": []}
                ],
            },
        ]
    )

    async def fake_stream():
        yield SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1"),
        )
        yield SimpleNamespace(type="response.output_text.delta", delta="checking")
        yield SimpleNamespace(type="response.web_search_call.in_progress")
        yield SimpleNamespace(type="response.web_search_call.completed")
        yield SimpleNamespace(type="response.output_text.delta", delta="answer")
        yield SimpleNamespace(type="response.completed", response=final_response)

    async def fake_create(**kwargs):
        return fake_stream()

    monkeypatch.setattr(provider.client.responses, "create", fake_create)

    results = [
        result
        async for result in provider._query_stream(
            {"model": "gpt-test", "input": "hi", "store": False},
            tools=None,
        )
    ]

    chunk_chains = [result.result_chain for result in results if result.is_chunk]
    assert [chain.type for chain in chunk_chains] == [None, "break", None]
    assert results[-1].completion_text == "checking\n\nanswer"
    assert [chain.get_plain_text() for chain in results[-1].reply_segments] == [
        "checking",
        "answer",
    ]

