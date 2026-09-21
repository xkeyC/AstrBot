from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.pipeline.process_stage.method.agent_sub_stages import third_party
from astrbot.core.provider.entities import LLMResponse


def _stage_ctx(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        astrbot_config=config,
        plugin_manager=SimpleNamespace(
            context=SimpleNamespace(
                conversation_manager=MagicMock(),
                persona_manager=MagicMock(),
            )
        ),
    )


def _provider_settings() -> dict:
    return {
        "streaming_response": False,
        "unsupported_streaming_strategy": "turn_off",
        "third_party_stream_consumption_close_timeout_sec": 30,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner_type", ["local", "dify", "coze", "dashscope", "deerflow", "unknown"]
)
async def test_third_party_stage_rejects_non_codex_runner(runner_type: str):
    stage = third_party.ThirdPartyAgentSubStage()
    config = {
        "agent_runner": {"runner_type": runner_type, "config": {}},
        "provider_settings": _provider_settings(),
    }

    with pytest.raises(ValueError, match="Unsupported third party agent runner"):
        await stage.initialize(_stage_ctx(config))


@pytest.mark.asyncio
async def test_codex_runner_receives_inline_profile_config(
    monkeypatch: pytest.MonkeyPatch,
):
    inline_config = {"model": "inline-model", "tool_call_timeout": 60}
    runner = MagicMock()
    runner.reset = AsyncMock()
    runner.get_final_llm_resp.return_value = LLMResponse(
        role="assistant",
        result_chain=MessageChain().message("done"),
    )
    runner.close = AsyncMock()

    async def step_until_done(max_step: int = 30):
        _ = max_step
        if False:
            yield None

    runner.step_until_done = step_until_done
    runner_factory_calls = []

    class RunnerFactory:
        @classmethod
        def __class_getitem__(cls, item):
            _ = item
            return cls

        def __new__(cls):
            runner_factory_calls.append(True)
            return runner

    monkeypatch.setattr(third_party, "CodexAgentRunner", RunnerFactory)
    monkeypatch.setattr(third_party, "prepare_codex_request", AsyncMock())
    monkeypatch.setattr(third_party, "try_steer", AsyncMock(return_value=None))
    monkeypatch.setattr(third_party, "build_turn_input", MagicMock(return_value=[]))
    registry = MagicMock()
    monkeypatch.setattr(third_party, "active_event_registry", registry)
    monkeypatch.setattr(
        third_party, "AstrAgentContext", MagicMock(return_value=object())
    )
    monkeypatch.setattr(
        third_party, "AgentContextWrapper", MagicMock(return_value=object())
    )
    monkeypatch.setattr(third_party, "call_event_hook", AsyncMock(return_value=False))
    monkeypatch.setattr(third_party.Metric, "upload", AsyncMock(return_value=None))

    config = {
        "agent_runner": {"runner_type": "codex", "config": inline_config},
        "provider_settings": _provider_settings(),
    }
    stage = third_party.ThirdPartyAgentSubStage()
    await stage.initialize(_stage_ctx(config))
    stage._resolve_persona_custom_error_message = AsyncMock(return_value=None)
    event = MagicMock()
    event.message_str = "hello"
    event.unified_msg_origin = "webchat:FriendMessage:test"
    event.message_obj.message = []
    event.platform_meta.support_streaming_message = True
    event.get_extra.return_value = None

    results = [item async for item in stage.process(event, "")]

    assert results == [None]
    assert runner.reset.await_args.kwargs["provider_config"] is inline_config
    assert runner_factory_calls == [True]
    registry.register_agent_stop_callback.assert_called_once()
    registry.unregister_agent_stop_callback.assert_called_once_with(event)


class _StatsRunner:
    """Yields a stats chunk, then the reply."""

    def __init__(self):
        self.stats_chain = MessageChain(type="agent_stats")
        self.reply = MessageChain().message("hi")

    async def step_until_done(self, max_step=30):
        from astrbot.core.agent.response import AgentResponse

        yield AgentResponse(type="agent_stats", data={"chain": self.stats_chain})
        yield AgentResponse(type="llm_result", data={"chain": self.reply})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("platform", "sent"), [("webchat", True), ("aiocqhttp", False)]
)
async def test_run_stats_reach_webchat_only(platform: str, sent: bool):
    runner = _StatsRunner()
    event = MagicMock()
    event.get_platform_name.return_value = platform
    event.send = AsyncMock()

    chunks = [
        chain
        async for chain, _ in third_party.run_third_party_agent(
            runner, stream_to_general=True, event=event
        )
    ]

    # The stats never become part of the reply.
    assert chunks == [runner.reply]
    if sent:
        event.send.assert_awaited_once_with(runner.stats_chain)
    else:
        event.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_stats_send_does_not_cost_the_reply():
    runner = _StatsRunner()
    event = MagicMock()
    event.get_platform_name.return_value = "webchat"
    event.send = AsyncMock(side_effect=ConnectionError("client gone"))

    results = [
        pair
        async for pair in third_party.run_third_party_agent(
            runner, stream_to_general=True, event=event
        )
    ]

    assert results == [(runner.reply, False)]
