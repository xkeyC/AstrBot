import pytest

import astrbot.core.provider.provider as provider_core
from astrbot.core.exceptions import EmptyModelOutputError
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.third_party import (
    _resolve_third_party_streaming_mode,
)
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.provider.provider import Provider


class _StreamingOnlyProvider(Provider):
    def __init__(self, *, emit_final: bool, emit_chunks: bool = True) -> None:
        super().__init__({"type": "test_provider"}, {})
        self.emit_final = emit_final
        self.emit_chunks = emit_chunks

    def get_current_key(self) -> str:
        return ""

    def set_key(self, key: str) -> None:
        return None

    async def get_models(self) -> list[str]:
        return []

    async def text_chat(self, **kwargs) -> LLMResponse:
        if self._should_force_streaming_text_chat():
            return await self._text_chat_from_stream_blocking(**kwargs)
        return LLMResponse(role="assistant", completion_text="non-stream")

    async def text_chat_stream(self, **kwargs):
        if not self.emit_chunks:
            if False:
                yield LLMResponse(role="assistant", is_chunk=True)
            return

        yield LLMResponse(role="assistant", completion_text="hel", is_chunk=True)
        yield LLMResponse(
            role="assistant",
            result_chain=MessageChain().message("lo"),
            is_chunk=True,
        )
        if self.emit_final:
            yield LLMResponse(role="assistant", completion_text="final")


@pytest.mark.asyncio
async def test_text_chat_uses_streaming_final_response(monkeypatch):
    monkeypatch.setattr(provider_core, "ENABLE_ALL_STREAMING_MODE", True)
    provider = _StreamingOnlyProvider(emit_final=True)

    response = await provider.text_chat()

    assert response.completion_text == "final"


@pytest.mark.asyncio
async def test_text_chat_aggregates_streaming_chunks_without_final(monkeypatch):
    monkeypatch.setattr(provider_core, "ENABLE_ALL_STREAMING_MODE", True)
    provider = _StreamingOnlyProvider(emit_final=False)

    response = await provider.text_chat()

    assert response.completion_text == "hello"


@pytest.mark.asyncio
async def test_text_chat_raises_when_streaming_has_no_usable_output(monkeypatch):
    monkeypatch.setattr(provider_core, "ENABLE_ALL_STREAMING_MODE", True)
    provider = _StreamingOnlyProvider(emit_final=False, emit_chunks=False)

    with pytest.raises(EmptyModelOutputError):
        await provider.text_chat()


def test_third_party_forces_upstream_streaming_but_suppresses_downstream_deltas(
    monkeypatch,
):
    monkeypatch.setattr(provider_core, "ENABLE_ALL_STREAMING_MODE", True)

    runner_streaming, suppress_deltas = _resolve_third_party_streaming_mode(
        streaming_response=False,
        stream_to_general=False,
    )

    assert runner_streaming is True
    assert suppress_deltas is True


def test_third_party_keeps_downstream_streaming_when_user_streaming_is_available(
    monkeypatch,
):
    monkeypatch.setattr(provider_core, "ENABLE_ALL_STREAMING_MODE", True)

    runner_streaming, suppress_deltas = _resolve_third_party_streaming_mode(
        streaming_response=True,
        stream_to_general=False,
    )

    assert runner_streaming is True
    assert suppress_deltas is False
