from types import SimpleNamespace

import pytest

from astrbot.builtin_stars.builtin_commands.commands import (
    conversation as conversation_module,
)
from astrbot.core.agent.runners.codex.constants import (
    CODEX_RUNNER_TYPE,
    CODEX_THREAD_STATE_KEY,
)


def test_only_codex_runner_has_third_party_session_state():
    assert conversation_module.THIRD_PARTY_AGENT_RUNNER_KEY == {
        CODEX_RUNNER_TYPE: CODEX_THREAD_STATE_KEY,
    }


@pytest.mark.asyncio
async def test_clear_third_party_agent_runner_state_removes_codex_thread_state(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[object] = []

    async def fake_remove_async(*args, **kwargs):
        _ = args
        calls.append(("remove", kwargs["scope"], kwargs["scope_id"], kwargs["key"]))

    monkeypatch.setattr(conversation_module.sp, "remove_async", fake_remove_async)

    await conversation_module._clear_third_party_agent_runner_state(
        SimpleNamespace(),
        "umo-1",
        CODEX_RUNNER_TYPE,
    )

    assert calls == [("remove", "umo", "umo-1", CODEX_THREAD_STATE_KEY)]


@pytest.mark.asyncio
@pytest.mark.parametrize("runner_type", ["local", "dify", "coze", "deerflow"])
async def test_clear_third_party_agent_runner_state_ignores_removed_runners(
    monkeypatch: pytest.MonkeyPatch,
    runner_type: str,
):
    calls: list[object] = []

    async def fake_remove_async(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(conversation_module.sp, "remove_async", fake_remove_async)

    await conversation_module._clear_third_party_agent_runner_state(
        SimpleNamespace(),
        "umo-1",
        runner_type,
    )

    assert calls == []
