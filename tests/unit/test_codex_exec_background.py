"""后台 shell 会话的闭环：命令跑完要能回到群里，跑不完要被收掉。

背景：`exec_command` 到 yield 时间就返回一个 session ID，后续输出只能靠
模型自己用 `write_stdin` 轮询——没有任何推送。模型一旦直接回答就结束了
回合，那条命令的结果永远没人收，tmux 面板也一直留在沙箱里。
"""

import asyncio

import pytest

from astrbot.core.tools.computer_tools import codex_exec
from astrbot.core.tools.computer_tools.codex_exec import (
    MAX_LIVE_SESSIONS_PER_CHAT,
    SandboxSession,
    SandboxSessions,
    claim_completion,
)

UMO = "aiocqhttp:GroupMessage:30003"


@pytest.fixture(autouse=True)
def _clean():
    SandboxSessions._sessions.clear()
    codex_exec._claimed.clear()
    codex_exec._watchers.clear()
    yield
    SandboxSessions._sessions.clear()
    codex_exec._claimed.clear()
    codex_exec._watchers.clear()


def test_only_one_side_reports_a_completion():
    """agent 轮询到退出和 watcher 观察到退出，只能有一个人播报。"""
    assert claim_completion(UMO, "sess-1") is True
    assert claim_completion(UMO, "sess-1") is False
    # 不同会话互不影响
    assert claim_completion(UMO, "sess-2") is True


def test_claim_registry_stays_bounded():
    """长期运行的 bot 不能因为这个表无限增长。"""
    for i in range(codex_exec._CLAIMED_LIMIT + 50):
        claim_completion(UMO, f"sess-{i}")

    assert len(codex_exec._claimed) <= codex_exec._CLAIMED_LIMIT
    # 最早的已被淘汰，可以再次被认领——只是不会重复播报同一批
    assert claim_completion(UMO, "sess-0") is True


def test_session_count_is_per_chat():
    SandboxSessions.add(UMO, SandboxSession(session_id="a", directory="/d/a"))
    SandboxSessions.add(UMO, SandboxSession(session_id="b", directory="/d/b"))
    SandboxSessions.add("other", SandboxSession(session_id="c", directory="/d/c"))

    assert SandboxSessions.count(UMO) == 2
    assert SandboxSessions.count("other") == 1
    assert SandboxSessions.count("nobody") == 0

    SandboxSessions.drop(UMO, "a")
    assert SandboxSessions.count(UMO) == 1


def test_the_live_session_cap_is_low_enough_to_matter():
    """上限存在的意义是先于沙箱的进程限制拦下来。"""
    assert 0 < MAX_LIVE_SESSIONS_PER_CHAT <= 16


@pytest.mark.asyncio
async def test_watch_background_session_starts_one_task_per_session(monkeypatch):
    started: list[str] = []

    async def _fake_watch(plugin_context, context, umo, session_id, sandbox):
        started.append(session_id)
        await asyncio.sleep(0)

    monkeypatch.setattr(codex_exec, "_watch", _fake_watch)

    class _Ctx:
        class context:  # noqa: N801 - mirrors ContextWrapper's shape
            context = object()

    codex_exec.watch_background_session(_Ctx(), UMO, "sess-1")
    codex_exec.watch_background_session(_Ctx(), UMO, "sess-1")
    codex_exec.watch_background_session(_Ctx(), UMO, "sess-2")
    await asyncio.gather(*list(codex_exec._watchers.values()))
    await asyncio.sleep(0)

    assert started == ["sess-1", "sess-2"], "同一个会话不该被重复监视"
    # 任务结束后要把自己从表里摘掉，否则同一个会话再也不会被监视
    assert codex_exec._watchers == {}
