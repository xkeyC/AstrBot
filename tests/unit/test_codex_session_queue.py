"""群聊里的插队与排队：同发起人插进运行中的 turn，其他人排队等。

背景：一个会话对应一个 Codex thread，thread 一次只跑一个 turn。同一个人
追加的消息应该被 steer 进正在跑的 turn（合成一条回复），其他人的消息则
在 AstrBot 侧排队。
"""

import asyncio

import pytest

from astrbot.core.agent.runners.codex import native
from astrbot.core.agent.runners.codex.native import (
    ACTIVE_TURNS,
    ActiveTurn,
    SessionBusy,
    session_slot,
    try_steer,
)

UMO = "aiocqhttp:GroupMessage:30003"


class _FakeEngine:
    """只提供 session_slot / try_steer 用到的两个入口。"""

    def __init__(self, steer_status="steered"):
        self._locks: dict[str, asyncio.Lock] = {}
        self.steer_status = steer_status
        self.submitted: list[dict] = []

    def session_lock(self, umo):
        return self._locks.setdefault(umo, asyncio.Lock())

    async def submit_turn(self, thread_id, request):
        self.submitted.append(request)
        return {"status": self.steer_status, "turn_id": "turn-2"}


@pytest.fixture(autouse=True)
def _clean_registries():
    ACTIVE_TURNS.clear()
    native._QUEUED_TURNS.clear()
    yield
    ACTIVE_TURNS.clear()
    native._QUEUED_TURNS.clear()


def _active(engine, *, sender_id="20017", turn_id="turn-1"):
    turn = ActiveTurn(engine, "thread-1", turn_id, sender_id)
    if turn_id:
        turn.ready.set()
    ACTIVE_TURNS[UMO] = turn
    return turn


# ------------------------------------------------------------------ 排队


@pytest.mark.asyncio
async def test_session_slot_runs_one_turn_at_a_time():
    engine = _FakeEngine()
    order: list[str] = []

    async def turn(name, hold):
        async with session_slot(engine, UMO, 0):
            order.append(f"{name}-start")
            await asyncio.sleep(hold)
            order.append(f"{name}-end")

    await asyncio.gather(turn("a", 0.05), turn("b", 0))

    # 没有交错：b 必须等 a 结束
    assert order == ["a-start", "a-end", "b-start", "b-end"]


@pytest.mark.asyncio
async def test_session_slot_refuses_once_the_queue_is_deep():
    engine = _FakeEngine()
    release = asyncio.Event()
    entered = asyncio.Event()

    async def holder():
        async with session_slot(engine, UMO, 2):
            entered.set()
            await release.wait()

    async def waiter():
        async with session_slot(engine, UMO, 2):
            pass

    held = asyncio.create_task(holder())
    await entered.wait()
    queued = asyncio.create_task(waiter())
    await asyncio.sleep(0)  # 让 waiter 进入等待

    # 第三个应当被直接拒绝，而不是无声地排进去
    with pytest.raises(SessionBusy) as excinfo:
        async with session_slot(engine, UMO, 2):
            pass
    assert excinfo.value.waiting == 2

    release.set()
    await asyncio.gather(held, queued)
    # 计数在全部退出后必须归零，否则会话会永久“忙碌”
    assert UMO not in native._QUEUED_TURNS


@pytest.mark.asyncio
async def test_a_refused_turn_does_not_leak_a_queue_slot():
    """被拒绝的那一轮不能占住名额，否则会话会越来越容易“忙碌”。"""
    engine = _FakeEngine()
    release = asyncio.Event()
    entered = asyncio.Event()

    async def holder():
        async with session_slot(engine, UMO, 1):
            entered.set()
            await release.wait()

    held = asyncio.create_task(holder())
    await entered.wait()

    for _ in range(3):
        with pytest.raises(SessionBusy):
            async with session_slot(engine, UMO, 1):
                pass
    assert native._QUEUED_TURNS[UMO] == 1, "被拒绝的三次都不该计数"

    release.set()
    await held
    assert UMO not in native._QUEUED_TURNS


# ------------------------------------------------------------------ 插队


@pytest.mark.asyncio
async def test_same_sender_is_steered_into_the_running_turn():
    engine = _FakeEngine()
    _active(engine)

    target = await try_steer(UMO, "20017", [{"type": "text", "text": "追加"}])

    assert target is not None
    assert engine.submitted[0]["mode"] == "steer"
    assert engine.submitted[0]["expected_turn_id"] == "turn-1"


@pytest.mark.asyncio
async def test_another_sender_is_not_steered():
    engine = _FakeEngine()
    _active(engine, sender_id="20017")

    assert (
        await try_steer(UMO, "20019", [{"type": "text", "text": "别人的问题"}]) is None
    )
    assert engine.submitted == []


@pytest.mark.asyncio
async def test_follow_up_waits_for_a_turn_id_instead_of_queueing():
    """turn 已登记但 submit 还没返回时，追加消息应当等它，而不是变成第二轮。"""
    engine = _FakeEngine()
    turn = _active(engine, turn_id="")

    steering = asyncio.create_task(
        try_steer(UMO, "20017", [{"type": "text", "text": "追加"}])
    )
    await asyncio.sleep(0)
    assert not steering.done(), "不应在拿到 turn_id 之前就放弃"

    turn.turn_id = "turn-1"
    turn.ready.set()

    assert await steering is not None


@pytest.mark.asyncio
async def test_follow_up_gives_up_when_the_turn_failed_to_start():
    engine = _FakeEngine()
    turn = _active(engine, turn_id="")

    steering = asyncio.create_task(
        try_steer(UMO, "20017", [{"type": "text", "text": "追加"}])
    )
    await asyncio.sleep(0)
    turn.aborted = True
    turn.ready.set()

    assert await steering is None
    assert engine.submitted == []


@pytest.mark.asyncio
async def test_follow_up_does_not_wait_forever(monkeypatch):
    monkeypatch.setattr(native, "STEER_READY_TIMEOUT_S", 0.01)
    engine = _FakeEngine()
    _active(engine, turn_id="")

    assert await try_steer(UMO, "20017", [{"type": "text", "text": "追加"}]) is None
