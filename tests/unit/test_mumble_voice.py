import asyncio

import pytest

from astrbot.core.platform.sources.mumble import voice
from astrbot.core.platform.sources.mumble.voice import VoiceOptions, VoiceSession


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def realtime_start(self, thread_id, request):
        self.calls.append("start")

    async def realtime_stop(self, thread_id):
        self.calls.append("stop")


class FakePump:
    def __init__(self) -> None:
        self.route = None

    def open_turn(self, tool_handler, approval_handler):
        queue: asyncio.Queue = asyncio.Queue()

        class Route:
            events = queue

        self.route = Route()
        return queue

    def close_turn(self):
        self.route = None


class FakeEngine:
    def __init__(self, gate: asyncio.Event) -> None:
        self.rt = FakeRuntime()
        self.gate = gate
        self.pumps: dict[str, FakePump] = {}
        self.forgotten: list[str] = []

    async def open_thread(self, state, params):
        await self.gate.wait()
        return {"thread_id": "t1", "rollout_path": None}, True

    def pump(self, thread_id):
        return self.pumps.setdefault(thread_id, FakePump())

    async def forget_thread(self, thread_id):
        self.forgotten.append(thread_id)


class FakeSp:
    async def get_async(self, **_kwargs):
        return {}

    async def put_async(self, **_kwargs):
        return None


@pytest.fixture
def engine(monkeypatch):
    gate = asyncio.Event()
    engine = FakeEngine(gate)

    async def codex_engine():
        return engine

    monkeypatch.setattr(voice, "_codex_engine", codex_engine)
    monkeypatch.setattr(voice, "sp", FakeSp())
    monkeypatch.setattr(voice, "_runner_config", lambda: {})
    return engine


def make_session(closed: list) -> VoiceSession:
    return VoiceSession(
        key="server",
        scope_id="test:voice:server",
        prompt="p",
        options=VoiceOptions(name="Jarvis", aliases=[]),
        send_audio=lambda frame, end: None,
        on_closed=closed.append,
    )


@pytest.mark.asyncio
async def test_close_during_start_leaves_nothing_running(engine):
    closed: list = []
    failures: list = []
    session = make_session(closed)
    session.launch(failures.append)
    await asyncio.sleep(0)  # the start is now waiting in open_thread

    closing = asyncio.create_task(session.close("disconnected"))
    await asyncio.sleep(0)
    assert not closing.done()  # waits for the start to reach a safe point
    engine.gate.set()
    await asyncio.wait_for(closing, 5)

    assert engine.rt.calls == []  # no realtime call was ever started
    assert engine.forgotten == ["t1"]  # the thread it opened is released
    assert closed == [session]
    assert failures == []
    await session.close("again")
    assert closed == [session]


@pytest.mark.asyncio
async def test_start_failure_is_reported_once(engine, monkeypatch):
    async def broken():
        raise RuntimeError("binding has no realtime support")

    monkeypatch.setattr(voice, "_codex_engine", broken)
    closed: list = []
    failures: list = []
    session = make_session(closed)
    session.launch(failures.append)
    for _ in range(10):
        await asyncio.sleep(0)
    assert [str(e) for e in failures] == ["binding has no realtime support"]
    assert closed == [session]


def test_voice_thread_config_follows_runner_limits(monkeypatch):
    monkeypatch.setattr(voice, "_runner_config", lambda: {"memory_enabled": False})
    config = voice.voice_thread_config("mumble:GroupMessage:server")
    assert config == {
        "model_tool_mode": "direct",
        "features.apps": False,
        "features.memories": False,
        "features.image_generation": False,
        "agents.enabled": False,
    }


def test_voice_thread_reads_paired_and_global_memories(monkeypatch):
    monkeypatch.setattr(
        voice,
        "_runner_config",
        lambda: {"memory_enabled": True, "memory_auto_consolidate": False},
    )
    config = voice.voice_thread_config("mumble:FriendMessage:abc")
    assert config["features.memories"] is True
    assert config["memories.scope_key"] == "mumble:FriendMessage:abc"
    # Reads global memories, never writes or deletes them.
    assert config["memories.may_write_global"] is False
    assert config["memories.may_delete"] is False
    assert config["memories.auto_consolidate"] is False
    assert config["features.apps"] is False
    assert config["agents.enabled"] is False


@pytest.mark.asyncio
async def test_close_while_waiting_for_answer_is_prompt(engine):
    engine.gate.set()
    closed: list = []
    session = make_session(closed)
    session.launch(lambda exc: None)
    for _ in range(1000):  # wait until the realtime start was requested
        if engine.rt.calls:
            break
        await asyncio.sleep(0.01)
    assert engine.rt.calls == ["start"]
    loop = asyncio.get_running_loop()
    began = loop.time()
    await asyncio.wait_for(session.close("standby"), 5)
    assert loop.time() - began < 2  # not the 30 s answer timeout
    assert engine.rt.calls == ["start", "stop"]
    assert engine.forgotten == ["t1"]
    assert closed == [session]


@pytest.mark.asyncio
async def test_cancelled_close_still_releases(engine):
    engine.gate.set()
    closed: list = []
    session = make_session(closed)
    session.launch(lambda exc: None)
    for _ in range(1000):
        if engine.rt.calls:
            break
        await asyncio.sleep(0.01)
    caller = asyncio.create_task(session.close("disconnected"))
    await asyncio.sleep(0)
    caller.cancel()
    for _ in range(300):
        if closed:
            break
        await asyncio.sleep(0.01)
    assert closed == [session]
    assert engine.rt.calls == ["start", "stop"]
    await session.close("again")  # already released: returns at once
