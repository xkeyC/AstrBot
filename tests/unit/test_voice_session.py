import asyncio

import pytest

from astrbot.core.platform.sources.mumble import voice as mumble_voice
from astrbot.core.platform.sources.mumble.audio import MumbleMedia
from astrbot.core.voice import session as voice
from astrbot.core.voice.session import VoiceOptions, VoiceSession


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def realtime_start(self, thread_id, request):
        self.calls.append("start")

    async def realtime_append_text(self, thread_id, text):
        self.calls.append(f"text:{text}")

    async def realtime_append_speech(self, thread_id, text):
        self.calls.append(f"speech:{text}")

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
        self.locks: dict[str, asyncio.Lock] = {}

    def session_lock(self, key):
        return self.locks.setdefault(key, asyncio.Lock())

    async def open_thread(self, state, params):
        await self.gate.wait()
        return {"thread_id": "t1", "rollout_path": None}, True

    def pump(self, thread_id):
        return self.pumps.setdefault(thread_id, FakePump())

    async def forget_thread(self, thread_id):
        self.forgotten.append(thread_id)


class FakeChat:
    def __init__(self) -> None:
        self.asked: list[str] = []
        self.answer: str | None = "It is three."
        self.is_busy = False
        self.persona = ""
        self.release = asyncio.Event()
        self.release.set()

    def busy(self) -> bool:
        return self.is_busy

    async def voice_persona(self) -> str:
        return self.persona

    async def ask(self, body: str) -> str | None:
        self.asked.append(body)
        await self.release.wait()
        return self.answer


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
        media=MumbleMedia(lambda frame, end: None),
        on_closed=closed.append,
        chat=FakeChat(),
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
    async def broken(realtime=True):
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


@pytest.mark.asyncio
async def test_the_voice_thread_only_carries_the_conversation(engine):
    captured = {}

    async def open_thread(state, params):
        captured.update(params)
        return {"thread_id": "t1", "rollout_path": None}, True

    engine.open_thread = open_thread
    session = make_session([])
    await session._open_agent()
    assert captured["dynamic_tools"] == []
    assert captured["no_environment"] is True
    config = captured["config"]
    assert config["realtime.host_routes_handoffs"] is True
    assert config["features.memories"] is False


def handoff(text: str) -> dict:
    return {
        "handoff_id": "h1",
        "item_id": "i1",
        "input_transcript": text,
        "active_transcript": [
            {"role": "user", "text": "what time is it"},
            {"role": "assistant", "text": "let me check"},
        ],
    }


@pytest.mark.asyncio
async def test_a_handoff_is_answered_by_the_chat_and_spoken(engine):
    session = make_session([])
    session._engine, session._thread_id = engine, "t1"
    await session._handoff(handoff("look up the time"))
    (body,) = session.chat.asked
    assert "look up the time" in body and "what time is it" in body
    assert engine.rt.calls == ["speech:It is three."]


@pytest.mark.asyncio
async def test_a_busy_chat_is_announced_and_requests_keep_their_order(engine):
    session = make_session([])
    session._engine, session._thread_id = engine, "t1"
    session.chat.is_busy = True
    session.chat.release.clear()
    first = asyncio.create_task(session._handoff(handoff("first")))
    await asyncio.sleep(0)
    session.chat.is_busy = False
    second = asyncio.create_task(session._handoff(handoff("second")))
    await asyncio.sleep(0)
    # Both waiting are told so: the chat is busy, then the first one.
    assert engine.rt.calls == [f"speech:{voice.BUSY_SPEECH}"] * 2
    session.chat.release.set()
    await asyncio.gather(first, second)
    assert [b.split("Task: ")[1] for b in session.chat.asked] == ["first", "second"]
    assert engine.rt.calls[2:] == ["speech:It is three."] * 2


@pytest.mark.asyncio
async def test_an_unanswered_handoff_is_said_to_have_failed(engine):
    session = make_session([])
    session._engine, session._thread_id = engine, "t1"
    session.chat.answer = None
    await session._handoff(handoff("x"))
    assert engine.rt.calls == [f"speech:{voice.FAILED_SPEECH}"]


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


@pytest.mark.asyncio
async def test_start_slower_than_close_wait_is_released_later(engine, monkeypatch):
    monkeypatch.setattr(voice, "CLOSE_WAIT", 0.05)
    closed: list = []
    session = make_session(closed)
    session.launch(lambda exc: None)
    await asyncio.sleep(0)  # the start is stuck opening the thread
    await asyncio.wait_for(session.close("standby"), 5)
    assert closed == [session]
    assert engine.forgotten == []  # no thread yet
    engine.gate.set()  # the thread opens after the close gave up waiting
    for _ in range(100):
        if engine.forgotten:
            break
        await asyncio.sleep(0.01)
    assert engine.forgotten == ["t1"]  # released once the start returned
    assert engine.rt.calls == []


@pytest.mark.asyncio
async def test_late_start_cannot_unload_a_newer_sessions_thread(engine, monkeypatch):
    monkeypatch.setattr(voice, "CLOSE_WAIT", 0.05)
    old_closed: list = []
    old = make_session(old_closed)
    old.launch(lambda exc: None)
    await asyncio.sleep(0)  # stuck opening thread t1
    await asyncio.wait_for(old.close("muted"), 5)
    new_closed: list = []
    new = make_session(new_closed)  # same key, same persisted thread
    new.launch(lambda exc: None)
    await asyncio.sleep(0)
    engine.gate.set()
    for _ in range(1000):
        if engine.rt.calls:
            break
        await asyncio.sleep(0.01)
    # The old start unloaded t1 before the new one could open it; the new
    # session then opened (resumed) it and kept it.
    assert engine.forgotten == ["t1"]
    assert engine.rt.calls == ["start"]
    assert new_closed == []
    await new.close("done")
    assert engine.forgotten == ["t1", "t1"]


def test_consent_expiry_is_disabled():
    import aioice.ice

    assert aioice.ice.CONSENT_FAILURES >= 1_000_000


def test_realtime_prompts_state_the_date(monkeypatch):
    import datetime

    options = voice.VoiceOptions(name="Jarvis", aliases=["贾维斯"])
    today = datetime.datetime.now().astimezone().strftime("%Y-%m-%d")
    for prompt in (
        mumble_voice.channel_prompt(options),
        mumble_voice.whisper_prompt(options, "alice"),
    ):
        assert f"Today is {today}" in prompt
        assert "must be delegated to the backend" in prompt
    assert '("Jarvis", "贾维斯")' in mumble_voice.channel_prompt(options)


@pytest.mark.asyncio
async def test_say_needs_a_ready_session(engine):
    session = make_session([])
    with pytest.raises(RuntimeError):
        await session.say("hello")
    session.ready = True
    session._engine = engine
    session._thread_id = "t1"
    await session.say("hello")
    assert engine.rt.calls == ["text:hello"]


def test_close_stops_the_media(engine):
    class Media:
        track = None
        stopped = 0

        async def play(self, track):
            return None

        def start(self):
            return None

        def stop(self):
            self.stopped += 1

    media = Media()
    session = VoiceSession(
        key="k",
        scope_id="s",
        prompt="p",
        options=VoiceOptions(name="n", aliases=[]),
        media=media,
        on_closed=lambda s: None,
        chat=FakeChat(),
    )

    async def run():
        await session.close("a")
        await session.close("b")

    asyncio.run(run())
    assert media.stopped == 1


@pytest.mark.asyncio
async def test_the_voice_persona_completes_the_prompt(engine, monkeypatch):
    for persona, extra, expected in (
        ("Speak like a pirate.", "platform extra", "p\n\nSpeak like a pirate."),
        ("", "platform extra", "p\n\nplatform extra"),
        ("", "", "p"),
    ):
        session = VoiceSession(
            key="k",
            scope_id="s",
            prompt="p",
            options=VoiceOptions(name="n", aliases=[], extra_prompt=extra),
            media=MumbleMedia(lambda frame, end: None),
            on_closed=lambda s: None,
            chat=FakeChat(),
        )
        session.chat.persona = persona

        async def stop_here():
            raise RuntimeError("stop")

        monkeypatch.setattr(session, "_open_agent", stop_here)
        with pytest.raises(RuntimeError):
            await session._start()
        assert session.prompt == expected
