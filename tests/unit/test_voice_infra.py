import asyncio
import base64
import json
from types import SimpleNamespace

import av
import pytest
from aiortc.mediastreams import MediaStreamError

from astrbot.core.config.agent_runner import get_agent_runner_config_default
from astrbot.core.voice import chat as chat_module
from astrbot.core.voice import infra as infra_module
from astrbot.core.voice import session as voice
from astrbot.core.voice.chat import VOICE_SESSIONS, VoiceChat
from astrbot.core.voice.infra import (
    INFRA_INSTRUCTIONS,
    InfraVoiceSession,
    SpeechTrack,
)
from astrbot.core.voice.session import VoiceOptions, VoiceSession, new_voice_session


class FakeChat(VoiceChat):
    """The real ordering and busy logic; the chat's turns are faked."""

    def __init__(self, private: bool) -> None:
        super().__init__(
            umo="test:FriendMessage:1" if private else "test:GroupMessage:server",
            private=private,
            sender_name="Alice" if private else "Voice",
        )
        self.asked: list[str] = []
        self.answer: str | None = "It is three."
        self.persona = "Speak like a pirate."

    async def voice_persona(self) -> str:
        return self.persona

    async def ask(self, body: str) -> str | None:
        self.asked.append(body)
        return self.answer


class Input:
    """Platform audio: one 20 ms frame of 48 kHz stereo, then nothing."""

    def __init__(self) -> None:
        self.sent = False

    async def recv(self):
        if self.sent:
            await asyncio.Event().wait()
        self.sent = True
        frame = av.AudioFrame(format="s16", layout="stereo", samples=960)
        frame.planes[0].update(bytes(960 * 4))
        frame.sample_rate = 48000
        frame.pts = 0
        return frame


class FakeMedia:
    def __init__(self) -> None:
        self.flushed = 0
        self.played: list = []
        self.track = Input()

    async def play(self, track) -> None:
        self.played.append(track)
        await asyncio.Event().wait()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def flush(self) -> None:
        self.flushed += 1


class FakeRuntime:
    def __init__(self) -> None:
        self.started: list[dict] = []
        self.audio: list[dict] = []
        self.texts: list[tuple[str, str]] = []
        self.stopped = 0

    async def realtime_start(self, thread_id, request):
        self.started.append(json.loads(request))

    async def realtime_append_audio(self, thread_id, frame):
        self.audio.append(json.loads(frame))

    async def realtime_append_text(self, thread_id, text, role="user"):
        self.texts.append((text, role))

    async def realtime_stop(self, thread_id):
        self.stopped += 1


class FakePump:
    def __init__(self) -> None:
        self.route = None
        self.tool_handler = None
        self.queue: asyncio.Queue | None = None

    def open_turn(self, tool_handler, approval_handler):
        self.tool_handler = tool_handler
        self.queue = asyncio.Queue()
        queue = self.queue

        class Route:
            events = queue

        self.route = Route()
        return queue

    def close_turn(self):
        self.route = None


class FakeEngine:
    def __init__(self) -> None:
        self.rt = FakeRuntime()
        self.pumps: dict[str, FakePump] = {}
        self.params: dict | None = None
        self.forgotten: list[str] = []
        self.locks: dict[str, asyncio.Lock] = {}

    def session_lock(self, key):
        return self.locks.setdefault(key, asyncio.Lock())

    async def open_thread(self, state, params):
        self.params = params
        return {"thread_id": "t1", "rollout_path": None}, True

    def pump(self, thread_id):
        return self.pumps.setdefault(thread_id, FakePump())

    async def forget_thread(self, thread_id):
        self.forgotten.append(thread_id)


class FakeSp:
    def __init__(self) -> None:
        self.keys: list[str] = []

    async def get_async(self, **_kwargs):
        return {}

    async def put_async(self, **kwargs):
        self.keys.append(kwargs["key"])


def runner_config(**voice_settings) -> dict:
    config = get_agent_runner_config_default("codex")
    config["realtime_voice"].update(
        {
            "backend": "local_infra",
            "infra_url": "ws://127.0.0.1:17890/v1/realtime",
            "infra_token": "secret",
            "emotion": "happy",
            "emotion_strength": 0.5,
            "text_model_provider": "deepseek",
            "text_model": "deepseek-v4.1-flash",
            "text_reasoning_effort": "none",
            **voice_settings,
        }
    )
    return config


@pytest.fixture
def engine(monkeypatch):
    engine = FakeEngine()

    async def codex_engine():
        return engine

    monkeypatch.setattr(voice, "_codex_engine", codex_engine)
    monkeypatch.setattr(voice, "sp", FakeSp())
    monkeypatch.setattr(voice, "_runner_config", runner_config)
    return engine


async def eventually(check) -> None:
    for _ in range(200):
        if check():
            return
        await asyncio.sleep(0.01)
    assert check()


async def open_session(engine, private: bool = True, started: bool = True):
    t = SimpleNamespace(
        media=FakeMedia(), chat=FakeChat(private), closed=[], failures=[]
    )
    t.session = new_voice_session(
        key="server",
        scope_id="test:voice:server",
        prompt="You are Jarvis, on a call.",
        options=VoiceOptions(name="Jarvis", aliases=["Jar"]),
        media=t.media,
        on_closed=t.closed.append,
        chat=t.chat,
        thread_key="mumble_voice_thread",
    )
    t.session.launch(t.failures.append)
    await eventually(lambda: engine.rt.started or t.failures)
    t.pump = engine.pumps.get("t1")
    if started and t.pump is not None:
        await t.pump.queue.put({"type": "realtime_conversation_started"})
        await eventually(lambda: t.session.ready)
    return t


async def event(t, payload: dict) -> None:
    await t.pump.queue.put(
        {"type": "realtime_conversation_realtime", "payload": payload}
    )


def test_the_configured_backend_picks_the_session(monkeypatch):
    monkeypatch.setattr(voice, "_runner_config", runner_config)
    kwargs = dict(
        key="k",
        scope_id="s",
        prompt="p",
        options=VoiceOptions(name="J", aliases=[]),
        media=FakeMedia(),
        on_closed=lambda s: None,
        chat=FakeChat(True),
    )
    assert type(new_voice_session(**kwargs)) is InfraVoiceSession
    monkeypatch.setattr(
        voice, "_runner_config", lambda: runner_config(backend="builtin")
    )
    assert type(new_voice_session(**kwargs)) is VoiceSession


@pytest.mark.asyncio
async def test_the_voice_thread_talks_on_the_chosen_model(engine):
    t = await open_session(engine, private=False)
    params = engine.params
    assert params["base_instructions"].startswith(INFRA_INSTRUCTIONS)
    # The platform's prompt and the voice persona, but not the time: the
    # prefix stays the same from call to call.
    assert "You are Jarvis, on a call." in params["base_instructions"]
    assert params["base_instructions"].endswith("Speak like a pirate.")
    assert "Today is" not in params["base_instructions"]
    assert [tool["name"] for tool in params["dynamic_tools"]] == ["backend_task"]
    config = params["config"]
    assert config["realtime.backend"] == "local_multimodal_infra"
    assert config["realtime.local_infra.url"] == "ws://127.0.0.1:17890/v1/realtime"
    assert config["realtime.local_infra.token"] == "secret"
    assert config["realtime.local_infra.session"] == {
        "name": "Jarvis",
        "aliases": ["Jar"],
        "group": True,
        "tts_emotion": "happy",
        "tts_emotion_strength": 0.5,
    }
    assert config["model_provider"] == "deepseek"
    assert config["model"] == "deepseek-v4.1-flash"
    assert config["model_reasoning_effort"] == "none"
    assert config["model_tool_mode"] == "direct"
    assert "realtime.host_routes_handoffs" not in config
    # Its own thread, apart from the realtime one.
    assert voice.sp.keys == ["mumble_voice_thread_infra"]
    request = engine.rt.started[0]
    assert request["transport"] == {"type": "websocket"}
    assert "Today is" in request["realtime_start_instructions"]
    assert VOICE_SESSIONS[t.chat.umo] is t.session
    await t.session.close("done")


@pytest.mark.asyncio
async def test_audio_goes_both_ways(engine):
    t = await open_session(engine)
    # Platform audio reaches Codex as 16 kHz mono.
    await eventually(lambda: engine.rt.audio)
    frame = engine.rt.audio[0]
    assert frame["sample_rate"] == 16000 and frame["num_channels"] == 1
    assert len(base64.b64decode(frame["data"])) == frame["samples_per_channel"] * 2
    # The server's speech is played.
    track = t.media.played[0]
    assert isinstance(track, SpeechTrack)
    pcm = bytes(range(10)) * 48
    await event(
        t,
        {
            "AudioOut": {
                "data": base64.b64encode(pcm).decode(),
                "sample_rate": 24000,
                "num_channels": 1,
            }
        },
    )
    played = await asyncio.wait_for(track.recv(), 2)
    assert bytes(played.planes[0])[: len(pcm)] == pcm
    # Talked over: what is buffered goes.
    await event(t, {"ResponseCancelled": {"response_id": "m1"}})
    await eventually(lambda: t.media.flushed == 1)
    await t.session.close("done")
    assert engine.rt.stopped == 1
    with pytest.raises(MediaStreamError):
        await track.recv()


@pytest.mark.asyncio
async def test_a_backend_task_runs_in_the_chat_and_its_answer_is_told(engine):
    t = await open_session(engine)
    await event(t, {"InputTranscriptDone": {"text": "what time is it"}})
    await eventually(lambda: t.session._heard)
    result = await t.pump.tool_handler(
        {"tool": "backend_task", "callId": "c1", "arguments": {"task": "Tell the time"}}
    )
    assert result["success"] is True
    assert "later" in result["contentItems"][0]["text"]
    await eventually(lambda: engine.rt.texts)
    assert t.chat.asked == [
        chat_module.TASK_BODY.format(heard="what time is it", task="Tell the time")
    ]
    text, role = engine.rt.texts[0]
    assert text == infra_module.RESULT_PROMPT.format(
        task="Tell the time", answer="It is three."
    )
    assert role == "developer"
    unknown = await t.pump.tool_handler({"tool": "shell", "arguments": {}})
    assert unknown["success"] is False
    await t.session.close("done")


@pytest.mark.asyncio
async def test_bad_settings_fail_the_start(engine, monkeypatch):
    monkeypatch.setattr(
        voice, "_runner_config", lambda: runner_config(infra_url="http://x")
    )
    t = await open_session(engine, started=False)
    await eventually(lambda: t.failures)
    assert "ws://" in str(t.failures[0])
    await eventually(lambda: t.closed)


@pytest.mark.asyncio
async def test_a_failed_server_start_fails_the_session(engine):
    t = await open_session(engine, started=False)
    await event(t, {"Error": "cannot connect to ws://127.0.0.1:17890/v1/realtime"})
    await eventually(lambda: t.failures)
    assert "cannot connect" in str(t.failures[0])


@pytest.mark.asyncio
async def test_the_session_ends_with_the_conversation(engine):
    t = await open_session(engine)
    await t.pump.queue.put(
        {"type": "realtime_conversation_closed", "reason": "transport_closed"}
    )
    await eventually(lambda: t.closed)
    assert t.session.closing
