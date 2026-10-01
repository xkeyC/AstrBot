import asyncio
import base64
import json
import wave
from types import SimpleNamespace

import av
import pytest
import pytest_asyncio
from aiortc.mediastreams import MediaStreamError
from sqlmodel import select

from astrbot.core.config.agent_runner import get_agent_runner_config_default
from astrbot.core.db.po import ProviderStat
from astrbot.core.voice import chat as chat_module
from astrbot.core.voice import infra as infra_module
from astrbot.core.voice import record as record_module
from astrbot.core.voice import session as voice
from astrbot.core.voice.chat import VOICE_SESSIONS, VoiceChat
from astrbot.core.voice.infra import (
    INFRA_INSTRUCTIONS,
    InfraVoiceSession,
    SpeechTrack,
    infra_settings_error,
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
        # The voice thread's model and cumulative usage (thread_usage).
        self.usage: dict = {
            "model": "deepseek-v4.1-flash",
            "model_provider": "deepseek",
            "total_token_usage": None,
        }

    async def realtime_start(self, thread_id, request):
        self.started.append(json.loads(request))

    async def realtime_append_audio(self, thread_id, frame):
        self.audio.append(json.loads(frame))

    async def realtime_append_text(self, thread_id, text, role="user"):
        self.texts.append((text, role))

    async def realtime_stop(self, thread_id):
        self.stopped += 1

    async def thread_usage(self, thread_id):
        return json.dumps(self.usage)


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
        self.values: dict[str, dict] = {}

    async def get_async(self, **kwargs):
        return self.values.get(kwargs["key"], {})

    async def put_async(self, **kwargs):
        self.keys.append(kwargs["key"])
        self.values[kwargs["key"]] = kwargs["value"]


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


@pytest_asyncio.fixture
async def voice_db(monkeypatch, temp_db):
    """The voice records' database, ready (its first use is slow)."""
    async with temp_db.get_db():
        pass
    monkeypatch.setattr(record_module, "db_helper", temp_db)
    return temp_db


@pytest.fixture
def engine(monkeypatch, voice_db):
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
    kwargs = {
        "key": "k",
        "scope_id": "s",
        "prompt": "p",
        "options": VoiceOptions(name="J", aliases=[]),
        "media": FakeMedia(),
        "on_closed": lambda s: None,
        "chat": FakeChat(True),
    }
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
        "tts_stream_text": True,
    }
    assert "realtime.local_infra.ref_text" not in config
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
    # Codex fills in the time of the hang-up.
    assert "{now}" in request["realtime_end_instructions"]
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
async def test_the_voice_and_its_transcript_go_to_the_server(
    engine, monkeypatch, tmp_path
):
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"RIFF")
    monkeypatch.setattr(
        voice,
        "_runner_config",
        lambda: runner_config(
            ref_audio=str(ref), ref_text=" 你好，我是小乐。 ", stream_text=False
        ),
    )
    t = await open_session(engine)
    config = engine.params["config"]
    assert config["realtime.local_infra.ref_audio_path"] == str(ref)
    assert config["realtime.local_infra.ref_text"] == "你好，我是小乐。"
    assert config["realtime.local_infra.session"]["tts_stream_text"] is False
    await t.session.close("done")


@pytest.mark.asyncio
async def test_a_reference_text_without_its_audio_is_left_out(engine, monkeypatch):
    monkeypatch.setattr(
        voice, "_runner_config", lambda: runner_config(ref_text="你好。")
    )
    t = await open_session(engine)
    assert "realtime.local_infra.ref_text" not in engine.params["config"]
    await t.session.close("done")


def test_a_transcribed_reference_must_be_short(tmp_path):
    def silence(seconds: int) -> str:
        path = tmp_path / f"{seconds}s.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(bytes(2 * 8000 * seconds))
        return str(path)

    long = runner_config(ref_audio=silence(16), ref_text="你好。")["realtime_voice"]
    assert "at most 15 s" in infra_settings_error(long)
    long["ref_text"] = ""
    assert infra_settings_error(long) is None
    short = runner_config(ref_audio=silence(5), ref_text="你好。")["realtime_voice"]
    assert infra_settings_error(short) is None


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
async def test_a_conversation_closed_while_starting_fails_the_start(engine):
    t = await open_session(engine, started=False)
    await t.pump.queue.put(
        {"type": "realtime_conversation_closed", "reason": "transport_closed"}
    )
    await eventually(lambda: t.failures and t.closed)
    assert len(t.failures) == 1
    assert "transport_closed" in str(t.failures[0])


@pytest.mark.asyncio
async def test_the_session_ends_with_the_conversation(engine):
    t = await open_session(engine)
    await t.pump.queue.put(
        {"type": "realtime_conversation_closed", "reason": "transport_closed"}
    )
    await eventually(lambda: t.closed)
    assert t.session.closing


@pytest.mark.asyncio
async def test_a_provider_without_its_model_fails_the_start(engine, monkeypatch):
    monkeypatch.setattr(voice, "_runner_config", lambda: runner_config(text_model=""))
    t = await open_session(engine, started=False)
    await eventually(lambda: t.failures)
    assert "voice text model" in str(t.failures[0])


@pytest.mark.asyncio
async def test_a_server_that_fails_before_starting_is_reported(engine):
    t = await open_session(engine, started=False)
    # Codex reports the failure and ends the conversation at once.
    await event(t, {"Error": "the voice server closed the connection"})
    await t.pump.queue.put({"type": "realtime_conversation_closed", "reason": "error"})
    await eventually(lambda: t.failures)
    assert "closed the connection" in str(t.failures[0])
    await eventually(lambda: t.closed)


@pytest.mark.asyncio
async def test_a_bad_event_does_not_end_the_conversation(engine):
    t = await open_session(engine)
    await event(t, {"AudioOut": {"data": "!!not base64!!", "sample_rate": 24000}})
    await t.pump.queue.put(
        {"type": "realtime_conversation_closed", "reason": "requested"}
    )
    await eventually(lambda: t.closed)


def _total(input_tokens, cached, output):
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output,
        "reasoning_output_tokens": 0,
        "total_tokens": input_tokens + output,
    }


async def _stat_rows(db):
    async with db.get_db() as session:
        result = await session.execute(select(ProviderStat))
        return result.scalars().all()


@pytest.mark.asyncio
async def test_a_call_leaves_its_transcript_and_stats(engine, voice_db):
    engine.rt.usage["total_token_usage"] = _total(1000, 800, 50)
    t = await open_session(engine, private=False)
    await event(t, {"InputTranscriptDone": {"text": "what time is it"}})
    await t.pump.queue.put({"type": "task_started", "turn_id": "u1"})
    await t.pump.queue.put(
        {"type": "token_count", "info": {"total_token_usage": _total(3000, 2500, 90)}}
    )
    await t.pump.queue.put(
        {"type": "task_complete", "turn_id": "u1", "time_to_first_token_ms": 400}
    )
    await event(t, {"OutputTranscriptDone": {"text": "It is three."}})
    await event(t, {"InputTranscriptDone": {"text": "thanks"}})
    await t.session.close("hung up")

    # Its own conversation under the chat, which stays the chat's choice.
    state = voice.sp.values["mumble_voice_thread_infra"]
    conv = await voice_db.get_conversation_by_id(cid=state["conversation_id"])
    assert (conv.user_id, conv.title) == ("test:GroupMessage:server", "Voice: voice")
    assert conv.content == [
        {"role": "user", "content": "what time is it"},
        {"role": "assistant", "content": "It is three."},
        {"role": "user", "content": "thanks"},
    ]
    [row] = await _stat_rows(voice_db)
    assert (
        row.agent_type,
        row.umo,
        row.conversation_id,
        row.provider_id,
        row.provider_model,
        row.status,
        row.token_input_other,
        row.token_input_cached,
        row.token_output,
        row.time_to_first_token,
    ) == (
        "codex_voice",
        "test:GroupMessage:server",
        conv.conversation_id,
        "deepseek",
        "deepseek-v4.1-flash",
        "completed",
        300,
        1700,
        40,
        0.4,
    )


@pytest.mark.asyncio
async def test_the_next_call_keeps_the_conversation_and_a_cut_turn_counts(
    engine, voice_db
):
    t = await open_session(engine)
    await event(t, {"InputTranscriptDone": {"text": "hello"}})
    await t.session.close("hung up")
    cid = voice.sp.values["mumble_voice_thread_infra"]["conversation_id"]

    engine.rt.started.clear()
    t = await open_session(engine)
    await t.pump.queue.put({"type": "task_started", "turn_id": "u2"})
    await t.pump.queue.put(
        {"type": "token_count", "info": {"total_token_usage": _total(500, 0, 20)}}
    )
    await eventually(lambda: t.pump.queue.empty())
    await t.session.close("hung up")

    assert voice.sp.values["mumble_voice_thread_infra"]["conversation_id"] == cid
    conv = await voice_db.get_conversation_by_id(cid=cid)
    assert conv.content == [{"role": "user", "content": "hello"}]
    [row] = await _stat_rows(voice_db)
    assert (row.status, row.token_input_other, row.token_output) == ("aborted", 500, 20)
