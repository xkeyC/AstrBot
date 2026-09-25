import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
import websockets
from aiortc.mediastreams import MediaStreamError

from astrbot.core.voice import cascade as cascade_module
from astrbot.core.voice import chat as chat_module
from astrbot.core.voice.cascade import CascadeOptions, CascadeVoiceSession, SpeechTrack
from astrbot.core.voice.chat import VOICE_SESSIONS, VoiceChat
from astrbot.core.voice.session import VoiceOptions


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
        self.is_busy = False

    def busy(self) -> bool:
        return self.is_busy or super().busy()

    async def voice_persona(self) -> str:
        return self.persona

    async def ask(self, body: str) -> str | None:
        self.asked.append(body)
        return self.answer


class FakeMedia:
    """No input; records what the session does with the output."""

    def __init__(self) -> None:
        self.flushed = 0
        self.played = asyncio.Event()

        class Silent:
            async def recv(self):
                await asyncio.Event().wait()

        self.track = Silent()

    async def play(self, track) -> None:
        self.played.set()
        await asyncio.Event().wait()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def flush(self) -> None:
        self.flushed += 1


class FakeServer:
    """A /v1/realtime server that records what it gets; the test sends
    through ``socket``."""

    def __init__(self, started: dict | None = None) -> None:
        self.received: asyncio.Queue = asyncio.Queue()
        self.log: list = []  # every event, in order
        self.headers = None
        self.socket = None
        self.started = started or {
            "type": "session.started",
            "input_rate": 16000,
            "output_rate": 24000,
        }

    async def handler(self, socket) -> None:
        self.headers = socket.request.headers
        self.socket = socket
        async for message in socket:
            event = json.loads(message) if isinstance(message, str) else message
            self.log.append(event)
            await self.received.put(event)
            if isinstance(event, dict) and event.get("type") == "session.start":
                await socket.send(json.dumps(self.started))

    async def next(self, kind: str) -> dict:
        while True:
            event = await asyncio.wait_for(self.received.get(), 5)
            if isinstance(event, dict) and event.get("type") == kind:
                return event


async def open_session(
    tmp_path, private: bool, started: dict | None = None, instructions: str = ""
):
    server = FakeServer(started)
    serving = await websockets.serve(server.handler, "127.0.0.1", 0)
    port = serving.sockets[0].getsockname()[1]
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"RIFF-voice")
    t = SimpleNamespace(
        server=server,
        serving=serving,
        media=FakeMedia(),
        chat=FakeChat(private),
        closed=[],
        failures=[],
    )
    t.session = CascadeVoiceSession(
        cascade=CascadeOptions(
            url=f"ws://127.0.0.1:{port}/v1/realtime",
            token="secret",
            ref_audio=str(ref),
            tool_filler="On it.",
            emotion="happy",
            emotion_strength=0.5,
        ),
        group=not private,
        instructions=instructions,
        key="server",
        scope_id="test:voice:server",
        prompt="p",
        options=VoiceOptions(name="Jarvis", aliases=["Jar"]),
        media=t.media,
        on_closed=t.closed.append,
        chat=t.chat,
    )
    t.session.launch(t.failures.append)
    t.start = await server.next("session.start")
    for _ in range(200):
        if t.session.ready or t.failures or t.closed:
            break
        await asyncio.sleep(0.01)
    return t


async def finish(t) -> None:
    await asyncio.gather(*chat_module._REQUESTS)
    await t.session.close("done")
    t.serving.close()
    await t.serving.wait_closed()


async def eventually(check) -> None:
    for _ in range(200):
        if check():
            return
        await asyncio.sleep(0.01)
    assert check()


@pytest.mark.asyncio
async def test_the_session_starts_with_the_bot_and_its_voice(tmp_path):
    t = await open_session(tmp_path, True)
    assert t.session.ready, t.failures
    config = t.start["config"]
    assert config["name"] == "Jarvis"
    assert config["aliases"] == ["Jar"]
    assert config["group"] is False
    assert config["speaker"] == "Alice"
    assert config["instructions"] == "Speak like a pirate."
    assert base64.b64decode(config["ref_audio"]) == b"RIFF-voice"
    assert config["tool_filler"] == "On it."
    assert config["tts_emotion"] == "happy"
    assert config["tts_emotion_strength"] == 0.5
    assert t.server.headers["Authorization"] == "Bearer secret"
    assert VOICE_SESSIONS[t.chat.umo] is t.session
    await asyncio.wait_for(t.media.played.wait(), 5)
    await t.session.close("done")
    assert await t.server.next("session.stop") == {"type": "session.stop"}
    assert t.chat.umo not in VOICE_SESSIONS
    t.serving.close()
    await t.serving.wait_closed()


@pytest.mark.asyncio
async def test_platform_instructions_go_before_the_persona(tmp_path):
    t = await open_session(tmp_path, True, instructions=" On a phone call. ")
    assert t.start["config"]["instructions"] == (
        "On a phone call.\n\nSpeak like a pirate."
    )
    await finish(t)


@pytest.mark.asyncio
async def test_a_group_session_has_no_single_speaker(tmp_path):
    t = await open_session(tmp_path, False)
    assert t.start["config"]["group"] is True
    assert t.start["config"]["speaker"] is None
    await finish(t)


@pytest.mark.asyncio
async def test_a_refused_session_fails_to_start(tmp_path):
    t = await open_session(tmp_path, True, {"type": "error", "message": "no model"})
    await eventually(lambda: t.closed)
    assert not t.session.ready
    assert "no model" in str(t.failures[0])
    t.serving.close()
    await t.serving.wait_closed()


@pytest.mark.asyncio
async def test_the_server_rate_is_used_for_its_speech(tmp_path):
    started = {"type": "session.started", "input_rate": 16000, "output_rate": 22050}
    t = await open_session(tmp_path, True, started)
    assert t.session._track.rate == 22050
    await finish(t)


@pytest.mark.asyncio
async def test_an_absurd_server_rate_fails_the_start(tmp_path):
    started = {"type": "session.started", "input_rate": 16000, "output_rate": -5}
    t = await open_session(tmp_path, True, started)
    await eventually(lambda: t.closed)
    assert not t.session.ready
    assert "-5 Hz" in str(t.failures[0])
    t.serving.close()
    await t.serving.wait_closed()


@pytest.mark.asyncio
async def test_a_handed_off_task_is_answered_by_the_chat(tmp_path):
    t = await open_session(tmp_path, False)
    await t.server.socket.send(
        json.dumps(
            {
                "type": "tool.call",
                "call_id": "c1",
                "name": "backend_task",
                "arguments": {"task": "Look up the time"},
                "heard": "Jarvis, what time is it?",
            }
        )
    )
    result = await t.server.next("tool.result")
    assert result == {"type": "tool.result", "call_id": "c1", "output": "It is three."}
    assert "Look up the time" in t.chat.asked[0]
    assert "Jarvis, what time is it?" in t.chat.asked[0]
    await finish(t)


@pytest.mark.asyncio
async def test_a_task_without_arguments_is_what_was_heard(tmp_path):
    t = await open_session(tmp_path, True)
    t.chat.is_busy = True  # a wait is not announced: the filler acknowledged it
    await t.server.socket.send(
        json.dumps(
            {
                "type": "tool.call",
                "call_id": "c2",
                "name": "backend_task",
                "arguments": {},
                "heard": "what time is it",
            }
        )
    )
    await t.server.next("tool.result")
    assert "Task: what time is it" in t.chat.asked[0]
    assert "None" not in t.chat.asked[0]
    assert not [
        e for e in t.server.log if isinstance(e, dict) and e.get("type") == "note"
    ]
    await finish(t)


@pytest.mark.asyncio
async def test_bad_frames_are_skipped_and_the_session_goes_on(tmp_path):
    t = await open_session(tmp_path, True)
    await t.server.socket.send("not json")
    await t.server.socket.send("[1, 2]")
    await t.server.socket.send(b"\x00" * 961)  # half a sample more (kept for the next)
    await t.server.socket.send(b"")
    await t.server.socket.send(json.dumps({"type": "response.cut"}))
    await eventually(lambda: t.media.flushed == 1)
    assert not t.session.closing
    await finish(t)


@pytest.mark.asyncio
async def test_a_cut_drops_the_speech_not_played_yet(tmp_path):
    t = await open_session(tmp_path, True)
    await t.server.socket.send(b"\x00\x00" * 480)
    await t.server.socket.send(json.dumps({"type": "response.cut"}))
    await eventually(lambda: t.media.flushed == 1)
    assert t.session._track._queue.empty()
    await finish(t)


@pytest.mark.asyncio
async def test_the_session_closes_when_the_server_goes_away(tmp_path):
    t = await open_session(tmp_path, True)
    await t.server.socket.close()
    await eventually(lambda: t.closed)
    assert t.closed == [t.session]
    assert t.chat.umo not in VOICE_SESSIONS
    t.serving.close()
    await t.serving.wait_closed()


@pytest.mark.asyncio
async def test_notes_and_openings_go_to_the_server(tmp_path):
    t = await open_session(tmp_path, True)
    await t.session.note("The build finished.")
    assert await t.server.next("note") == {
        "type": "note",
        "text": "The build finished.",
    }
    await t.session.say("You called me back.")
    assert await t.server.next("say") == {"type": "say", "text": "You called me back."}
    await finish(t)


@pytest.mark.asyncio
async def test_speech_is_followed_by_a_little_silence():
    track = SpeechTrack(rate=24000)
    track.put(b"\x01\x00" * 480)
    frame = await track.recv()
    assert frame.samples == 480
    tail = [await track.recv() for _ in range(cascade_module.TAIL_FRAMES)]
    assert all(f.samples == 480 and not any(bytes(f.planes[0])) for f in tail)
    waiting = asyncio.create_task(track.recv())
    await asyncio.sleep(0.3)
    assert not waiting.done()  # then it waits for speech
    track.end()
    with pytest.raises(MediaStreamError):
        await waiting


@pytest.mark.asyncio
async def test_a_sample_split_between_messages_is_kept_whole():
    track = SpeechTrack(rate=24000)
    track.put(b"\x01\x00\x02")
    track.put(b"\x00\x03\x00")
    first, second = await track.recv(), await track.recv()
    assert bytes(first.planes[0])[:2] == b"\x01\x00"
    assert bytes(second.planes[0])[:4] == b"\x02\x00\x03\x00"


@pytest.mark.asyncio
async def test_an_ended_track_stays_ended():
    track = SpeechTrack(rate=24000)
    track.put(b"\x01\x00" * 480)
    track.end()
    for _ in range(2):
        with pytest.raises(MediaStreamError):
            await track.recv()


@pytest.mark.asyncio
async def test_speech_at_real_time_pace_gets_no_silence_in_between():
    track = SpeechTrack(rate=24000)
    loud = b"\x01\x00" * 480

    async def server():
        # A burst (the server's lead), then one frame every 20 ms, late at times.
        for _ in range(15):
            track.put(loud)
        for i in range(60):
            await asyncio.sleep(0.02 + (0.008 if i % 7 == 0 else 0))
            track.put(loud)

    producing = asyncio.create_task(server())
    frames = [await track.recv() for _ in range(75)]
    await producing
    assert all(any(bytes(f.planes[0])) for f in frames)


@pytest.mark.asyncio
async def test_a_cut_during_a_wait_ends_the_silence():
    track = SpeechTrack(rate=24000)
    track.put(b"\x01\x00" * 480)
    await track.recv()
    waiting = asyncio.create_task(track.recv())
    await asyncio.sleep(0)
    track.clear()
    await waiting  # the silent frame it was waiting to give
    later = asyncio.create_task(track.recv())
    await asyncio.sleep(0.5)
    assert not later.done()  # no endless silence after a cut
    track.end()
    with pytest.raises(MediaStreamError):
        await later


def test_settings_are_checked(tmp_path, monkeypatch):
    CascadeOptions().validate()
    with pytest.raises(ValueError):
        CascadeOptions(url="http://127.0.0.1:17890/v1/realtime").validate()
    with pytest.raises(ValueError):
        CascadeOptions(ref_audio=str(tmp_path / "missing.wav")).validate()
    ref = tmp_path / "voice.mp3"
    ref.write_bytes(b"x")
    with pytest.raises(ValueError):
        CascadeOptions(ref_audio=str(ref)).validate()
    big = tmp_path / "big.wav"
    big.write_bytes(b"x" * (cascade_module.MAX_REF_AUDIO_BYTES + 1))
    with pytest.raises(ValueError):
        CascadeOptions(ref_audio=str(big)).validate()
    CascadeOptions(emotion="none", emotion_strength=0).validate()
    with pytest.raises(ValueError):
        CascadeOptions(emotion="bored").validate()
    with pytest.raises(ValueError):
        CascadeOptions(emotion="sad", emotion_strength=1.5).validate()
    # A relative path is under the data directory.
    monkeypatch.setattr(cascade_module, "get_astrbot_data_path", lambda: str(tmp_path))
    (tmp_path / "voice.wav").write_bytes(b"RIFF")
    options = CascadeOptions(ref_audio="voice.wav")
    options.validate()
    assert options.ref_audio_path() == tmp_path / "voice.wav"
