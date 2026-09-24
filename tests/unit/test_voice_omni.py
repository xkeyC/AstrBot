"""Local MiniCPM-o voice backend (astrbot.core.voice.omni), without a server."""

import asyncio
import base64
import fractions
import json

import av
import numpy as np
import pytest

from astrbot.core.voice import chat as chat_module
from astrbot.core.voice import omni
from astrbot.core.voice.chat import VoiceChat
from astrbot.core.voice.omni import OmniOptions, OmniVoiceSession
from astrbot.core.voice.session import VoiceOptions


class FakeTrack:
    """20 ms frames of a constant tone, as a VoiceMedia input track."""

    def __init__(self) -> None:
        self.pts = 0

    async def recv(self) -> av.AudioFrame:
        await asyncio.sleep(0)
        frame = av.AudioFrame(format="s16", layout="mono", samples=960)
        frame.planes[0].update((np.ones(960, np.int16) * 1000).tobytes())
        frame.sample_rate = 48000
        frame.pts = self.pts
        frame.time_base = fractions.Fraction(1, 48000)
        self.pts += 960
        return frame


class FakeMedia:
    def __init__(self) -> None:
        self.track = FakeTrack()
        self.calls: list[str] = []

    async def play(self, track) -> None:
        self.calls.append("play")

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")

    def flush(self) -> None:
        self.calls.append("flush")


class FakeChat(VoiceChat):
    """The paired chat: records requests, answers each with ``answer``."""

    def __init__(self) -> None:
        super().__init__(umo="test:FriendMessage:1", private=True)
        self.asked: list[str] = []
        self.answer: str | None = "答案"
        self.is_busy = False

    def busy(self) -> bool:
        return self.is_busy or super().busy()

    async def voice_persona(self) -> str:
        return ""

    async def ask(self, body: str) -> str | None:
        self.asked.append(body)
        return self.answer


async def settle() -> None:
    """Lets requests handed to the chat finish."""
    for _ in range(5):
        await asyncio.gather(*chat_module._REQUESTS)
        await asyncio.sleep(0)


class FakeWs:
    """Server events in, requests out."""

    def __init__(self, events: list[dict] | None = None) -> None:
        self.events = events or []
        self.sent: list[dict] = []
        self.changed = asyncio.Event()

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))
        self.changed.set()

    async def wait_sent(self, count: int) -> None:
        while len(self.sent) < count:
            self.changed.clear()
            await self.changed.wait()

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self.events:
            yield json.dumps(event)


def make_session(
    group: bool = True, filler: str = "好的，我查一下。"
) -> OmniVoiceSession:
    session = OmniVoiceSession(
        omni=OmniOptions(tool_filler=filler),
        group=group,
        key="server",
        scope_id="test:voice:server",
        prompt="p",
        options=VoiceOptions(name="小乐", aliases=["晓乐"]),
        media=FakeMedia(),
        on_closed=lambda _s: None,
        chat=FakeChat(),
    )
    return session


def audio_event(seconds: float) -> dict:
    samples = np.zeros(int(omni.OUT_RATE * seconds), np.float32)
    return {
        "type": "response.output.delta",
        "kind": "audio",
        "audio": base64.b64encode(samples.tobytes()).decode(),
    }


def test_router_config_group_and_private():
    group = omni.router_config("小乐", ["晓乐", "小乐"], True, 4.0)
    assert group["tools"] == ["silence", "reply", "backend_task"]
    assert group["bias"] == {"silence": 4.0}
    assert "多人语音频道" in group["system"] and "、晓乐" in group["system"]
    assert "、小乐" not in group["system"]  # the name is not its own alias
    assert '"name": "backend_task"' in group["system"]
    private = omni.router_config("小乐", [], False, 4.0)
    assert private["bias"] == {"silence": 0.0}
    assert "一对一" in private["system"]


def test_duplex_prompt():
    assert "多人语音频道" in omni.duplex_prompt("小乐")
    assert "alice" in omni.duplex_prompt("小乐", "alice")


def test_speakable_strips_markup():
    text = "## 结果\n**晴**，见 [天气网](https://x.y/z) https://a.b\n```py\ncode\n```完"
    assert omni.speakable(text) == "结果 晴，见 天气网 完"


@pytest.mark.asyncio
async def test_backend_task_is_handed_off_with_a_filler():
    session = make_session()
    session._on_tool_call(
        {
            "name": "backend_task",
            "arguments": {"task": "查询当前时间"},
            "heard": "小乐，现在几点了？",
            "interrupted": True,
        }
    )
    await settle()
    # The filler is said; the paired chat's answer becomes a context note
    # the model tells in its own words.
    assert session._say == ["好的，我查一下。"]
    assert session._notes == [("答案", True)]
    assert session.media.calls == ["flush"]  # its own answer was made up
    (text,) = session.chat.asked
    assert "查询当前时间" in text and "小乐，现在几点了？" in text


@pytest.mark.asyncio
async def test_a_busy_chat_is_announced_instead_of_the_filler():
    session = make_session()
    session.chat.is_busy = True
    session._on_tool_call({"name": "backend_task", "arguments": {"task": "查天气"}})
    await settle()
    assert session._say == [omni.BUSY_SPEECH]
    assert session._notes == [("答案", True)]


@pytest.mark.asyncio
async def test_a_failed_request_is_said_to_have_failed():
    session = make_session(filler="")
    session.chat.answer = None  # stopped, failed
    session._on_tool_call({"name": "backend_task", "arguments": {"task": "查天气"}})
    await settle()
    assert session._say == [omni.FAILED_SPEECH]


@pytest.mark.asyncio
async def test_silence_and_reply_do_nothing():
    session = make_session()
    session._on_tool_call(
        {"name": "silence", "heard": "老王吃火锅吗", "interrupted": True}
    )
    session._on_tool_call({"name": "reply", "heard": "小乐讲个笑话"})
    assert chat_module._REQUESTS == set() and session._say == []
    # The stopped speech is dropped (it was turning to the chatter).
    assert session.media.calls == ["flush"]


@pytest.mark.asyncio
async def test_no_filler_when_not_configured():
    session = make_session(filler="")
    session._on_tool_call({"name": "backend_task", "arguments": {}, "heard": "查天气"})
    await settle()
    assert session._say == [] and session._notes == [("答案", True)]
    assert "查天气" in session.chat.asked[0]


@pytest.mark.asyncio
async def test_receive_plays_audio_and_drops_it_after_a_cut():
    session = make_session()
    session._ws = FakeWs(
        [
            audio_event(0.5),
            {
                "type": "response.tool_call",
                "name": "backend_task",
                "arguments": {"task": "x"},
                "interrupted": True,
            },
            audio_event(0.5),  # still arriving from the cut speech
            {"type": "session.closed", "reason": "test"},
        ]
    )
    await session._receive()
    frames = []
    while not session._track._queue.empty():
        frames.append(session._track._queue.get_nowait())
    assert len(frames) == 1
    assert session.closing


@pytest.mark.asyncio
async def test_barge_in_cuts_only_one_to_one():
    for group, cut in ((False, True), (True, False)):
        session = make_session(group=group)
        session._speaking_for = 2.0
        session._ws = FakeWs(
            [
                {"type": "response.output.delta", "kind": "text", "text": "说"},
                audio_event(2.0),
                {"type": "response.output.delta", "kind": "listen"},
                {"type": "session.closed", "reason": "test"},
            ]
        )
        await session._receive()
        assert ("flush" in session.media.calls) is cut


@pytest.mark.asyncio
async def test_chat_answers_are_spoken():
    session = make_session()
    await session._tell("现在是下午三点。")
    await session._tell("")  # done, nothing to say
    await session._tell(None)  # failed
    assert session._notes == [("现在是下午三点。", True)]
    assert session._say == [omni.DONE_SPEECH, omni.FAILED_SPEECH]


@pytest.mark.asyncio
async def test_a_progress_answer_is_spoken_as_given():
    session = make_session(filler="")
    session.chat.answer = "还在查，大约**十秒**。"
    session._on_tool_call(
        {"name": "backend_task", "arguments": {"task": "询问进度：查询比特币价格"}}
    )
    await settle()
    assert session._say == ["还在查，大约十秒。"] and session._notes == []


@pytest.mark.asyncio
async def test_a_later_result_is_a_context_note_to_announce():
    session = make_session()
    await session.note("备份完成了")
    assert session._notes == [("备份完成了", True)] and session._say == []


@pytest.mark.asyncio
async def test_say_asks_the_agent_for_an_opening():
    session = make_session()
    with pytest.raises(RuntimeError):
        await session.say("提醒他开会")
    session.ready = True
    await session.say("提醒他开会")
    await settle()
    (text,) = session.chat.asked
    assert "提醒他开会" in text and "just started" in text
    assert session._say == ["答案"]


class FakeUtterances:
    def __init__(self, results) -> None:
        self.results = list(results)

    def feed(self, unit: np.ndarray):
        assert unit.shape == (omni.IN_RATE,)
        return self.results.pop(0)


@pytest.mark.asyncio
async def test_send_units_with_voice_flags_transcripts_and_speech():
    session = make_session()
    session._ws = FakeWs()
    session._say = ["结果"]
    send = asyncio.create_task(
        session._send(
            FakeUtterances([(True, None), (False, "小乐，几点了"), (False, None)])
        )
    )
    await asyncio.wait_for(session._ws.wait_sent(2), 5)
    send.cancel()
    first, second = (msg["input"] for msg in session._ws.sent[:2])
    assert session._ws.sent[0]["type"] == "input.append"
    audio = np.frombuffer(base64.b64decode(first["audio"]), np.float32)
    assert audio.shape == (omni.IN_RATE,)
    assert first["voiced"] is True and "transcript" not in first
    assert first["say"] == "结果"
    assert second["voiced"] is False and second["transcript"] == "小乐，几点了"
    assert "say" not in second
    assert session.last_transcript_at > 0


def test_speakable_bounds_long_answers():
    text = "第一句话。" * 200
    out = omni.speakable(text)
    assert len(out) <= omni.MAX_SPOKEN_CHARS and out.endswith("。")


@pytest.mark.asyncio
async def test_string_arguments_are_understood():
    session = make_session()
    session._on_tool_call(
        {"name": "backend_task", "arguments": '{"task": "查天气"}', "heard": "x"}
    )
    session._on_tool_call({"name": "backend_task", "arguments": "查新闻", "heard": "y"})
    session._on_tool_call({"name": "backend_task", "arguments": 3, "heard": "z"})
    await settle()
    texts = session.chat.asked
    assert "查天气" in texts[0] and "查新闻" in texts[1] and "z" in texts[2]


@pytest.mark.asyncio
async def test_a_bad_event_is_skipped_not_fatal():
    session = make_session()
    session._ws = FakeWs(
        [
            {"type": "response.output.delta", "kind": "audio"},  # no audio
            audio_event(0.5),
        ]
    )
    await session._receive()
    assert session._track._queue.qsize() == 1  # the good one still played


@pytest.mark.asyncio
async def test_a_cut_cancels_pending_speech_and_holds_new_speech():
    session = make_session()
    session._server_speaking = True  # what arrives next is from the cut speech
    session._say = ["结果"]
    session._cut()  # a router stop keeps what is still to be spoken
    assert session._say == ["结果"] and not session._say_cancel
    session._say = ["旧答案"]
    session._cut(pending=True)  # talked over: it goes too
    assert session._say == [] and session._say_cancel
    assert session.media.calls == ["flush", "flush"]
    session._ws = FakeWs()
    session._say.append("好的，我查一下。")
    send = asyncio.create_task(session._send(FakeUtterances([(False, None)] * 3)))
    await asyncio.wait_for(session._ws.wait_sent(1), 5)
    first = session._ws.sent[0]["input"]
    assert first["say_cancel"] is True and "say" not in first  # held back
    session._cut_until = 0.0
    await asyncio.wait_for(session._ws.wait_sent(2), 5)
    send.cancel()
    assert session._ws.sent[1]["input"]["say"] == "好的，我查一下。"


@pytest.mark.asyncio
async def test_send_failure_closes_the_session():
    class Broken:
        def feed(self, unit):
            raise RuntimeError("asr broke")

    session = make_session()
    session._ws = FakeWs()
    await session._send(Broken())
    assert session.closing


def test_utterance_pieces_join_with_a_space_between_words():
    class Vad:
        def __init__(self, segments):
            self.segments = segments

        def accept_waveform(self, samples):
            pass

        def is_speech_detected(self):
            return False

        def empty(self):
            return not self.segments

        @property
        def front(self):
            return self.segments[0]

        def pop(self):
            self.segments.pop(0)

    class Segment:
        def __init__(self, start, text):
            self.start, self.samples, self.text = start, [0.0] * 8000, text

    class Recognizer:
        def create_stream(self):
            class Stream:
                def accept_waveform(inner, rate, samples):
                    pass

            return Stream()

        def decode_stream(self, stream):
            stream.result = type("R", (), {"text": self.next})()

    utt = omni.Utterances.__new__(omni.Utterances)
    utt._recognizer = Recognizer()
    utt._last_text, utt._last_end = "", -(10**9)
    unit = np.zeros(16000, np.float32)
    for start, text, want in (
        (0, "hello", "hello"),
        (16000, "how are you", "hello how are you"),
        (32000, "小乐", "hello how are you小乐"),
        (200000, "你好", "你好"),  # long after: a new utterance
    ):
        utt._vad = Vad([Segment(start, text)])
        utt._recognizer.next = text
        assert utt.feed(unit) == (False, want)


def test_takes_floor():
    for text in ("嗯", "嗯嗯。", "对对对", "好的", "是啊", "咳", "哈哈哈"):
        assert not omni.takes_floor(text), text
    for text in (
        "明白了",
        "没错没错",
        "可以可以",
        "Okay.",
        "yeah yeah",
        "uh huh",
        "mm-hmm",
        "haha",
        "好的呀",
    ):
        assert not omni.takes_floor(text), text
    for text in (
        "别说了",
        "等一下，我问个事",
        "stop talking",
        "停下",
        "可以吗",
        "明白吗",
    ):
        assert omni.takes_floor(text), text
    names = ["小乐", "晓乐", "", None]
    assert not omni.takes_floor("老王你别说了", names)
    assert omni.takes_floor("晓乐，别说了", names)
    assert omni.takes_floor("hey JARVIS stop", ["Jarvis"])


@pytest.mark.asyncio
async def test_talking_over_the_bot_needs_more_than_a_backchannel():
    for group, text, cut in (
        (False, "嗯", False),
        (False, "等一下，先别说", True),
        (True, "等一下，先别说", False),  # a group: others talk anyway
        (True, "小乐，先别说", True),
    ):
        session = make_session(group=group)
        session._ws = FakeWs()
        session._say = ["排队的答案"]
        session._playing_until = omni.time.monotonic() + 30
        send = asyncio.create_task(session._send(FakeUtterances([(False, text)])))
        await asyncio.wait_for(session._ws.wait_sent(1), 5)
        send.cancel()
        assert ("flush" in session.media.calls) is cut, (group, text)
        assert (session._ws.sent[0]["input"].get("say_cancel") is True) is cut


@pytest.mark.asyncio
async def test_a_listen_cut_keeps_waiting_answers():
    session = make_session(group=False)
    session._say = ["答案"]
    session._speaking_for = 2.0
    session._ws = FakeWs(
        [
            {"type": "response.output.delta", "kind": "text", "text": "说"},
            audio_event(2.0),
            {"type": "response.output.delta", "kind": "listen"},
        ]
    )
    await session._receive()
    assert "flush" in session.media.calls
    assert session._say == ["答案"] and not session._say_cancel


@pytest.mark.asyncio
async def test_one_acknowledgement_for_a_request_in_two_pieces():
    session = make_session()
    for heard in ("帮我查天气", "北京的"):
        session._on_tool_call({"name": "backend_task", "arguments": {"task": heard}})
    await settle()
    assert session._say == ["好的，我查一下。"]
    assert session._notes == [("答案", True), ("答案", True)]
    assert len(session.chat.asked) == 2


def test_reference_audio_is_decoded_once_and_trimmed(tmp_path, monkeypatch):
    import wave

    path = tmp_path / "ref.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(np.zeros(44100 * 30, np.int16).tobytes())  # 30 s
    audio = omni._load_ref_audio(str(path))
    assert audio.shape == (int(omni.REF_AUDIO_SECONDS * omni.IN_RATE),)
    monkeypatch.setattr(omni.av, "open", lambda *_: pytest.fail("decoded again"))
    assert omni._load_ref_audio(str(path)) is audio


@pytest.mark.asyncio
async def test_a_close_during_start_leaves_no_connection(monkeypatch):
    import websockets

    class Server(FakeWs):
        closed = False

        async def recv(self):
            session._request_close("closed while starting")  # e.g. standby
            await asyncio.sleep(0)
            return json.dumps({"type": "session.created"})

        async def close(self):
            Server.closed = True

    async def load(_dir):
        return object(), None

    async def connect(*_args, **_kwargs):
        return Server()

    monkeypatch.setattr(omni, "_load_recognizer", load)
    monkeypatch.setattr(omni, "Utterances", lambda *_: None)
    monkeypatch.setattr(websockets, "connect", connect)
    session = make_session()
    with pytest.raises(Exception):  # the start stops: closed meanwhile
        await session._connect()
    assert Server.closed
    assert session._tasks == [] and not session.ready


@pytest.mark.asyncio
async def test_listen_cuts_only_when_speech_stops():
    for events, cut in (
        # Speech ended long ago, its audio still playing: a listen (every
        # unit while listening) is no barge-in.
        ([{"type": "response.output.delta", "kind": "listen"}], False),
        # Speaking, then listening while the speaker talks: cut.
        (
            [
                {"type": "response.output.delta", "kind": "text", "text": "说"},
                audio_event(2.0),
                {"type": "response.output.delta", "kind": "listen"},
            ],
            True,
        ),
    ):
        session = make_session(group=False)
        session._speaking_for = 2.0
        session._playing_until = omni.time.monotonic() + 30
        session._ws = FakeWs(events)
        await session._receive()
        assert ("flush" in session.media.calls) is cut


def test_a_cut_after_the_server_finished_keeps_the_next_reply():
    session = make_session()
    session._server_speaking = False
    session._cut(pending=True)
    assert session._cut_until == 0.0  # the next audio is a new reply


@pytest.mark.asyncio
async def test_barge_in_drops_the_late_audio_of_the_cut_speech():
    session = make_session(group=False)
    session._speaking_for = 2.0
    session._ws = FakeWs(
        [
            {"type": "response.output.delta", "kind": "text", "text": "我给你讲"},
            audio_event(2.0),
            {"type": "response.output.delta", "kind": "listen"},  # talked over
            audio_event(0.5),  # late audio of the cut speech
        ]
    )
    await session._receive()
    assert "flush" in session.media.calls
    assert session._track._queue.qsize() == 1  # only the audio before the cut


@pytest.mark.asyncio
async def test_late_audio_does_not_mark_the_server_speaking():
    session = make_session(group=False)
    session._speaking_for = 2.0
    session._ws = FakeWs(
        [
            {"type": "response.output.delta", "kind": "text", "text": "答完了"},
            {"type": "response.output.delta", "kind": "listen"},  # the turn ended
            audio_event(2.0),  # its audio, arriving after the listen
            {"type": "response.output.delta", "kind": "listen"},  # next unit
        ]
    )
    events = session._ws.events
    session._ws.events = events[:2]
    await session._receive()
    session._closed = False  # keep going with the same session
    session._speaking_for = 2.0
    session._ws.events = events[2:]
    await session._receive()
    assert "flush" not in session.media.calls  # the answer's tail plays out


def test_takes_floor_one_character_left_is_a_particle():
    for text in ("好嘞", "欸", "哇", "对吧", "是吧", "OK的", "mhmm", "然后呢", "嗯？"):
        assert not omni.takes_floor(text), text
    for text in ("可以吗", "真的？", "哦是吗", "停", "嗯，不", "喂"):
        assert omni.takes_floor(text), text


@pytest.mark.asyncio
async def test_a_backchannel_at_the_end_of_an_answer_keeps_its_tail():
    session = make_session(group=False)
    session._speaking_for = 0.8  # a short "嗯" (with the VAD hangover)
    session._ws = FakeWs(
        [
            {"type": "response.output.delta", "kind": "text", "text": "答完了"},
            audio_event(2.0),
            {"type": "response.output.delta", "kind": "listen"},
        ]
    )
    await session._receive()
    assert "flush" not in session.media.calls


@pytest.mark.asyncio
async def test_barge_in_decided_when_speech_goes_on_after_the_listen():
    session = make_session(group=False)
    session._speaking_for = 0.5  # just started talking when the model stopped
    session._ws = FakeWs(
        [
            {"type": "response.output.delta", "kind": "text", "text": "我给你讲"},
            audio_event(3.0),
            {"type": "response.output.delta", "kind": "listen"},
        ]
    )
    await session._receive()
    assert "flush" not in session.media.calls  # not yet
    session._speaking_for = 1.5  # ... and keeps talking
    session._check_barge_in()
    assert "flush" in session.media.calls


def test_utterances_measure_how_long_speech_goes_on():
    class Vad:
        def __init__(self, speech):
            self.speech, self.n = speech, 0

        def accept_waveform(self, samples):
            self.n += 1

        def is_speech_detected(self):
            return self.speech(self.n)

        def empty(self):
            return True

    utt = omni.Utterances.__new__(omni.Utterances)
    utt.speaking_for = 0.0
    utt._vad = Vad(lambda n: True)
    utt.feed(np.zeros(16000, np.float32))
    assert utt.speaking_for == pytest.approx(1.0)
    utt._vad = Vad(lambda n: n < 10)  # stops after 10 windows
    utt.feed(np.zeros(16000, np.float32))
    assert utt.speaking_for == 0.0


@pytest.mark.asyncio
async def test_answers_wait_while_a_barge_in_may_be_decided():
    session = make_session(group=False)
    session._ws = FakeWs()
    session._say = ["答案"]
    session._barge_until = omni.time.monotonic() + 30
    send = asyncio.create_task(session._send(FakeUtterances([(False, None)] * 3)))
    await asyncio.wait_for(session._ws.wait_sent(1), 5)
    assert "say" not in session._ws.sent[0]["input"]
    session._barge_until = 0.0
    await asyncio.wait_for(session._ws.wait_sent(2), 5)
    send.cancel()
    assert session._ws.sent[1]["input"]["say"] == "答案"


def test_private_router_sees_the_context_notes():
    private = omni.router_config("小乐", [], False, 0.0)
    assert "{context}" in private["user_template"]
    assert "{heard}" in private["user_template"]
    assert private["context_empty"] == "（无）"
    assert "询问进度" in private["system"]
    group = omni.router_config("小乐", [], True, 4.0)
    assert "{context}" in group["user_template"] and "频道" in group["user_template"]
    # One short sentence of rules in a group (more tipped it to silence).
    assert omni.ROUTER_GROUP_CONTEXT_RULES in group["system"]
    assert "明天小雨" not in group["system"]


@pytest.mark.asyncio
async def test_a_progress_question_while_its_task_runs_is_answered_here():
    session = make_session(filler="")
    session.chat.is_busy = True
    session._on_tool_call(
        {"name": "backend_task", "arguments": {"task": "询问进度：查询比特币价格"}}
    )
    await settle()
    assert session._say == [omni.STILL_WORKING_SPEECH]
    assert session.chat.asked == []  # not queued behind the task it asks about


def test_notes_are_cleaned_and_bounded():
    note = omni.speakable("**结果**：" + "很长的一句话。" * 300, omni.MAX_NOTE_CHARS)
    assert "*" not in note and len(note) <= omni.MAX_NOTE_CHARS


@pytest.mark.asyncio
async def test_a_model_load_is_shared_and_outlives_its_caller(monkeypatch):
    started = []
    gate = asyncio.Event()

    async def load(asr_dir):
        started.append(asr_dir)
        await gate.wait()
        return "recognizer", "vad"

    monkeypatch.setattr(omni, "_load_recognizer", load)
    monkeypatch.setattr(omni, "_loads", {})
    waiter = asyncio.ensure_future(omni._recognizer("d"))
    await asyncio.sleep(0)
    waiter.cancel()  # the session gave up (closed, timed out)
    gate.set()
    assert await omni._recognizer("d") == ("recognizer", "vad")
    assert started == ["d"]  # one load, not restarted


@pytest.mark.asyncio
async def test_an_omni_session_takes_results_reaching_its_chat(monkeypatch):
    session = make_session()

    async def nothing():
        return None

    monkeypatch.setattr(session, "_open_agent", nothing)
    monkeypatch.setattr(session, "_connect", nothing)
    await session._start()
    assert chat_module.VOICE_SESSIONS[session.chat.umo] is session
    session.ready = True
    assert await chat_module.announce(session.chat.umo, "备份完成了")
    assert session._notes == [("备份完成了", True)]
    chat_module.VOICE_SESSIONS.pop(session.chat.umo)
