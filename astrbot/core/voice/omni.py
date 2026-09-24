"""Full-duplex voice through a local MiniCPM-o 4.5 server (llama.cpp-omni).

The server's duplex model listens and speaks. A tool router in the server (the
same LLM in text mode) decides every utterance: let the model answer, stay
silent, or hand a task to the paired chat (Codex, as for Codex realtime; see
``chat``), whose answer the model then speaks verbatim in its own voice.
Utterances are found and transcribed here, on CPU (Silero VAD + SenseVoice),
and sent to the server with the audio. The platform's audio comes and goes
through the session's ``VoiceMedia``, as for Codex realtime.

Server: the ``astrbot-omni`` branch of llama.cpp-omni (session voice clone,
forced speech, tool router); see docs/zh/platform/mumble.md.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import fractions
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

from astrbot import logger
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .chat import OPENING_BODY, TASK_BODY
from .session import VoiceSession

IN_RATE = 16000  # the server's input: 1 s units of 16 kHz float32
OUT_RATE = 24000  # the server's speech: 24 kHz float32
# Noise under the input: the duplex model answers more naturally over a noise
# floor than over digital silence (measured with the same prompts).
DITHER = 10 ** (-55 / 20)
# The first session also loads every model on the server.
INIT_TIMEOUT = 120.0
# A voice reference longer than this confuses the model (it is part of the
# system prompt), and the voice clone does not need more.
REF_AUDIO_SECONDS = 10.0
# Speech arrives faster than real time; the server may send a whole answer
# at once, so a player must buffer this much of it (see MumbleMedia).
PLAYOUT_BUFFER_SECONDS = 120.0
# Audio still arriving for speech that was just cut is dropped this long.
CUT_SECONDS = 0.6
# After the model stops speaking, speech this long (as the VAD reports it,
# with its 0.6 s hangover) within BARGE_IN_SECONDS means it was talked over;
# a short "嗯" stays under it.
BARGE_IN_SPEECH = 1.2
BARGE_IN_SECONDS = 1.5
# One character left after backchannels that still takes the floor.
FLOOR_WORDS = set("停不别等喂")
# A task handed off within this long of the last one gets no second
# acknowledgement (one request often arrives as two utterances).
FILLER_GAP_SECONDS = 8.0
# Utterances made only of these show the speaker is listening; they do not
# take the floor from the bot (longest first when matching).
BACKCHANNELS = sorted(
    {
        *"嗯啊哦噢哎诶唉呃额对好是行哈嘿呵嗷哼呀啦嘛呢咳",
        *("好的", "是的", "对的", "好吧", "行吧", "明白", "明白了", "知道了"),
        *("可以", "没错", "没问题", "有道理", "原来如此", "这样啊", "好嘞", "然后呢"),
        *("ok", "okay", "yeah", "yes", "yep", "yup", "uh", "huh", "um", "mm"),
        *("hmm", "mhm", "right", "sure", "gotit", "isee", "cool", "nice"),
        *("haha", "hehe", "lol", "mhmm"),
    },
    key=len,
    reverse=True,
)

# An agent answer longer than this is cut (it is read aloud at ~5 characters
# a second, and all of it goes into the duplex model's short context).
MAX_SPOKEN_CHARS = 400

ASR_FILES = {
    "model.int8.onnx": "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    "tokens.txt": "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    "silero_vad.onnx": "csukuangfj/vad",
}

DUPLEX_GROUP_PROMPT = (
    "你是语音助手{name}，在一个多人语音频道里和大家聊天。请用自然、简短的中文口语回答。"
)
DUPLEX_PRIVATE_PROMPT = (
    "你是语音助手{name}，正在和{speaker}一对一语音聊天。请用自然、简短的中文口语回答。"
)

ROUTER_TOOLS = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name": "silence", "description": "{silence}", "parameters": {{"type": "object", "properties": {{}}, "required": []}}}}}}
{{"type": "function", "function": {{"name": "reply", "description": "这句话是对{name}说的，{name}可以直接用口语回答。", "parameters": {{"type": "object", "properties": {{}}, "required": []}}}}}}
{{"type": "function", "function": {{"name": "backend_task", "description": "交给后台执行：联网搜索、查询实时信息、执行操作。结果之后由{name}念出来。", "parameters": {{"type": "object", "properties": {{"task": {{"type": "string", "description": "要完成的任务，一句完整的话，包含所有必要细节。"}}}}, "required": ["task"]}}}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

ROUTER_TASKS = """- 对{name}说的，且需要实时信息（时间、日期、天气、新闻、价格、比分等）、联网搜索、查资料、或者执行操作（设置提醒、发消息、控制设备、记录内容等）：调用 backend_task。
- 对{name}说的，闲聊或者凭常识就能回答的：调用 reply。

{name}自己不知道现在的时间、日期、天气和任何最新消息，这些一律用 backend_task。每次必须且只能调用一个工具，不要输出其他文字。"""

ROUTER_GROUP = """你是语音助手"{name}"的决策模块。你收到的是多人语音频道里最新一句话的转写（可能有识别错误）。频道里的人大多在互相聊天，只有叫到"{name}"（或同音字{aliases}）的话，或者紧接着和{name}对话的话，才是对{name}说的。

根据这句话选择一种处理方式：
- 不是对{name}说的：调用 silence。
{tasks}

示例：
"老张你那边信号不好，听不清" -> silence（在和老张说话）
"哈哈哈笑死我了" -> silence（没有叫{name}）
"{name}，今天几号？" -> backend_task，task："查询今天的日期"
"{name}，帮我把这首歌加到收藏" -> backend_task，task："把当前播放的歌曲加入收藏"
"{name}，你觉得猫可爱还是狗可爱？" -> reply
"{name}，用英语怎么说谢谢？" -> reply
"{name}，三十七乘以四等于多少？" -> reply（能直接算出来）

{tools}"""

ROUTER_PRIVATE = """你是语音助手"{name}"的决策模块。{name}正在和一个人一对一语音聊天，你收到的是对方最新一句话的转写（可能有识别错误）。

根据这句话选择一种处理方式：
- 没有需要回应的内容（只有"嗯""啊"之类的语气词、咳嗽、背景声音）：调用 silence。
{tasks}

示例：
"嗯……" -> silence
"今天几号？" -> backend_task，task："查询今天的日期"
"帮我把这首歌加到收藏" -> backend_task，task："把当前播放的歌曲加入收藏"
"你觉得猫可爱还是狗可爱？" -> reply
"三十七乘以四等于多少？" -> reply（能直接算出来）

{tools}"""

# Spoken while a task waits for the paired chat's turn, and when it fails.
BUSY_SPEECH = "我这边还在忙前面的事，稍等一下。"
DONE_SPEECH = "好了，办完了。"
FAILED_SPEECH = "这件事没能办成。"


@dataclass
class OmniOptions:
    url: str = "ws://127.0.0.1:19060/backend"
    ref_audio: str = ""
    silence_bias: float = 4.0
    tool_filler: str = "好的，我查一下。"
    asr_dir: str = ""


def router_config(name: str, aliases: list[str], group: bool, bias: float) -> dict:
    """The server's tool router configuration for a conversation."""
    alias_text = "".join(f"、{a}" for a in aliases if a and a != name)
    tools = ROUTER_TOOLS.format(
        name=name,
        silence="这句话不是对你说的，保持沉默。"
        if group
        else "没有需要回应的内容，保持沉默。",
    )
    template = ROUTER_GROUP if group else ROUTER_PRIVATE
    system = template.format(
        name=name,
        aliases=alias_text,
        tasks=ROUTER_TASKS.format(name=name),
        tools=tools,
    )
    return {
        "tools": ["silence", "reply", "backend_task"],
        # Unaddressed chatter still reads as a question to the model: the
        # bias makes it pick silence unless the utterance is clearly for it.
        "bias": {"silence": bias if group else 0.0},
        "audio_units": 12,
        "silence_hold": 1,
        "tool_hold": 3,
        # Utterances end with the transcripts sent from here (Utterances).
        "client_transcripts": True,
        "transcribe_prompt": "请仔细听这段音频片段，并将其内容逐字记录。",
        "user_template": "{heard}",
        "system": system,
    }


def duplex_prompt(name: str, speaker: str = "") -> str:
    """The omni model's system prompt; the session appends the voice persona
    (or the platform's extra prompt) when it starts."""
    return (
        DUPLEX_PRIVATE_PROMPT.format(name=name, speaker=speaker)
        if speaker
        else DUPLEX_GROUP_PROMPT.format(name=name)
    )


def takes_floor(text: str, names: list[str] | None = None) -> bool:
    """Whether an utterance over the bot's speech means to stop it.

    Args:
        text: The transcript.
        names: For a group: only an utterance naming the bot counts.

    Returns:
        With ``names``, whether one of them is said; otherwise whether it is
        more than backchannels ("嗯", "对对", "好的", "okay", laughter, a cough).
    """
    if names is not None:
        lowered = text.lower()
        return any(name and name.lower() in lowered for name in names)
    question = text.rstrip().endswith(("?", "？"))
    rest = "".join(ch for ch in text.lower() if ch.isalnum())
    while rest:
        word = next((w for w in BACKCHANNELS if rest.startswith(w)), None)
        if word is None:
            # Two characters or more ("别说了", "停下"), a question ("可以吗",
            # "真的？") or a one-word command ("停"); one character else is a
            # particle ("对吧", "OK的", "哇").
            return len(rest) >= 2 or "吗" in rest or question or rest in FLOOR_WORDS
        rest = rest[len(word) :]
    return False


def speakable(text: str) -> str:
    """Plain spoken text from an agent answer (no Markdown, links or code)."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#>|]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_SPOKEN_CHARS:
        cut = max(text.rfind(p, 0, MAX_SPOKEN_CHARS) for p in "。！？.!?")
        text = text[: cut + 1 if cut > 0 else MAX_SPOKEN_CHARS]
    return text


# ---------------------------------------------------------------- speech recognition

_asr_lock = threading.Lock()
_asr_load = asyncio.Lock()
_recognizers: dict[Path, object] = {}  # model directory -> SenseVoice recogniser


async def _asr_models(asr_dir: str) -> Path:
    """The SenseVoice + Silero VAD model directory, downloaded on first use."""
    from astrbot.core.utils.io import download_file

    directory = (
        Path(asr_dir)
        if asr_dir
        else Path(get_astrbot_data_path()) / "models" / "sensevoice"
    )
    directory.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    for name, repo in ASR_FILES.items():
        path = directory / name
        if path.exists() and path.stat().st_size > 0:
            continue
        logger.info("Omni voice: downloading %s", name)
        partial = path.with_suffix(path.suffix + ".part")
        await download_file(
            f"{endpoint}/{repo}/resolve/main/{name}",
            str(partial),
            allow_insecure_ssl_fallback=False,
        )
        partial.replace(path)
    return directory


async def _load_recognizer(asr_dir: str):
    """The SenseVoice recogniser of a model directory, shared by sessions
    (loaded once, used under _asr_lock)."""
    async with _asr_load:
        directory = await _asr_models(asr_dir)
        if directory not in _recognizers:
            import sherpa_onnx

            _recognizers[directory] = await asyncio.to_thread(
                sherpa_onnx.OfflineRecognizer.from_sense_voice,
                model=str(directory / "model.int8.onnx"),
                tokens=str(directory / "tokens.txt"),
                num_threads=2,
                use_itn=True,
            )
        return _recognizers[directory], directory / "silero_vad.onnx"


class Utterances:
    """Finds utterances in the input (VAD) and transcribes each one."""

    # A pause shorter than this between two pieces of speech (e.g. after
    # "<name>,") makes them one utterance, even if the first was already
    # reported on its own.
    JOIN_SECONDS = 1.5

    def __init__(self, recognizer, vad_model: Path) -> None:
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(vad_model)
        config.silero_vad.min_silence_duration = 0.6
        config.sample_rate = IN_RATE
        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        self._recognizer = recognizer
        # Seconds the current speech has gone on (0 when there is none).
        self.speaking_for = 0.0
        self._last_text = ""
        self._last_end = -(10**9)  # sample where the last utterance ended

    def feed(self, unit: np.ndarray) -> tuple[bool, str | None]:
        """Takes one unit of input.

        Returns:
            Whether the unit has speech, and the transcript of an utterance
            that ended in it (``None`` if none did).
        """
        voiced = False
        for i in range(0, len(unit), 512):
            self._vad.accept_waveform(unit[i : i + 512])
            if self._vad.is_speech_detected():
                voiced = True
                self.speaking_for += len(unit[i : i + 512]) / IN_RATE
            else:
                self.speaking_for = 0.0
        text = None
        while not self._vad.empty():
            start = self._vad.front.start
            samples = np.asarray(self._vad.front.samples, dtype=np.float32)
            self._vad.pop()
            with _asr_lock:
                stream = self._recognizer.create_stream()
                stream.accept_waveform(IN_RATE, samples)
                self._recognizer.decode_stream(stream)
                piece = stream.result.text.strip()
            if start - self._last_end < self.JOIN_SECONDS * IN_RATE:
                last = self._last_text
                spaced = (
                    last[-1:].isascii() and last[-1:].isalnum() and piece[:1].isascii()
                )
                piece = last + (" " if spaced else "") + piece
            self._last_text, self._last_end = piece, start + len(samples)
            text = piece
        return voiced, text


# ---------------------------------------------------------------- audio


class PcmTrack(MediaStreamTrack):
    """The server's speech as a track for ``OutboundVoice``."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[av.AudioFrame | None] = asyncio.Queue()
        self._pts = 0

    def put(self, samples: np.ndarray) -> None:
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1), format="flt", layout="mono"
        )
        frame.sample_rate = OUT_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, OUT_RATE)
        self._pts += len(samples)
        self._queue.put_nowait(frame)

    def end(self) -> None:
        self._queue.put_nowait(None)

    async def recv(self) -> av.AudioFrame:
        frame = await self._queue.get()
        if frame is None:
            raise MediaStreamError
        return frame


_ref_audio_cache: dict[tuple[str, int, int], np.ndarray] = {}


def _load_ref_audio(path: str) -> np.ndarray:
    """The first REF_AUDIO_SECONDS of a recording, 16 kHz mono float32
    (decoded once per file version)."""
    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns)
    if (cached := _ref_audio_cache.get(key)) is not None:
        return cached
    limit = int(REF_AUDIO_SECONDS * IN_RATE)
    resampler = av.AudioResampler(format="flt", layout="mono", rate=IN_RATE)
    chunks, total = [], 0
    with av.open(path) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
                total += chunks[-1].size
            if total >= limit:
                break
        else:
            for out in resampler.resample(None):  # the resampler's tail
                chunks.append(out.to_ndarray().reshape(-1))
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    audio = audio[:limit].astype(np.float32)
    _ref_audio_cache.clear()  # one reference voice at a time is typical
    _ref_audio_cache[key] = audio
    return audio


# ---------------------------------------------------------------- session


class OmniVoiceSession(VoiceSession):
    """A voice conversation on the local omni server; the tasks the router
    hands off go to the paired chat, as in ``VoiceSession``.

    The server serves one conversation at a time: a second one is refused
    when it starts, and the platform tries again later.
    """

    def __init__(self, *args, omni: OmniOptions, group: bool, **kwargs) -> None:
        """
        Args:
            omni: Server connection and voice settings.
            group: Whether several people talk here (a channel) rather than
                one person with the bot (a whisper, a call).
            *args, **kwargs: As for ``VoiceSession``.
        """
        super().__init__(*args, **kwargs)
        self.omni = omni
        self.group = group
        self._ws = None
        self._track = PcmTrack()
        self._say: list[str] = []
        self._say_cancel = False
        self._cut_until = 0.0
        self._speaking_for = 0.0  # how long the speaker has been talking
        self._barge_until = 0.0  # the model stopped: talked over if speech goes on
        self._task_at = 0.0
        # When the speech handed to the media so far ends playing, roughly.
        self._playing_until = 0.0
        # The server is producing speech: text since its last listen. (Text
        # and listens arrive in order; audio comes from the TTS thread and
        # often after the listen that ended it.)
        self._server_speaking = False

    async def _connect(self) -> None:
        import websockets

        # A first use downloads the models (~240 MB); a close stops that.
        recognizer, vad_model = await self._wait_open(
            _load_recognizer(self.omni.asr_dir), 1800
        )
        utterances = Utterances(recognizer, vad_model)
        payload: dict = {
            "mode": "full_duplex",
            "use_tts": True,
            "vision": False,
            "system_prompt": self.prompt,
            "config": {
                "listen_prob_scale": 1.0,
                "force_listen_count": 0,
                # Longer answers run on past the next question.
                "length_penalty": 1.2,
                "max_new_speak_tokens_per_chunk": 20,
                "router": router_config(
                    self.options.name,
                    self.options.aliases,
                    self.group,
                    self.omni.silence_bias,
                ),
            },
        }
        if self.omni.ref_audio:
            ref = await self._wait_open(
                asyncio.to_thread(_load_ref_audio, self.omni.ref_audio), 60
            )
            payload["voice"] = {"ref_audio": base64.b64encode(ref.tobytes()).decode()}
        self._ws = await self._wait_open(
            websockets.connect(
                self.omni.url,
                max_size=64 * 1024 * 1024,
                open_timeout=15,
                # The server answers pings only between inputs, not while it
                # loads models or builds a voice for session.init.
                ping_interval=None,
            ),
            20,
        )
        try:
            await self._ws.send(
                json.dumps({"type": "session.init", "payload": payload})
            )
            reply = json.loads(await self._wait_open(self._ws.recv(), INIT_TIMEOUT))
            if reply.get("type") != "session.created":
                raise RuntimeError(f"omni server refused the session: {reply}")
            # A close that gave up waiting for this start has released
            # everything already: nothing may be started after it.
            self._check_open()
        except BaseException:
            await self._release_transport()
            raise
        self._phase("omni session created")
        self._spawn(self.media.play(self._track), "outbound")
        self._spawn(self._receive(), "receive")
        self._spawn(self._send(utterances), "send")
        self.media.start()
        self.started_at = time.monotonic()
        self.ready = True
        logger.info(
            "%s omni voice session %s started in %.1fs",
            self.label,
            self.key,
            self.started_at - self.created_at,
        )

    async def say(self, text: str) -> None:
        """Has the bot speak first about ``text`` (e.g. why it placed a call):
        the paired chat words it, the model speaks it.

        Raises:
            RuntimeError: The session is not ready.
        """
        if not self.ready or self._closed:
            raise RuntimeError("voice session is not ready")
        self._ask(OPENING_BODY.format(purpose=text))

    async def _open_agent(self) -> None:
        """Nothing to open: tasks go to the paired chat, and the omni server
        is not a Codex realtime conversation."""
        self._check_open()

    async def _send(self, utterances: Utterances) -> None:
        """Sends the input in real-time units of one second."""
        try:
            await self._send_units(utterances)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported, the session ends
            logger.warning(
                "%s omni voice %s: sending failed: %s", self.label, self.key, exc
            )
            self._request_close(f"omni send failed: {exc}")

    async def _send_units(self, utterances: Utterances) -> None:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=IN_RATE)
        rng = np.random.default_rng()
        buffer = np.zeros(0, np.float32)
        while True:
            frame = await self.media.track.recv()
            for out in resampler.resample(frame):
                buffer = np.concatenate([buffer, out.to_ndarray().reshape(-1)])
            if len(buffer) < IN_RATE:
                continue
            unit, buffer = buffer[:IN_RATE], buffer[IN_RATE:]
            voiced, transcript = await asyncio.to_thread(utterances.feed, unit)
            self._speaking_for = getattr(utterances, "speaking_for", 0.0)
            self._check_barge_in()
            unit = unit + rng.normal(0, DITHER, IN_RATE).astype(np.float32)
            request: dict = {
                "audio": base64.b64encode(unit.astype(np.float32).tobytes()).decode(),
                "voiced": voiced,
            }
            if transcript is not None:
                request["transcript"] = transcript
                if transcript:
                    self.last_transcript_at = time.monotonic()
                    logger.debug(
                        "%s omni voice %s heard: %s", self.label, self.key, transcript
                    )
                    names = None
                    if self.group:
                        names = [self.options.name, *self.options.aliases]
                    if time.monotonic() < self._playing_until and takes_floor(
                        transcript, names
                    ):
                        # Talked over (forced speech is not stopped by the
                        # model): one to one anything but a backchannel, in
                        # a group only when the bot is named.
                        self._cut(pending=True)
            if self._say_cancel:
                request["say_cancel"] = True
                self._say_cancel = False
            # After a cut, speech waits until what is still arriving from the
            # cut speech has been dropped (it is dropped by time), and while a
            # barge-in may still be decided (it would cut this speech too).
            now = time.monotonic()
            if self._say and now >= self._cut_until and now >= self._barge_until:
                request["say"] = " ".join(self._say)
                self._say.clear()
            await self._ws.send(json.dumps({"type": "input.append", "input": request}))

    async def _receive(self) -> None:
        import websockets

        said: list[str] = []
        try:
            async for raw in self._ws:
                try:
                    self._on_event(json.loads(raw), said)
                except Exception as exc:  # noqa: BLE001 - one bad event
                    logger.warning(
                        "%s omni voice %s: bad server event skipped: %s",
                        self.label,
                        self.key,
                        exc,
                    )
                if self._closed:
                    return
        except websockets.ConnectionClosed as exc:
            logger.debug(
                "%s omni voice %s: connection closed: %s", self.label, self.key, exc
            )
        self._request_close("omni server connection closed")

    def _on_event(self, event: dict, said: list[str]) -> None:
        kind = event.get("type")
        if kind == "response.output.delta":
            delta = event.get("kind")
            now = time.monotonic()
            if delta == "text":
                self._server_speaking = True
            if delta == "audio" and now >= self._cut_until:
                samples = np.frombuffer(base64.b64decode(event["audio"]), np.float32)
                self._track.put(samples)
                self._playing_until = (
                    max(now, self._playing_until) + len(samples) / OUT_RATE
                )
            elif delta == "text":
                said.append(event.get("text") or "")
            elif delta == "listen":
                if self._server_speaking and not self.group:
                    # The model stopped speaking: if the speaker keeps
                    # talking, it was talked over (see _check_barge_in).
                    self._barge_until = now + BARGE_IN_SECONDS
                    self._check_barge_in()
                self._server_speaking = False
        elif kind == "response.done" and said:
            logger.debug(
                "%s omni voice %s said: %s", self.label, self.key, "".join(said)
            )
            said.clear()
        elif kind == "response.tool_call":
            self._on_tool_call(event)
        elif kind == "session.closed":
            self._request_close(f"omni session closed: {event.get('reason')}")

    def _on_tool_call(self, event: dict) -> None:
        name = event.get("name") or ""
        heard = event.get("heard") or ""
        arguments = event.get("arguments") or {}
        if isinstance(arguments, str):
            # Some generations quote the arguments object.
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"task": arguments}
        if not isinstance(arguments, dict):
            arguments = {}
        logger.info(
            "%s omni voice %s: %s %s for %r",
            self.label,
            self.key,
            name,
            json.dumps(arguments, ensure_ascii=False) if arguments else "",
            heard,
        )
        if event.get("interrupted"):
            # The model was still talking when this utterance came, and it
            # talks straight into what it hears: what it has not said yet is
            # likely a reply to chatter (silence) or a made-up answer (task).
            # It was producing that speech, so its audio is still arriving.
            self._cut(arriving=True)
        if name in ("", "reply", "silence"):
            return
        now = time.monotonic()
        task = str(arguments.get("task") or heard)
        busy = self._ask(TASK_BODY.format(heard=heard or task, task=task))
        # One acknowledgement for a request that came in pieces: none for a
        # piece following the last one closely.
        if now - self._task_at > FILLER_GAP_SECONDS:
            if busy:
                self._say.append(BUSY_SPEECH)
            elif self.omni.tool_filler:
                self._say.append(self.omni.tool_filler)
        self._task_at = now

    def _check_barge_in(self) -> None:
        """One to one: the model stopped speaking and the speaker went on
        talking (not a backchannel): what it had still to say is dropped
        (answers waiting to be spoken are kept). Its audio is still
        arriving."""
        now = time.monotonic()
        if (
            now < self._barge_until
            and now < self._playing_until
            and self._speaking_for >= BARGE_IN_SPEECH
        ):
            self._barge_until = 0.0
            self._cut(arriving=True)

    def _cut(self, pending: bool = False, arriving: bool | None = None) -> None:
        """Drops the speech being played and the rest of it still arriving.

        Args:
            pending: Also drop what is still to be spoken (forced speech the
                server has not said yet, answers waiting here): the speaker
                talked over the bot and moved on.
            arriving: Whether audio of the cut speech is still to arrive
                (default: while the server is producing speech).
        """
        if self._server_speaking if arriving is None else arriving:
            # Still producing it: what arrives in a moment is from it too.
            # (Otherwise the next audio is a new reply, not to be dropped.)
            self._cut_until = time.monotonic() + CUT_SECONDS
        self._playing_until = 0.0
        if pending:
            self._say.clear()
            self._say_cancel = True
        self.media.flush()

    async def _tell(self, answer: str | None) -> None:
        """Speaks a request's answer (None: it failed)."""
        if answer is None:
            text = FAILED_SPEECH
        else:
            text = speakable(answer) or DONE_SPEECH
        logger.info("%s omni voice %s: chat answered: %s", self.label, self.key, text)
        self._say.append(text)

    async def note(self, text: str) -> None:
        """A result that reached the chat later (see ``VoiceSession.note``).

        The hook for a context note the duplex model decides on (whether and
        how to tell it); until the server takes such notes it is spoken
        verbatim, like an answer.
        """
        if not self._closed and (spoken := speakable(text)):
            self._say.append(spoken)

    async def _release_transport(self) -> None:
        # Sending and receiving stop before the socket closes under them.
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current:
                task.cancel()
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        self._track.end()
