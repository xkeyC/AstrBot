"""Voice through the realtime voice server of local-multimodal-infra.

The server (``/v1/realtime``, its ``voice-cascade`` model) runs the whole
conversation: Silero VAD, SenseVoice, a Qwen3 chat model and IndexTTS, with
talk-over handling. This session only carries the audio
both ways and runs the tasks the chat model hands off (``tool.call``) as
turns of the paired chat, like ``VoiceSession`` does for Codex realtime; the
answers go back as ``tool.result`` and the server's model tells them.

Each session is a new conversation on the server: it keeps no history across
standby, and the server loads its models before a session starts.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import fractions
import json
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import av
import websockets
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

from astrbot import logger
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .chat import TASK_BODY
from .session import DONE_SPEECH, FAILED_SPEECH, VoiceSession

IN_RATE = 16000  # the server's input: 16-bit mono PCM
# A session loads every model on the server before it starts (IndexTTS takes
# tens of seconds the first time).
START_TIMEOUT = 300.0
# The reference voice goes in one WebSocket message (the server takes 16 MB).
MAX_REF_AUDIO_BYTES = 8 * 1024 * 1024
# After the server's speech stops, this much silence (20 ms frames) follows,
# so the platform's player starts and ends even a very short reply. Speech has
# stopped when nothing arrived for TAIL_GAP: well over the server's 20 ms
# frame period and its jitter, well under its 300 ms lead.
TAIL_FRAMES = 15
TAIL_GAP = 0.2


@dataclass
class CascadeOptions:
    # The realtime WebSocket of the local-multimodal-infra controller.
    url: str = "ws://127.0.0.1:17890/v1/realtime"
    # One of its LOCAL_MCP_INFER_TOKENS (empty when inference is open).
    token: str = ""
    # The voice: a WAV file sent to the server, relative to the data
    # directory unless absolute (empty: the server's default voice).
    ref_audio: str = ""
    # Said by the server right away when a task is handed off (empty: none).
    tool_filler: str = "好的，我查一下。"

    def ref_audio_path(self) -> Path | None:
        """The reference voice file, or None when none is set."""
        if not self.ref_audio:
            return None
        path = Path(self.ref_audio)
        return path if path.is_absolute() else Path(get_astrbot_data_path()) / path

    def validate(self) -> None:
        """Checks the settings (the server has the models).

        Raises:
            ValueError: A setting is unusable.
        """
        parts = urlsplit(self.url)
        if parts.scheme not in ("ws", "wss") or not parts.netloc:
            raise ValueError(
                f"cascade voice URL must be ws:// or wss://, got {self.url!r}"
            )
        path = self.ref_audio_path()
        if path is not None:
            if not path.is_file() or path.suffix.lower() != ".wav":
                raise ValueError(
                    f"cascade reference audio must be an existing .wav file: {path}"
                )
            if path.stat().st_size > MAX_REF_AUDIO_BYTES:
                raise ValueError(
                    f"cascade reference audio is over {MAX_REF_AUDIO_BYTES // 2**20} MB: {path}"
                )


class SpeechTrack(MediaStreamTrack):
    """The server's speech (already at real-time pace) as a track."""

    kind = "audio"

    def __init__(self, rate: int = 24000) -> None:
        """Creates an empty track.

        Args:
            rate: Sample rate of the server's 16-bit mono PCM.
        """
        super().__init__()
        self.rate = rate
        self._queue: asyncio.Queue[av.AudioFrame | None] = asyncio.Queue()
        self._pts = 0
        self._tail = 0
        self._odd = b""  # a byte of a sample split between two messages
        self._ended = False

    def _frame(self, pcm: bytes) -> av.AudioFrame:
        frame = av.AudioFrame(format="s16", layout="mono", samples=len(pcm) // 2)
        frame.planes[0].update(pcm)
        frame.sample_rate = self.rate
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self.rate)
        self._pts += frame.samples
        return frame

    def put(self, pcm: bytes) -> None:
        """Queues server speech.

        Args:
            pcm: 16-bit mono PCM; a trailing odd byte goes before the next.
        """
        pcm = self._odd + pcm
        cut = len(pcm) - len(pcm) % 2
        pcm, self._odd = pcm[:cut], pcm[cut:]
        if pcm:
            self._queue.put_nowait(self._frame(pcm))
            self._tail = TAIL_FRAMES

    def clear(self) -> None:
        """Drops what the player has not taken yet."""
        while not self._queue.empty():
            self._queue.get_nowait()
        self._tail = 0

    def end(self) -> None:
        self._ended = True
        self._queue.put_nowait(None)

    async def recv(self) -> av.AudioFrame:
        """The next frame of speech, or of the silence after it."""
        if self._ended:
            raise MediaStreamError
        while True:
            if self._tail > 0 and self._queue.empty():
                # The first silent frame only once speech has clearly
                # stopped, then one per frame period.
                wait = TAIL_GAP if self._tail == TAIL_FRAMES else 0.02
                try:
                    frame = await asyncio.wait_for(self._queue.get(), wait)
                except asyncio.TimeoutError:
                    self._tail = max(self._tail - 1, 0)
                    return self._frame(bytes(2 * (self.rate // 50)))
            else:
                frame = await self._queue.get()
            if frame is None:
                raise MediaStreamError
            return frame


class CascadeVoiceSession(VoiceSession):
    """A voice conversation on a local-multimodal-infra realtime server."""

    def __init__(
        self,
        *args,
        cascade: CascadeOptions,
        group: bool,
        instructions: str = "",
        **kwargs,
    ) -> None:
        """Creates the session (it starts with ``launch``).

        Args:
            cascade: Server connection and voice settings.
            group: Whether several people talk here (a channel) rather than
                one person with the bot (a whisper, a call).
            instructions: What the platform tells the server's model about
                this conversation (e.g. that it is a phone call), before the
                voice persona. The ``prompt`` is for Codex realtime and is
                not sent: the server has its own.
            *args, **kwargs: As for ``VoiceSession``.
        """
        super().__init__(*args, **kwargs)
        self.cascade = cascade
        self.group = group
        self.instructions = instructions
        self._ws = None
        self._track = SpeechTrack()

    async def _open_agent(self) -> None:
        """No realtime thread: the conversation runs on the server."""

    async def _connect(self) -> None:
        """Opens the server session; the server has loaded its models when
        it answers."""
        self.cascade.validate()
        ref_audio = None
        if path := self.cascade.ref_audio_path():
            data = await asyncio.to_thread(path.read_bytes)
            ref_audio = base64.b64encode(data).decode()
        headers = (
            {"Authorization": f"Bearer {self.cascade.token}"}
            if self.cascade.token
            else {}
        )
        self._ws = await self._wait_open(
            websockets.connect(
                self.cascade.url,
                additional_headers=headers,
                max_size=None,
                open_timeout=15,
                # A local or LAN server: never through a system proxy.
                proxy=None,
                # The server does not answer pings while it loads models.
                ping_interval=None,
            ),
            20,
        )
        try:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "session.start",
                        "config": {
                            "name": self.options.name,
                            "aliases": self.options.aliases,
                            "group": self.group,
                            "speaker": self.chat.sender_name
                            if self.chat.private
                            else None,
                            # The platform's instructions, then the voice
                            # persona (or the platform's extra prompt) the
                            # base session resolved.
                            "instructions": "\n\n".join(
                                text.strip()
                                for text in (self.instructions, self.persona)
                                if text.strip()
                            )
                            or None,
                            "ref_audio": ref_audio,
                            "tool_filler": self.cascade.tool_filler,
                        },
                    }
                )
            )
            while True:
                reply = json.loads(
                    await self._wait_open(self._ws.recv(), START_TIMEOUT)
                )
                if reply.get("type") == "session.started":
                    rate = int(reply.get("output_rate") or 24000)
                    if not 8000 <= rate <= 48000:
                        raise RuntimeError(f"cascade server output rate {rate} Hz")
                    self._track.rate = rate
                    break
                if reply.get("type") == "error":
                    raise RuntimeError(f"cascade server refused the session: {reply}")
            self._check_open()
        except BaseException:
            await self._release_transport()
            raise
        self._phase("cascade session started")
        self._spawn(self.media.play(self._track), "outbound")
        self._spawn(self._receive(), "receive")
        self._spawn(self._send(), "send")
        self.media.start()
        self.started_at = time.monotonic()
        self.ready = True
        logger.info(
            "%s cascade voice session %s started in %.1fs",
            self.label,
            self.key,
            self.started_at - self.created_at,
        )

    async def _send(self) -> None:
        """Sends the input as 16 kHz mono PCM."""
        resampler = av.AudioResampler(format="s16", layout="mono", rate=IN_RATE)
        try:
            while True:
                frame = await self.media.track.recv()
                for out in resampler.resample(frame):
                    await self._ws.send(bytes(out.planes[0])[: out.samples * 2])
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported, the session ends
            logger.warning(
                "%s cascade voice %s: sending failed: %s", self.label, self.key, exc
            )
            self._request_close(f"cascade send failed: {exc}")

    async def _receive(self) -> None:
        """Plays the server's speech and acts on its events until it closes."""
        try:
            async for message in self._ws:
                try:
                    if isinstance(message, bytes):
                        self._track.put(message)
                        continue
                    event = json.loads(message)
                    kind = event.get("type")
                    if kind == "tool.call":
                        self._hand_off(event)
                    elif kind == "response.cut":
                        self._track.clear()
                        self.media.flush()
                    elif kind == "response.text":
                        self.last_answer_at = time.monotonic()
                    elif kind == "input.transcript":
                        self.last_transcript_at = time.monotonic()
                        logger.debug(
                            "%s cascade voice %s heard%s: %s",
                            self.label,
                            self.key,
                            " (partial)" if event.get("partial") else "",
                            event.get("text"),
                        )
                    elif kind == "error":
                        logger.warning(
                            "%s cascade voice %s: server error: %s",
                            self.label,
                            self.key,
                            event.get("message"),
                        )
                except Exception as exc:  # noqa: BLE001 - one bad event
                    logger.warning(
                        "%s cascade voice %s: bad server event skipped: %s",
                        self.label,
                        self.key,
                        exc,
                    )
        except websockets.ConnectionClosed as exc:
            logger.debug(
                "%s cascade voice %s: connection closed: %s", self.label, self.key, exc
            )
        self._request_close("cascade server connection closed")

    def _hand_off(self, event: dict) -> None:
        """Runs a handed-off task as a turn of the paired chat; the answer
        goes back to the server as the tool's result (the server already
        acknowledged the task with its filler, so a wait is not announced)."""
        heard = str(event.get("heard") or "")
        arguments = event.get("arguments")
        task = str(
            (arguments.get("task") if isinstance(arguments, dict) else None) or heard
        )
        call_id = str(event.get("call_id") or "")
        logger.info("%s cascade voice %s: task %r", self.label, self.key, task)

        async def tell(answer: str | None) -> None:
            if answer is None:
                answer = FAILED_SPEECH
            await self._send_event(
                {
                    "type": "tool.result",
                    "call_id": call_id,
                    "output": answer or DONE_SPEECH,
                }
            )

        self._ask(TASK_BODY.format(heard=heard or task, task=task), tell)

    async def _send_event(self, event: dict) -> None:
        """Sends a JSON event; a failure is logged (the receive loop notices a
        closed connection)."""
        if self._ws is None or self._closed:
            return
        try:
            await self._ws.send(json.dumps(event))
        except Exception as exc:  # noqa: BLE001 - the conversation goes on
            logger.warning(
                "%s cascade voice %s: event not sent: %s", self.label, self.key, exc
            )

    async def note(self, text: str) -> None:
        """A result that reached the chat: the server's model tells it if it
        matters."""
        await self._send_event({"type": "note", "text": text})

    async def say(self, text: str) -> None:
        """Has the bot speak first about ``text`` (e.g. why it placed a call).

        Raises:
            RuntimeError: The session is not ready.
        """
        if not self.ready or self._closed:
            raise RuntimeError("voice session is not ready")
        await self._send_event({"type": "say", "text": text})

    async def _release_transport(self) -> None:
        """Stops the tasks and ends the server session (each once)."""
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current:
                task.cancel()
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "session.stop"}))
            with contextlib.suppress(Exception):
                await ws.close()
        self._track.end()
