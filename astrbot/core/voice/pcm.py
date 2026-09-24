"""Raw PCM audio for a voice session, e.g. from a phone bridge.

Audio is 16-bit little-endian mono at 48 kHz both ways, the rate WebRTC
(Opus) runs at, so nothing is resampled on the way in.
"""

from __future__ import annotations

import asyncio
import fractions
import time
from collections import deque
from collections.abc import Callable
from typing import Protocol

import av
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

SAMPLE_RATE = 48000
FRAME_SAMPLES = 960  # 20 ms
FRAME_BYTES = FRAME_SAMPLES * 2
# Frames buffered before inbound audio is handed out, and again after it ran
# dry, so network jitter is not heard as gaps between half-empty frames.
JITTER_FRAMES = 2  # 40 ms
# Inbound audio beyond this is dropped, oldest first. It is large so that
# what is said while the session is still connecting is kept; it drains
# during the next pause.
MAX_BACKLOG_BYTES = SAMPLE_RATE * 2 * 10
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
# With a playout buffer, a stretch of speech starts once this much is queued
# or has been waiting this long: WebRTC hands over a realtime model's audio
# in bursts (measured gaps p95 ~110 ms, p99 ~180 ms), and the far side plays
# what arrives, so each gap would be heard as a dropout.
PREBUFFER_FRAMES = 10  # 200 ms
# Sent at once when a stretch starts, to fill the far side's small buffer.
LEAD_FRAMES = 3  # 60 ms


class FrameSource(Protocol):
    def pull(self) -> bytes | None:
        """One 20 ms frame of 16-bit mono PCM, or None for silence."""


class FrameTrack(MediaStreamTrack):
    """Serves a source's 20 ms frames as an aiortc track at real-time pace;
    silence when the source has nothing."""

    kind = "audio"

    def __init__(self, source: FrameSource) -> None:
        super().__init__()
        self.source = source
        self._pts = 0
        self._start: float | None = None
        self._silence = bytes(FRAME_BYTES)

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        if self._start is None:
            self._start = time.monotonic()
        wait = self._start + self._pts / SAMPLE_RATE - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        elif wait < -1.0:
            # The loop stalled; resync instead of bursting to catch up.
            self._start = time.monotonic() - self._pts / SAMPLE_RATE
        data = self.source.pull() or self._silence
        frame = av.AudioFrame(format="s16", layout="mono", samples=FRAME_SAMPLES)
        frame.planes[0].update(data)
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
        self._pts += FRAME_SAMPLES
        return frame


class PcmMedia:
    """A voice session's audio as raw PCM (``astrbot.core.voice.VoiceMedia``).

    The platform calls ``feed`` with what the other side says, in chunks of
    any size, and gets the bot's voice through ``send``: as it arrives, or,
    with a playout buffer, at real-time pace so ``flush`` can still drop it.
    """

    def __init__(
        self, send: Callable[[bytes], None], buffer_seconds: float = 0
    ) -> None:
        """
        Args:
            send: Receives the bot's voice, 16-bit mono PCM at 48 kHz, in
                20 ms chunks.
            buffer_seconds: Playout buffer for a model that delivers its
                speech ahead of time (a local model); 0 passes the audio on
                as it arrives (a realtime peer already paces it).
        """
        self._send = send
        self._queue: deque[bytes] | None = (
            deque(maxlen=int(buffer_seconds / FRAME_SECONDS))
            if buffer_seconds
            else None
        )
        self._buffer = bytearray()
        self._playing = False
        # Until the model listens, inbound audio is kept, not handed out.
        self.holding = True
        self.muted = False
        self.track = FrameTrack(self)

    def feed(self, pcm: bytes) -> None:
        """Queues inbound audio for the model."""
        if self.muted:
            return
        self._buffer += pcm
        if len(self._buffer) > MAX_BACKLOG_BYTES:
            # Keep whole samples: drop an even number of bytes.
            del self._buffer[: (len(self._buffer) - MAX_BACKLOG_BYTES) & ~1]

    def pull(self) -> bytes | None:
        if self.holding:
            return None
        if not self._playing:
            if len(self._buffer) < JITTER_FRAMES * FRAME_BYTES:
                return None
            self._playing = True
        elif len(self._buffer) < FRAME_BYTES:
            self._playing = False  # ran dry: buffer up again
            return None
        frame = bytes(self._buffer[:FRAME_BYTES])
        del self._buffer[:FRAME_BYTES]
        return frame

    async def play(self, track: MediaStreamTrack) -> None:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        buffer = bytearray()
        ended = asyncio.Event()
        pacer = (
            asyncio.create_task(self._pace(ended)) if self._queue is not None else None
        )
        try:
            while True:
                try:
                    frame = await track.recv()
                except MediaStreamError:
                    break
                for resampled in resampler.resample(frame):
                    buffer += bytes(resampled.planes[0])[: resampled.samples * 2]
                while len(buffer) >= FRAME_BYTES:
                    chunk = bytes(buffer[:FRAME_BYTES])
                    del buffer[:FRAME_BYTES]
                    if self.muted:
                        continue
                    if self._queue is None:
                        self._send(chunk)
                    else:
                        self._queue.append(chunk)  # oldest dropped when full
            ended.set()
            if pacer is not None:
                await pacer  # plays out what is queued
        finally:
            if pacer is not None:
                pacer.cancel()

    async def _pace(self, ended: asyncio.Event) -> None:
        """Sends queued speech at real-time pace, one 20 ms chunk at a time,
        until the queue is empty after ``ended`` is set.

        Each stretch of speech (and each restart after the queue ran dry)
        first buffers up PREBUFFER_FRAMES, or waits that long.
        """
        loop = asyncio.get_running_loop()
        queue = self._queue
        assert queue is not None
        next_at = 0.0
        playing = False
        waiting_since: float | None = None
        while True:
            if not playing:
                now = loop.time()
                if not queue:
                    if ended.is_set() or self.muted:
                        return
                    waiting_since = None
                    await asyncio.sleep(FRAME_SECONDS)
                    continue
                if waiting_since is None:
                    waiting_since = now
                if (
                    len(queue) < PREBUFFER_FRAMES
                    and now - waiting_since < PREBUFFER_FRAMES * FRAME_SECONDS
                    and not ended.is_set()
                ):
                    await asyncio.sleep(FRAME_SECONDS)
                    continue
                playing, waiting_since = True, None
                next_at = now - LEAD_FRAMES * FRAME_SECONDS
            if not queue:
                playing = False  # ran dry: buffer up again
                continue
            wait = next_at - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            elif wait < -1.0:
                next_at = loop.time()  # the loop stalled: resync, don't burst
            next_at += FRAME_SECONDS
            if queue:
                self._send(queue.popleft())

    def start(self) -> None:
        self.holding = False

    def stop(self) -> None:
        self.muted = True
        self._buffer.clear()
        self.flush()

    def flush(self) -> None:
        """Drops the bot's speech not sent yet (only a playout buffer has any)."""
        if self._queue is not None:
            self._queue.clear()
