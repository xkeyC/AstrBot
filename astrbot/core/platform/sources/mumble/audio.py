"""Audio between Mumble (Opus, one stream per speaker) and a WebRTC peer.

Inbound: each speaker's Opus frames are decoded and mixed into one 48 kHz
mono stream, served to aiortc as a track at real-time pace. Outbound: the
peer's audio is resampled to 48 kHz mono, cut into 20 ms frames, Opus
encoded and handed to Mumble while there is sound, ending each stretch of
speech with a terminator frame.
"""

from __future__ import annotations

import asyncio
import fractions
import time
from collections import deque
from collections.abc import Callable

import av
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

from astrbot import logger

SAMPLE_RATE = 48000
FRAME_SAMPLES = 960  # 20 ms
FRAME_BYTES = FRAME_SAMPLES * 2
# Inbound frames buffered per speaker before mixing starts (see InboundMixer).
JITTER_FRAMES = 2  # 40 ms
# A speaker's backlog beyond this is dropped, oldest first. It is large so
# that what is said while a voice session is still connecting (typically the
# very request that woke the bot) is kept; a backlog only builds up during
# continuous speech and drains in the next pause.
MAX_BACKLOG_SAMPLES = SAMPLE_RATE * 10
# Outbound frames quieter than this (int16 RMS) count as silence.
SILENCE_RMS = 120
# Silence that ends an outbound stretch of speech; long enough not to split
# a sentence at a short pause.
HANGOVER_FRAMES = 25  # 500 ms
# WebRTC hands over the model's audio in bursts (measured p95 gap ~190 ms),
# while Mumble clients play what arrives. Each stretch of speech starts only
# once this much is buffered, and is then sent at a steady 20 ms pace.
PLAYOUT_FRAMES = 10  # 200 ms
# Quiet frames kept in front of the first loud one, so onsets are not cut.
PREROLL_FRAMES = 2
# Buffered audio beyond this is dropped, oldest first, to bound latency.
MAX_QUEUED_FRAMES = 150  # 3 s
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


class InboundMixer:
    """Per-speaker Opus decoding and mixing into 20 ms frames.

    While ``holding`` is set, audio is decoded and kept but not handed out:
    a new realtime session is not ready to listen the moment WebRTC
    connects, and the words that woke it must not be sent into the void.
    """

    def __init__(self) -> None:
        self._decoders: dict[int, av.CodecContext] = {}
        self._pending: dict[int, deque[np.ndarray]] = {}
        self._pending_samples: dict[int, int] = {}
        # Jitter buffer per speaker: mixing starts once JITTER_FRAMES are
        # queued, and resumes that way after running dry mid-speech, instead
        # of mixing half-empty frames (heard as crackle and gaps).
        self._playing: set[int] = set()
        self._ended: set[int] = set()
        self.last_voice_at = 0.0
        self.holding = True

    def feed(self, speaker: int, opus_data: bytes, is_terminator: bool) -> None:
        """Decodes one frame from ``speaker`` into its pending samples.

        Args:
            speaker: Mumble session of the sender.
            opus_data: One Opus frame; may be empty on a bare terminator.
            is_terminator: Whether the speaker stopped talking with this frame.
        """
        decoder = self._decoders.get(speaker)
        if decoder is None:
            decoder = av.CodecContext.create("opus", "r")
            decoder.sample_rate = SAMPLE_RATE
            decoder.layout = "mono"
            self._decoders[speaker] = decoder
        if opus_data:
            try:
                frames = decoder.decode(av.Packet(opus_data))
            except av.error.FFmpegError as exc:
                logger.debug("Mumble: undecodable Opus frame from %s: %s", speaker, exc)
                frames = []
            chunks = self._pending.setdefault(speaker, deque())
            for frame in frames:
                samples = frame.to_ndarray().reshape(-1)
                if samples.dtype != np.int16:
                    samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                chunks.append(samples)
                total = self._pending_samples.get(speaker, 0) + len(samples)
                while total > MAX_BACKLOG_SAMPLES and len(chunks) > 1:
                    total -= len(chunks.popleft())
                self._pending_samples[speaker] = total
            self.last_voice_at = time.monotonic()
        if is_terminator:
            # A new transmission starts a fresh decoder state; what is left
            # of this one is mixed without waiting for more.
            self._decoders.pop(speaker, None)
            self._ended.add(speaker)
        else:
            self._ended.discard(speaker)

    def forget(self, speaker: int) -> None:
        self._decoders.pop(speaker, None)
        self._pending.pop(speaker, None)
        self._pending_samples.pop(speaker, None)
        self._playing.discard(speaker)
        self._ended.discard(speaker)

    def clear(self) -> None:
        self._decoders.clear()
        self._pending.clear()
        self._pending_samples.clear()
        self._playing.clear()
        self._ended.clear()

    def pull(self) -> bytes | None:
        """One mixed 20 ms frame, or ``None`` when nobody is talking."""
        if self.holding or not self._pending:
            return None
        mix = np.zeros(FRAME_SAMPLES, dtype=np.int32)
        mixed = False
        for speaker in list(self._pending):
            chunks = self._pending[speaker]
            available = self._pending_samples.get(speaker, 0)
            ended = speaker in self._ended
            if speaker not in self._playing:
                if available < JITTER_FRAMES * FRAME_SAMPLES and not ended:
                    continue
                self._playing.add(speaker)
            elif available < FRAME_SAMPLES and not ended:
                self._playing.discard(speaker)  # ran dry: buffer up again
                continue
            mixed = True
            filled = 0
            while chunks and filled < FRAME_SAMPLES:
                chunk = chunks[0]
                take = min(FRAME_SAMPLES - filled, len(chunk))
                mix[filled : filled + take] += chunk[:take]
                filled += take
                if take == len(chunk):
                    chunks.popleft()
                else:
                    chunks[0] = chunk[take:]
            self._pending_samples[speaker] -= filled
            if not chunks:
                del self._pending[speaker]
                del self._pending_samples[speaker]
                self._playing.discard(speaker)
        if not mixed:
            return None
        return np.clip(mix, -32768, 32767).astype(np.int16).tobytes()


class SpeechDetector:
    """Energy VAD over decoded Mumble audio, used to leave standby.

    Mumble clients transmit on their own voice activity or push-to-talk, but
    open microphones and noise transmit too, so a packet alone is not speech.
    Speech is ``voiced_frames`` loud 20 ms frames from one speaker within
    ``window_frames``.
    """

    def __init__(
        self, rms: float = 500.0, voiced_frames: int = 10, window_frames: int = 25
    ) -> None:
        self._rms = rms
        self._needed = voiced_frames
        self._window = window_frames
        self._decoders: dict[int, av.CodecContext] = {}
        self._recent: dict[int, list[bool]] = {}

    def feed(self, speaker: int, opus_data: bytes, is_terminator: bool) -> bool:
        """Returns True once ``speaker`` is found to be talking."""
        decoder = self._decoders.get(speaker)
        if decoder is None:
            decoder = av.CodecContext.create("opus", "r")
            decoder.sample_rate = SAMPLE_RATE
            decoder.layout = "mono"
            self._decoders[speaker] = decoder
        recent = self._recent.setdefault(speaker, [])
        if opus_data:
            try:
                frames = decoder.decode(av.Packet(opus_data))
            except av.error.FFmpegError:
                frames = []
            for frame in frames:
                samples = frame.to_ndarray().reshape(-1).astype(np.float32)
                if frame.format.name.startswith("s16"):
                    samples /= 32768.0
                rms = float(np.sqrt(np.mean(samples * samples))) * 32768.0
                recent.append(rms >= self._rms)
            del recent[: -self._window]
        speaking = sum(recent) >= self._needed
        if is_terminator or speaking:
            self._decoders.pop(speaker, None)
            self._recent.pop(speaker, None)
        return speaking

    def forget(self, speaker: int) -> None:
        self._decoders.pop(speaker, None)
        self._recent.pop(speaker, None)

    def clear(self) -> None:
        self._decoders.clear()
        self._recent.clear()


class MixerTrack(MediaStreamTrack):
    """The mixed Mumble audio as an aiortc track; silence when nobody talks."""

    kind = "audio"

    def __init__(self, mixer: InboundMixer) -> None:
        super().__init__()
        self.mixer = mixer
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
        data = self.mixer.pull() or self._silence
        frame = av.AudioFrame(format="s16", layout="mono", samples=FRAME_SAMPLES)
        frame.planes[0].update(data)
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
        self._pts += FRAME_SAMPLES
        return frame


class OutboundVoice:
    """Forwards a peer's audio track to Mumble as Opus while it has sound.

    Receiving and sending are decoupled: ``run`` queues 20 ms frames as the
    track delivers them, and a player sends them at a steady pace after a
    short playout buffer, so bursty WebRTC delivery does not reach listeners
    as stutter.
    """

    def __init__(self, send: Callable[[bytes, bool], None], bitrate: int = 64000):
        """
        Args:
            send: Called with ``(opus_frame, is_terminator)`` for every frame.
            bitrate: Opus bitrate in bits per second.
        """
        self._send = send
        self._bitrate = bitrate
        self.muted = False
        self.talking = False
        self._queue: deque[tuple[bytes, bool]] = deque()
        self._track_ended = False

    def _encoder(self) -> av.CodecContext:
        encoder = av.CodecContext.create("libopus", "w")
        encoder.sample_rate = SAMPLE_RATE
        encoder.layout = "mono"
        encoder.format = "s16"
        encoder.bit_rate = self._bitrate
        # As the official client at its default quality: "audio" (high
        # quality speech) and constant bitrate; "voip" narrows the band,
        # which re-encoded synthetic speech suffers from most.
        encoder.options = {"application": "audio", "frame_duration": "20", "vbr": "off"}
        encoder.open()
        return encoder

    async def run(self, track: MediaStreamTrack) -> None:
        """Consumes ``track`` until it ends, then plays out what is queued."""
        player = asyncio.create_task(self._play())
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        buffer = bytearray()
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
                        self._queue.clear()
                        continue
                    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                    loud = float(np.sqrt(np.mean(samples * samples))) >= SILENCE_RMS
                    self._queue.append((chunk, loud))
                    while len(self._queue) > MAX_QUEUED_FRAMES:
                        self._queue.popleft()
            self._track_ended = True
            await player
        finally:
            player.cancel()
            self.finish()

    async def _play(self) -> None:
        loop = asyncio.get_running_loop()
        encoder = self._encoder()
        queue = self._queue
        pts = 0
        quiet = 0
        next_at = 0.0
        while True:
            if not self.talking:
                if self.muted:
                    queue.clear()
                first_loud = next(
                    (i for i, (_, loud) in enumerate(queue) if loud), None
                )
                if first_loud is None:
                    if self._track_ended:
                        return
                    while len(queue) > PREROLL_FRAMES:
                        queue.popleft()
                    await asyncio.sleep(FRAME_SECONDS)
                    continue
                for _ in range(max(0, first_loud - PREROLL_FRAMES)):
                    queue.popleft()
                if (
                    len(queue) < PLAYOUT_FRAMES + PREROLL_FRAMES
                    and not self._track_ended
                ):
                    await asyncio.sleep(FRAME_SECONDS / 2)
                    continue
                self.talking = True
                quiet = 0
                next_at = loop.time()
            wait = next_at - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            elif wait < -0.2:
                next_at = loop.time()  # the loop stalled: resync, don't burst
            next_at += FRAME_SECONDS
            if self.muted:
                self._end(encoder, pts)
                continue
            if not queue:
                # Underrun: send nothing now; a long one ends the stretch.
                quiet += 1
                if quiet >= HANGOVER_FRAMES or self._track_ended:
                    self._end(encoder, pts)
                    if self._track_ended:
                        return
                continue
            chunk, loud = queue.popleft()
            quiet = 0 if loud else quiet + 1
            if quiet >= HANGOVER_FRAMES:
                self._end(encoder, pts)
                continue
            out = av.AudioFrame(format="s16", layout="mono", samples=FRAME_SAMPLES)
            out.planes[0].update(chunk)
            out.sample_rate = SAMPLE_RATE
            out.pts = pts
            out.time_base = fractions.Fraction(1, SAMPLE_RATE)
            pts += FRAME_SAMPLES
            for packet in encoder.encode(out):
                self._send(bytes(packet), False)

    def finish(self) -> None:
        """Ends a stretch of speech cut off mid-way, so Mumble clients do not
        keep showing the bot as talking."""
        if self.talking:
            self.talking = False
            self._send(b"", True)

    def _end(self, encoder: av.CodecContext, pts: int) -> None:
        """Closes the current stretch of speech with a terminator frame."""
        if not self.talking:
            return
        self.talking = False
        silence = av.AudioFrame(format="s16", layout="mono", samples=FRAME_SAMPLES)
        silence.planes[0].update(bytes(FRAME_BYTES))
        silence.sample_rate = SAMPLE_RATE
        silence.pts = pts
        silence.time_base = fractions.Fraction(1, SAMPLE_RATE)
        packets = encoder.encode(silence)
        self._send(bytes(packets[-1]) if packets else b"", True)
