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
from collections.abc import Callable

import av
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

from astrbot import logger

SAMPLE_RATE = 48000
FRAME_SAMPLES = 960  # 20 ms
FRAME_BYTES = FRAME_SAMPLES * 2
# A speaker's backlog beyond this is dropped, oldest first. It is large so
# that what is said while a voice session is still connecting (typically the
# very request that woke the bot) is kept; a backlog only builds up during
# continuous speech and drains in the next pause.
MAX_BACKLOG_SAMPLES = SAMPLE_RATE * 10
# Outbound frames quieter than this (int16 RMS) count as silence.
SILENCE_RMS = 120
# Silence that ends an outbound stretch of speech.
HANGOVER_FRAMES = 15  # 300 ms


class InboundMixer:
    """Per-speaker Opus decoding and mixing into 20 ms frames."""

    def __init__(self) -> None:
        self._decoders: dict[int, av.CodecContext] = {}
        self._pending: dict[int, np.ndarray] = {}
        self.last_voice_at = 0.0

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
            for frame in frames:
                samples = frame.to_ndarray().reshape(-1)
                if samples.dtype != np.int16:
                    samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                previous = self._pending.get(speaker)
                pending = (
                    samples if previous is None else np.concatenate([previous, samples])
                )
                self._pending[speaker] = pending[-MAX_BACKLOG_SAMPLES:]
            self.last_voice_at = time.monotonic()
        if is_terminator:
            # A new transmission starts a fresh decoder state.
            self._decoders.pop(speaker, None)

    def forget(self, speaker: int) -> None:
        self._decoders.pop(speaker, None)
        self._pending.pop(speaker, None)

    def clear(self) -> None:
        self._decoders.clear()
        self._pending.clear()

    def pull(self) -> bytes | None:
        """One mixed 20 ms frame, or ``None`` when nobody is talking."""
        if not self._pending:
            return None
        mix = np.zeros(FRAME_SAMPLES, dtype=np.int32)
        for speaker in list(self._pending):
            pending = self._pending[speaker]
            chunk = pending[:FRAME_SAMPLES]
            mix[: len(chunk)] += chunk
            rest = pending[FRAME_SAMPLES:]
            if len(rest):
                self._pending[speaker] = rest
            else:
                del self._pending[speaker]
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
    """Forwards a peer's audio track to Mumble as Opus while it has sound."""

    def __init__(self, send: Callable[[bytes, bool], None], bitrate: int = 40000):
        """
        Args:
            send: Called with ``(opus_frame, is_terminator)`` for every frame.
            bitrate: Opus bitrate in bits per second.
        """
        self._send = send
        self._bitrate = bitrate
        self.muted = False
        self.talking = False

    def _encoder(self) -> av.CodecContext:
        encoder = av.CodecContext.create("libopus", "w")
        encoder.sample_rate = SAMPLE_RATE
        encoder.layout = "mono"
        encoder.format = "s16"
        encoder.bit_rate = self._bitrate
        encoder.options = {"application": "voip", "frame_duration": "20"}
        encoder.open()
        return encoder

    async def run(self, track: MediaStreamTrack) -> None:
        """Consumes ``track`` until it ends."""
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        encoder = self._encoder()
        buffer = bytearray()
        quiet = 0
        pts = 0
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
                    if self.talking:
                        self._end(encoder, pts)
                    continue
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                loud = float(np.sqrt(np.mean(samples * samples))) >= SILENCE_RMS
                if loud:
                    quiet = 0
                    self.talking = True
                elif not self.talking:
                    continue
                else:
                    quiet += 1
                    if quiet >= HANGOVER_FRAMES:
                        quiet = 0
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
        if self.talking:
            self._end(encoder, pts)

    def _end(self, encoder: av.CodecContext, pts: int) -> None:
        """Closes the current stretch of speech with a terminator frame."""
        self.talking = False
        silence = av.AudioFrame(format="s16", layout="mono", samples=FRAME_SAMPLES)
        silence.planes[0].update(bytes(FRAME_BYTES))
        silence.sample_rate = SAMPLE_RATE
        silence.pts = pts
        silence.time_base = fractions.Fraction(1, SAMPLE_RATE)
        packets = encoder.encode(silence)
        self._send(bytes(packets[-1]) if packets else b"", True)
