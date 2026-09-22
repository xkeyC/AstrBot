import asyncio
import fractions

import av
import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError

from astrbot.core.platform.sources.mumble.audio import (
    FRAME_SAMPLES,
    InboundMixer,
    MixerTrack,
    OutboundVoice,
    SpeechDetector,
)


def tone(frames: int, amplitude: float = 8000.0, freq: float = 440.0) -> np.ndarray:
    t = np.arange(frames * FRAME_SAMPLES) / 48000
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.int16)


def encode(pcm: np.ndarray) -> list[bytes]:
    encoder = av.CodecContext.create("libopus", "w")
    encoder.sample_rate = 48000
    encoder.layout = "mono"
    encoder.format = "s16"
    encoder.bit_rate = 32000
    encoder.open()
    packets = []
    for i in range(0, len(pcm), FRAME_SAMPLES):
        frame = av.AudioFrame.from_ndarray(
            pcm[i : i + FRAME_SAMPLES].reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = 48000
        frame.pts = i
        frame.time_base = fractions.Fraction(1, 48000)
        packets.extend(bytes(p) for p in encoder.encode(frame))
    return packets


def test_mixer_holds_until_released():
    mixer = InboundMixer()
    for packet in encode(tone(5)):
        mixer.feed(1, packet, False)
    assert mixer.pull() is None  # kept while the session connects
    mixer.holding = False
    assert mixer.pull() is not None


def test_mixer_backlog_is_bounded():
    from astrbot.core.platform.sources.mumble import audio

    mixer = InboundMixer()
    mixer.holding = False
    packet = encode(tone(2))[0]
    for _ in range(audio.MAX_BACKLOG_SAMPLES // FRAME_SAMPLES + 100):
        mixer.feed(1, packet, False)
    frames = 0
    while mixer.pull() is not None:
        frames += 1
    assert frames <= audio.MAX_BACKLOG_SAMPLES // FRAME_SAMPLES + 1


def test_mixer_mixes_speakers_and_drains():
    mixer = InboundMixer()
    mixer.holding = False
    assert mixer.pull() is None
    for packet in encode(tone(5)):
        mixer.feed(1, packet, False)
    for packet in encode(tone(3, freq=220.0)):
        mixer.feed(2, packet, False)
    frames = []
    while (frame := mixer.pull()) is not None:
        frames.append(np.frombuffer(frame, dtype=np.int16))
    assert all(len(f) == FRAME_SAMPLES for f in frames)
    # Decoder delay trims a little; both speakers contribute at the start.
    assert 4 <= len(frames) <= 5
    assert np.abs(frames[1]).max() > 8000  # two tones summed
    assert mixer.last_voice_at > 0


def test_speech_detector_needs_sustained_sound():
    loud = encode(tone(15))
    quiet = encode(tone(15, amplitude=50.0))
    detector = SpeechDetector()
    assert not any(detector.feed(1, p, False) for p in quiet)
    assert any(detector.feed(2, p, False) for p in loud)
    detector = SpeechDetector()
    # A short click is not speech.
    assert not any(detector.feed(3, p, False) for p in encode(tone(4)))


@pytest.mark.asyncio
async def test_mixer_track_serves_silence_then_audio():
    mixer = InboundMixer()
    mixer.holding = False
    track = MixerTrack(mixer)
    silent = await track.recv()
    assert silent.samples == FRAME_SAMPLES
    assert not np.frombuffer(bytes(silent.planes[0]), dtype=np.int16)[
        :FRAME_SAMPLES
    ].any()
    for packet in encode(tone(3)):
        mixer.feed(1, packet, False)
    frame = await track.recv()
    assert frame.pts == FRAME_SAMPLES
    track.stop()
    with pytest.raises(MediaStreamError):
        await track.recv()


class FakeTrack:
    """A remote track yielding 20 ms stereo float frames, then ending."""

    def __init__(self, pcm: np.ndarray) -> None:
        self.frames = []
        for i in range(0, len(pcm), FRAME_SAMPLES):
            chunk = pcm[i : i + FRAME_SAMPLES].astype(np.float32) / 32768.0
            stereo = np.vstack([chunk, chunk])
            frame = av.AudioFrame.from_ndarray(stereo, format="fltp", layout="stereo")
            frame.sample_rate = 48000
            frame.pts = i
            frame.time_base = fractions.Fraction(1, 48000)
            self.frames.append(frame)

    async def recv(self):
        if not self.frames:
            raise MediaStreamError
        return self.frames.pop(0)


@pytest.mark.asyncio
async def test_outbound_sends_speech_and_terminates_on_silence():
    sent: list[tuple[bytes, bool]] = []
    outbound = OutboundVoice(lambda data, end: sent.append((data, end)))
    pcm = np.concatenate([np.zeros(FRAME_SAMPLES * 5, np.int16), tone(20)])
    pcm = np.concatenate([pcm, np.zeros(FRAME_SAMPLES * 30, np.int16), tone(10)])
    await asyncio.wait_for(outbound.run(FakeTrack(pcm)), 10)
    terminators = [i for i, (_, end) in enumerate(sent) if end]
    # Leading silence is not sent; each stretch of speech ends with a terminator.
    assert len(terminators) == 2
    assert terminators[-1] == len(sent) - 1
    assert 20 <= terminators[0] <= 20 + 15
    assert not outbound.talking


@pytest.mark.asyncio
async def test_outbound_drops_audio_while_muted():
    sent: list[tuple[bytes, bool]] = []
    outbound = OutboundVoice(lambda data, end: sent.append((data, end)))
    outbound.muted = True
    await asyncio.wait_for(outbound.run(FakeTrack(tone(20))), 10)
    assert sent == []
