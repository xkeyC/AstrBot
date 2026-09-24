import fractions

import av
import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError

from astrbot.core.voice import pcm
from astrbot.core.voice.pcm import FRAME_BYTES, PcmMedia


def test_inbound_is_held_until_started_then_jitter_buffered():
    media = PcmMedia(lambda chunk: None)
    media.feed(b"\x01\x00" * (pcm.FRAME_SAMPLES * 3))
    assert media.pull() is None  # holding
    media.start()
    assert media.pull() == b"\x01\x00" * pcm.FRAME_SAMPLES
    assert media.pull() is not None
    assert media.pull() is not None
    assert media.pull() is None  # ran dry
    media.feed(bytes(FRAME_BYTES))
    assert media.pull() is None  # buffers two frames before resuming
    media.feed(bytes(FRAME_BYTES))
    assert media.pull() is not None


def test_inbound_backlog_is_bounded_to_whole_samples(monkeypatch):
    monkeypatch.setattr(pcm, "MAX_BACKLOG_BYTES", FRAME_BYTES * 2)
    media = PcmMedia(lambda chunk: None)
    media.feed(b"\x01\x02" * (pcm.FRAME_SAMPLES * 5))
    media.feed(b"\x01\x02" * 7)
    assert len(media._buffer) == FRAME_BYTES * 2
    assert media._buffer[:2] == b"\x01\x02"  # still sample aligned


def test_stop_drops_inbound_and_outbound():
    sent = []
    media = PcmMedia(sent.append)
    media.start()
    media.feed(bytes(FRAME_BYTES * 3))
    media.stop()
    assert media.pull() is None
    media.feed(bytes(FRAME_BYTES * 3))
    assert media.pull() is None


class ToneTrack:
    """24 kHz stereo frames, as a peer might send, then the end."""

    def __init__(self, frames: int) -> None:
        self.left = frames
        self.pts = 0

    async def recv(self):
        if not self.left:
            raise MediaStreamError
        self.left -= 1
        samples = (np.ones((1, 480 * 2)) * 1000).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="stereo")
        frame.sample_rate = 24000
        frame.pts = self.pts
        frame.time_base = fractions.Fraction(1, 24000)
        self.pts += 480
        return frame


@pytest.mark.asyncio
async def test_play_resamples_to_48k_mono_20ms_chunks():
    sent = []
    media = PcmMedia(sent.append)
    await media.play(ToneTrack(10))  # 200 ms
    assert sent and all(len(chunk) == FRAME_BYTES for chunk in sent)
    assert 8 <= len(sent) <= 10  # the resampler may hold a few samples back
    media.stop()
    sent.clear()
    await media.play(ToneTrack(5))
    assert sent == []
