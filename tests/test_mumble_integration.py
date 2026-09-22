"""Mumble client against a real server.

Run with a server (autoban off: the tests connect often), e.g.::

    docker run -d -p 64738:64738 -e MUMBLE_CONFIG_AUTOBAN_ATTEMPTS=0 \\
        mumblevoip/mumble-server
    MUMBLE_TEST_SERVER=127.0.0.1:64738 pytest tests/test_mumble_integration.py
"""

import asyncio
import fractions
import os
import uuid

import pytest
import pytest_asyncio

from astrbot.core.platform.sources.mumble.client import (
    MumbleClient,
    MumbleRejected,
    TextMessage,
    VoicePacket,
)
from astrbot.core.platform.sources.mumble.messages import AudioContext, AudioTarget

SERVER = os.environ.get("MUMBLE_TEST_SERVER")
pytestmark = pytest.mark.skipif(not SERVER, reason="MUMBLE_TEST_SERVER not set")


def make_client(name: str) -> MumbleClient:
    host, _, port = (SERVER or "").partition(":")
    return MumbleClient(host, int(port or 64738), username=name)


async def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.05)


def opus_frames(count: int) -> list[bytes]:
    import av
    import numpy as np

    encoder = av.CodecContext.create("libopus", "w")
    encoder.sample_rate = 48000
    encoder.layout = "mono"
    encoder.format = "s16"
    encoder.bit_rate = 32000
    encoder.open()
    packets = []
    for i in range(count):
        t = np.arange(i * 960, (i + 1) * 960) / 48000
        pcm = (np.sin(2 * np.pi * 440 * t) * 8000).astype("int16").reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = 48000
        frame.pts = i * 960
        frame.time_base = fractions.Fraction(1, 48000)
        packets.extend(bytes(p) for p in encoder.encode(frame))
    return packets[:count]


@pytest_asyncio.fixture
async def pair():
    suffix = uuid.uuid4().hex[:6]
    bot, human = make_client(f"bot-{suffix}"), make_client(f"human-{suffix}")
    await bot.connect()
    await human.connect()
    await wait_for(lambda: human.session in bot.users and bot.session in human.users)
    yield bot, human
    await bot.close()
    await human.close()


@pytest.mark.asyncio
async def test_text_messages(pair):
    bot, human = pair
    received: list[TextMessage] = []
    bot.on_text = received.append

    human.send_text("hello channel", channel_ids=[human.me.channel_id])
    human.send_text("hello bot", sessions=[bot.session])
    await wait_for(lambda: len(received) == 2)

    channel, private = received
    assert (channel.actor, channel.message, channel.is_private) == (
        human.session,
        "hello channel",
        False,
    )
    assert (private.actor, private.message, private.is_private) == (
        human.session,
        "hello bot",
        True,
    )


@pytest.mark.asyncio
async def test_voice_normal_and_whisper(pair):
    bot, human = pair
    heard: list[VoicePacket] = []
    bot.on_voice = heard.append
    frames = opus_frames(10)

    for i, data in enumerate(frames[:5]):
        human.send_audio(data, is_terminator=i == 4)
    await wait_for(lambda: len(heard) == 5)

    human.set_voice_target(1, sessions=[bot.session])
    for i, data in enumerate(frames[5:]):
        human.send_audio(data, target=1, is_terminator=i == 4)
    await wait_for(lambda: len(heard) == 10)

    assert [p.sender_session for p in heard] == [human.session] * 10
    assert [p.context for p in heard] == [AudioContext.NORMAL] * 5 + [
        AudioContext.WHISPER
    ] * 5
    assert [p.opus_data for p in heard] == frames
    assert [p.is_terminator for p in heard] == ([False] * 4 + [True]) * 2

    import av

    decoder = av.CodecContext.create("opus", "r")
    decoder.sample_rate = 48000
    decoder.layout = "mono"
    decoded = [f for p in heard for f in decoder.decode(av.Packet(p.opus_data))]
    assert sum(f.samples for f in decoded) == 10 * 960


@pytest.mark.asyncio
async def test_loopback_and_user_removal(pair):
    bot, human = pair
    echoed: list[VoicePacket] = []
    bot.on_voice = echoed.append
    bot.send_audio(opus_frames(1)[0], target=AudioTarget.LOOPBACK, is_terminator=True)
    await wait_for(lambda: len(echoed) == 1)
    assert echoed[0].sender_session == bot.session

    removed = []
    bot.on_user_removed = lambda user, _msg: removed.append(user.session)
    await human.close()
    await wait_for(lambda: removed == [human.session])


@pytest.mark.asyncio
async def test_same_name_reconnect_replaces_ghost(pair):
    # Without certificates both connections have the same (empty) hash, so
    # the server treats the new one as the same user and kicks the old one.
    bot, _human = pair
    dropped: list[Exception | None] = []
    bot.on_disconnected = dropped.append
    clone = make_client(bot.username)
    try:
        await clone.connect()
        await wait_for(lambda: len(dropped) == 1)
        assert not bot.connected
        assert clone.connected
    finally:
        await clone.close()


@pytest.mark.asyncio
async def test_invalid_username_is_rejected():
    host, _, port = (SERVER or "").partition(":")
    client = MumbleClient(host, int(port or 64738), username="a b\tc\n")
    with pytest.raises(MumbleRejected) as info:
        await client.connect()
    assert info.value.reject_type == 2  # InvalidUsername
    assert not client.connected
