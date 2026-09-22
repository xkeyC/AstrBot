import struct

import pytest

from astrbot.core.platform.sources.mumble import protobuf
from astrbot.core.platform.sources.mumble.client import (
    MumbleClient,
    TextMessage,
    VoicePacket,
    decode_audio,
    encode_audio,
)
from astrbot.core.platform.sources.mumble.messages import (
    AUDIO,
    SCHEMAS,
    AudioContext,
    MessageType,
    version_v1,
    version_v2,
)


def roundtrip(message_type: MessageType, message: dict) -> dict:
    schema = SCHEMAS[message_type]
    return protobuf.decode(schema, protobuf.encode(schema, message))


def test_varint_boundaries():
    for value in (0, 1, 127, 128, 300, 2**32 - 1, 2**63):
        encoded = protobuf.encode_varint(value)
        assert protobuf.decode_varint(encoded, 0) == (value, len(encoded))
    assert protobuf.encode_varint(300) == b"\xac\x02"


def test_messages_roundtrip():
    auth = {
        "username": "bot 机器人",
        "password": "pw",
        "tokens": ["a", "b"],
        "opus": True,
        "client_type": 1,
    }
    assert roundtrip(MessageType.Authenticate, auth) == auth
    state = {"session": 7, "name": "alice", "channel_id": 0, "self_mute": False}
    assert roundtrip(MessageType.UserState, state) == state
    ping = {"timestamp": 123456789, "tcp_ping_avg": 1.5}
    assert roundtrip(MessageType.Ping, ping) == ping


def test_negative_int32_uses_ten_byte_varint():
    encoded = protobuf.encode(SCHEMAS[MessageType.ChannelState], {"position": -1})
    assert encoded == b"\x48" + b"\xff" * 9 + b"\x01"
    assert roundtrip(MessageType.ChannelState, {"position": -5}) == {"position": -5}


def test_absent_fields_are_omitted_and_unknown_fields_skipped():
    schema = SCHEMAS[MessageType.TextMessage]
    assert protobuf.encode(schema, {"message": "hi", "session": None}) == b"*\x02hi"
    unknown = (
        protobuf.encode_varint(99 << 3 | 2)
        + b"\x03abc"
        + protobuf.encode_varint(98 << 3 | 0)
        + b"\x05"
        + protobuf.encode_varint(97 << 3 | 5)
        + b"\0\0\0\0"
        + protobuf.encode_varint(96 << 3 | 1)
        + b"\0" * 8
    )
    assert protobuf.decode(schema, unknown + b"*\x02hi") == {"message": "hi"}


def test_packed_repeated_scalars_decode():
    schema = SCHEMAS[MessageType.TextMessage]
    packed = protobuf.encode_varint(2 << 3 | 2) + b"\x03\x01\x02\x03"
    assert protobuf.decode(schema, packed) == {"session": [1, 2, 3]}


def test_truncated_input_raises():
    with pytest.raises(protobuf.DecodeError):
        protobuf.decode(SCHEMAS[MessageType.TextMessage], b"*\x05hi")
    with pytest.raises(protobuf.DecodeError):
        protobuf.decode_varint(b"\x80", 0)


def test_nested_voice_target():
    target = {"id": 3, "targets": [{"session": [4, 5]}, {"channel_id": 0}]}
    assert roundtrip(MessageType.VoiceTarget, target) == target


def test_versions():
    assert version_v1(1, 5, 0) == 0x010500
    assert version_v1(1, 5, 915) == 0x0105FF
    assert version_v2(1, 5, 915) == 1 << 48 | 5 << 32 | 915 << 16


def test_audio_packet_roundtrip():
    packet = encode_audio(b"\x01\x02", frame_number=40, target=31, is_terminator=True)
    assert packet[0] == 0
    assert protobuf.decode(AUDIO, packet[1:]) == {
        "target": 31,
        "frame_number": 40,
        "opus_data": b"\x01\x02",
        "is_terminator": True,
    }
    received = bytes([0]) + protobuf.encode(
        AUDIO,
        {"context": 2, "sender_session": 9, "frame_number": 2, "opus_data": b"x"},
    )
    assert decode_audio(received) == VoicePacket(
        sender_session=9,
        context=AudioContext.WHISPER,
        frame_number=2,
        opus_data=b"x",
        is_terminator=False,
    )
    assert decode_audio(bytes([1, 8, 1])) is None


def frame(message_type: MessageType, message: dict) -> tuple[int, bytes]:
    return int(message_type), protobuf.encode(SCHEMAS[message_type], message)


def test_client_tracks_state_and_dispatches():
    client = MumbleClient("localhost")
    texts: list[TextMessage] = []
    voices: list[VoicePacket] = []
    changes: list[tuple[str, set[str]]] = []
    removed: list[str] = []
    client.on_text = texts.append
    client.on_voice = voices.append
    client.on_user_changed = lambda user, fields: changes.append((user.name, fields))
    client.on_user_removed = lambda user, _msg: removed.append(user.name)

    for message_type, message in [
        (MessageType.ChannelState, {"channel_id": 0, "name": "Root"}),
        (MessageType.ChannelState, {"channel_id": 1, "parent": 0, "name": "Lobby"}),
        (MessageType.ChannelState, {"channel_id": 2, "parent": 1, "name": "Voice"}),
        (MessageType.ChannelState, {"channel_id": 2, "links_add": [1]}),
        (MessageType.UserState, {"session": 1, "name": "bot", "channel_id": 0}),
        (MessageType.UserState, {"session": 2, "name": "alice", "channel_id": 1}),
        (MessageType.ServerSync, {"session": 1, "welcome_text": "hi"}),
        (MessageType.UserState, {"session": 2, "channel_id": 2}),
        (MessageType.TextMessage, {"actor": 2, "session": [1], "message": "psst"}),
        (MessageType.UserRemove, {"session": 2}),
        (MessageType.ChannelRemove, {"channel_id": 1}),
    ]:
        client._dispatch(*frame(message_type, message))
    tunnel = bytes([0]) + protobuf.encode(
        AUDIO, {"sender_session": 2, "opus_data": b"o"}
    )
    client._dispatch(int(MessageType.UDPTunnel), tunnel)
    client._dispatch(int(MessageType.UDPTunnel), b"")
    client._dispatch(250, b"ignored")

    assert client.session == 1
    assert client.me is not None and client.me.name == "bot"
    assert client.welcome_text == "hi"
    assert sorted(client.channels) == [0, 2]
    assert client.channels[2].links == {1}
    assert client.find_channel("Voice") is client.channels[2]
    assert client.find_channel("Lobby/Voice") is None  # parent removed
    assert changes == [
        ("bot", {"name"}),
        ("alice", {"name", "channel_id"}),
        ("alice", {"channel_id"}),
    ]
    assert texts == [
        TextMessage(actor=2, message="psst", sessions=[1], channel_ids=[], tree_ids=[])
    ]
    assert texts[0].is_private
    assert removed == ["alice"]
    assert [v.opus_data for v in voices] == [b"o"]


def test_find_channel_by_path():
    client = MumbleClient("localhost")
    for message in (
        {"channel_id": 0, "name": "Root"},
        {"channel_id": 1, "parent": 0, "name": "Games"},
        {"channel_id": 2, "parent": 1, "name": "Lobby"},
        {"channel_id": 3, "parent": 0, "name": "Lobby"},
    ):
        client._dispatch(*frame(MessageType.ChannelState, message))
    assert client.find_channel("Games/Lobby").channel_id == 2
    assert client.find_channel("/Lobby").channel_id == 3
    assert client.find_channel("Games/Missing") is None


def test_header_layout():
    client = MumbleClient("localhost")
    written = bytearray()

    class Writer:
        def write(self, data):
            written.extend(data)

    client._writer = Writer()  # type: ignore[assignment]
    client.send_text("hi", channel_ids=[0])
    message_type, length = struct.unpack(">HI", written[:6])
    assert (message_type, length) == (MessageType.TextMessage, len(written) - 6)
    assert protobuf.decode(SCHEMAS[MessageType.TextMessage], written[6:]) == {
        "channel_id": [0],
        "message": "hi",
    }


def test_bad_messages_and_failing_callbacks_do_not_escape_dispatch():
    client = MumbleClient("localhost")

    def boom(_message):
        raise RuntimeError("callback bug")

    client.on_text = boom
    client._dispatch(*frame(MessageType.TextMessage, {"actor": 2, "message": "x"}))
    # Wire type mismatch in a control message.
    client._dispatch(int(MessageType.UserState), b"\x0a\x01x")
    # Packed floats whose length is not a multiple of 4.
    client._dispatch(int(MessageType.UDPTunnel), bytes([0, 0x32, 5, 1, 2, 3, 4, 5]))


def test_reconnect_state_is_reset():
    client = MumbleClient("localhost")
    client._dispatch(
        *frame(MessageType.UserState, {"session": 5, "name": "old", "hash": "h"})
    )
    client._dispatch(*frame(MessageType.ServerConfig, {"message_length": 10}))
    assert client.users and client.server_config

    async def refused(*_args, **_kwargs):
        raise ConnectionRefusedError

    import asyncio

    original = asyncio.open_connection
    asyncio.open_connection = refused
    try:
        with pytest.raises(ConnectionRefusedError):
            asyncio.run(client.connect(timeout=1))
    finally:
        asyncio.open_connection = original
    assert client.users == {} and client.server_config == {} and client.session is None
