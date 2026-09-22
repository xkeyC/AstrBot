"""Mumble control (TCP) and voice (UDP-format) message schemas.

Field numbers follow ``Mumble.proto`` and ``MumbleUDP.proto`` of Mumble 1.5.
Only the messages and fields the bot reads or writes are described; anything
else on the wire is skipped by the decoder.
"""

from __future__ import annotations

from enum import IntEnum

from .protobuf import Field, Schema


class MessageType(IntEnum):
    Version = 0
    UDPTunnel = 1
    Authenticate = 2
    Ping = 3
    Reject = 4
    ServerSync = 5
    ChannelRemove = 6
    ChannelState = 7
    UserRemove = 8
    UserState = 9
    BanList = 10
    TextMessage = 11
    PermissionDenied = 12
    ACL = 13
    QueryUsers = 14
    CryptSetup = 15
    ContextActionModify = 16
    ContextAction = 17
    UserList = 18
    VoiceTarget = 19
    PermissionQuery = 20
    CodecVersion = 21
    UserStats = 22
    RequestBlob = 23
    ServerConfig = 24
    SuggestConfig = 25
    PluginDataTransmission = 26


def version_v2(major: int, minor: int, patch: int) -> int:
    return major << 48 | minor << 32 | patch << 16


def version_v1(major: int, minor: int, patch: int) -> int:
    return major << 16 | min(minor, 0xFF) << 8 | min(patch, 0xFF)


def _schema(*fields: tuple[int, str, str] | tuple[int, str, str, bool]) -> Schema:
    return {spec[0]: Field(spec[1], spec[2], *spec[3:]) for spec in fields}


VERSION = _schema(
    (1, "version_v1", "uint32"),
    (5, "version_v2", "uint64"),
    (2, "release", "string"),
    (3, "os", "string"),
    (4, "os_version", "string"),
)

UDP_TUNNEL = _schema((1, "packet", "bytes"))

AUTHENTICATE = _schema(
    (1, "username", "string"),
    (2, "password", "string"),
    (3, "tokens", "string", True),
    (4, "celt_versions", "int32", True),
    (5, "opus", "bool"),
    (6, "client_type", "int32"),
)

PING = _schema(
    (1, "timestamp", "uint64"),
    (2, "good", "uint32"),
    (3, "late", "uint32"),
    (4, "lost", "uint32"),
    (5, "resync", "uint32"),
    (6, "udp_packets", "uint32"),
    (7, "tcp_packets", "uint32"),
    (8, "udp_ping_avg", "float"),
    (9, "udp_ping_var", "float"),
    (10, "tcp_ping_avg", "float"),
    (11, "tcp_ping_var", "float"),
)

REJECT = _schema((1, "type", "enum"), (2, "reason", "string"))

SERVER_SYNC = _schema(
    (1, "session", "uint32"),
    (2, "max_bandwidth", "uint32"),
    (3, "welcome_text", "string"),
    (4, "permissions", "uint64"),
)

CHANNEL_REMOVE = _schema((1, "channel_id", "uint32"))

CHANNEL_STATE = _schema(
    (1, "channel_id", "uint32"),
    (2, "parent", "uint32"),
    (3, "name", "string"),
    (4, "links", "uint32", True),
    (5, "description", "string"),
    (6, "links_add", "uint32", True),
    (7, "links_remove", "uint32", True),
    (8, "temporary", "bool"),
    (9, "position", "int32"),
    (11, "max_users", "uint32"),
)

USER_REMOVE = _schema(
    (1, "session", "uint32"),
    (2, "actor", "uint32"),
    (3, "reason", "string"),
    (4, "ban", "bool"),
)

USER_STATE = _schema(
    (1, "session", "uint32"),
    (2, "actor", "uint32"),
    (3, "name", "string"),
    (4, "user_id", "uint32"),
    (5, "channel_id", "uint32"),
    (6, "mute", "bool"),
    (7, "deaf", "bool"),
    (8, "suppress", "bool"),
    (9, "self_mute", "bool"),
    (10, "self_deaf", "bool"),
    (14, "comment", "string"),
    (15, "hash", "string"),
    (18, "priority_speaker", "bool"),
    (19, "recording", "bool"),
)

TEXT_MESSAGE = _schema(
    (1, "actor", "uint32"),
    (2, "session", "uint32", True),
    (3, "channel_id", "uint32", True),
    (4, "tree_id", "uint32", True),
    (5, "message", "string"),
)

PERMISSION_DENIED = _schema(
    (1, "permission", "uint32"),
    (2, "channel_id", "uint32"),
    (3, "session", "uint32"),
    (4, "reason", "string"),
    (5, "type", "enum"),
    (6, "name", "string"),
)

CRYPT_SETUP = _schema(
    (1, "key", "bytes"),
    (2, "client_nonce", "bytes"),
    (3, "server_nonce", "bytes"),
)

VOICE_TARGET_TARGET = _schema(
    (1, "session", "uint32", True),
    (2, "channel_id", "uint32"),
    (3, "group", "string"),
    (4, "links", "bool"),
    (5, "children", "bool"),
)

VOICE_TARGET: Schema = {
    1: Field("id", "uint32"),
    2: Field("targets", "message", True, VOICE_TARGET_TARGET),
}

CODEC_VERSION = _schema(
    (1, "alpha", "int32"),
    (2, "beta", "int32"),
    (3, "prefer_alpha", "bool"),
    (4, "opus", "bool"),
)

SERVER_CONFIG = _schema(
    (1, "max_bandwidth", "uint32"),
    (2, "welcome_text", "string"),
    (3, "allow_html", "bool"),
    (4, "message_length", "uint32"),
    (5, "image_message_length", "uint32"),
    (6, "max_users", "uint32"),
    (7, "recording_allowed", "bool"),
)

SCHEMAS: dict[MessageType, Schema] = {
    MessageType.Version: VERSION,
    MessageType.UDPTunnel: UDP_TUNNEL,
    MessageType.Authenticate: AUTHENTICATE,
    MessageType.Ping: PING,
    MessageType.Reject: REJECT,
    MessageType.ServerSync: SERVER_SYNC,
    MessageType.ChannelRemove: CHANNEL_REMOVE,
    MessageType.ChannelState: CHANNEL_STATE,
    MessageType.UserRemove: USER_REMOVE,
    MessageType.UserState: USER_STATE,
    MessageType.TextMessage: TEXT_MESSAGE,
    MessageType.PermissionDenied: PERMISSION_DENIED,
    MessageType.CryptSetup: CRYPT_SETUP,
    MessageType.VoiceTarget: VOICE_TARGET,
    MessageType.CodecVersion: CODEC_VERSION,
    MessageType.ServerConfig: SERVER_CONFIG,
}

# MumbleUDP.proto; the first byte of a voice packet selects the message.
UDP_AUDIO = 0
UDP_PING = 1

AUDIO = _schema(
    (1, "target", "uint32"),
    (2, "context", "uint32"),
    (3, "sender_session", "uint32"),
    (4, "frame_number", "uint64"),
    (5, "opus_data", "bytes"),
    (6, "positional_data", "float", True),
    (7, "volume_adjustment", "float"),
    (16, "is_terminator", "bool"),
)


class AudioContext(IntEnum):
    """``Audio.context`` on received voice: how it reached us."""

    NORMAL = 0
    SHOUT = 1  # sent to a channel target
    WHISPER = 2  # sent to us directly
    LISTEN = 3  # heard through a channel listener


class AudioTarget(IntEnum):
    NORMAL = 0  # the current channel
    LOOPBACK = 31  # server echoes it back to the sender
