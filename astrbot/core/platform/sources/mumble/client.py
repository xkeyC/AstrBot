"""Asyncio Mumble client: TLS control channel with voice tunnelled over it.

Scope: Mumble 1.5+ servers. The client announces 1.5 so the server sends
voice in the protobuf ("MumbleUDP") format, and it never opens the UDP
socket: servers fall back to tunnelling voice through ``UDPTunnel`` on the
TCP connection, which is what every server supports.
"""

from __future__ import annotations

import asyncio
import platform
import ssl
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from astrbot import logger

from . import protobuf
from .messages import (
    AUDIO,
    SCHEMAS,
    UDP_AUDIO,
    UDP_PING,
    AudioContext,
    AudioTarget,
    MessageType,
    version_v1,
    version_v2,
)

PROTOCOL_VERSION = (1, 5, 0)
CLIENT_TYPE_BOT = 1
PING_INTERVAL = 15.0
# No message at all from the server for this long means the connection is
# dead even if TCP has not noticed (NAT timeout, server host gone).
RECEIVE_TIMEOUT = 60.0
# The server's hard cap on a control message is 8 MiB; images in text
# messages are the only large ones.
MAX_MESSAGE_SIZE = 8 * 1024 * 1024
_HEADER = struct.Struct(">HI")


class MumbleError(Exception):
    pass


class MumbleRejected(MumbleError):
    def __init__(self, reject_type: int | None, reason: str):
        super().__init__(f"server rejected the connection ({reject_type}): {reason}")
        self.reject_type = reject_type
        self.reason = reason


@dataclass
class Channel:
    channel_id: int
    name: str = ""
    parent: int | None = None
    description: str = ""
    links: set[int] = field(default_factory=set)
    temporary: bool = False
    position: int = 0


@dataclass
class User:
    session: int
    name: str = ""
    user_id: int | None = None
    channel_id: int = 0
    mute: bool = False
    deaf: bool = False
    suppress: bool = False
    self_mute: bool = False
    self_deaf: bool = False
    comment: str = ""
    hash: str = ""


@dataclass
class TextMessage:
    actor: int | None
    message: str
    sessions: list[int]
    channel_ids: list[int]
    tree_ids: list[int]

    @property
    def is_private(self) -> bool:
        return bool(self.sessions) and not self.channel_ids and not self.tree_ids


@dataclass
class VoicePacket:
    sender_session: int
    context: AudioContext | int
    frame_number: int
    opus_data: bytes
    is_terminator: bool
    volume_adjustment: float | None = None


def encode_audio(
    opus_data: bytes, frame_number: int, target: int, is_terminator: bool
) -> bytes:
    message: dict[str, Any] = {
        "target": target,
        "frame_number": frame_number,
        "opus_data": opus_data,
    }
    if is_terminator:
        message["is_terminator"] = True
    return bytes([UDP_AUDIO]) + protobuf.encode(AUDIO, message)


def decode_audio(packet: bytes) -> VoicePacket | None:
    """Decodes a received voice packet; ``None`` for non-audio packets."""
    if not packet or packet[0] != UDP_AUDIO:
        return None
    message = protobuf.decode(AUDIO, memoryview(packet)[1:])
    context = message.get("context", 0)
    try:
        context = AudioContext(context)
    except ValueError:
        pass
    return VoicePacket(
        sender_session=message.get("sender_session", 0),
        context=context,
        frame_number=message.get("frame_number", 0),
        opus_data=message.get("opus_data", b""),
        is_terminator=message.get("is_terminator", False),
        volume_adjustment=message.get("volume_adjustment"),
    )


class MumbleClient:
    """One connection to a Mumble server.

    Callbacks run on the event loop and must not block:

    - ``on_text(TextMessage)``
    - ``on_voice(VoicePacket)``: one Opus frame, up to 50 per speaker per second
    - ``on_user_changed(User, changed_fields)`` / ``on_user_removed(User, dict)``
    - ``on_disconnected(Exception | None)``
    """

    def __init__(
        self,
        host: str,
        port: int = 64738,
        username: str = "AstrBot",
        password: str = "",
        tokens: list[str] | None = None,
        certfile: str | None = None,
        keyfile: str | None = None,
        verify_server: bool = False,
        release: str = "AstrBot",
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.tokens = tokens or []
        self.certfile = certfile
        self.keyfile = keyfile
        self.verify_server = verify_server
        self.release = release

        self.session: int | None = None
        self.channels: dict[int, Channel] = {}
        self.users: dict[int, User] = {}
        self.server_version: dict[str, Any] = {}
        self.server_config: dict[str, Any] = {}
        self.welcome_text = ""
        self.max_bandwidth: int | None = None
        self.tcp_ping_ms: float | None = None

        self.on_text: Callable[[TextMessage], None] | None = None
        self.on_voice: Callable[[VoicePacket], None] | None = None
        self.on_user_changed: Callable[[User, set[str]], None] | None = None
        self.on_user_removed: Callable[[User, dict[str, Any]], None] | None = None
        self.on_disconnected: Callable[[Exception | None], None] | None = None

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._tasks: list[asyncio.Task] = []
        self._synced: asyncio.Future[None] | None = None
        self._frame_number = 0
        self._audio_epoch: float | None = None
        self._closing = False
        self._last_received = 0.0

    # -- connection -------------------------------------------------------

    def _ssl_context(self) -> ssl.SSLContext:
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        if not self.verify_server:
            # Most Mumble servers use self-signed certificates.
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        if self.certfile:
            context.load_cert_chain(self.certfile, self.keyfile)
        return context

    async def connect(self, timeout: float = 15.0) -> None:
        """Connects and returns once the server has sent ``ServerSync``."""
        loop = asyncio.get_running_loop()
        self._closing = False
        self._synced = loop.create_future()
        # The server resends all state after connecting; what is left from a
        # previous connection is stale (and session ids get reused).
        self.session = None
        self.users.clear()
        self.channels.clear()
        self.server_config.clear()
        self._last_received = time.monotonic()
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(
                self.host,
                self.port,
                ssl=self._ssl_context(),
                server_hostname=self.host if self.verify_server else None,
            ),
            timeout,
        )
        major, minor, patch = PROTOCOL_VERSION
        self.send(
            MessageType.Version,
            {
                "version_v1": version_v1(major, minor, patch),
                "version_v2": version_v2(major, minor, patch),
                "release": self.release,
                "os": platform.system(),
                "os_version": platform.release(),
            },
        )
        self.send(
            MessageType.Authenticate,
            {
                "username": self.username,
                "password": self.password or None,
                "tokens": self.tokens or None,
                "opus": True,
                "client_type": CLIENT_TYPE_BOT,
            },
        )
        self._tasks = [
            asyncio.create_task(self._read_loop(), name="mumble-read"),
            asyncio.create_task(self._ping_loop(), name="mumble-ping"),
        ]
        try:
            await asyncio.wait_for(asyncio.shield(self._synced), timeout)
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        self._closing = True
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current:
                task.cancel()
        self._tasks = []
        if self._writer is not None:
            writer, self._writer = self._writer, None
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), 2)
            except (asyncio.TimeoutError, OSError, ssl.SSLError):
                pass

    @property
    def server_release(self) -> str:
        """Server version for display; servers may omit the release string."""
        version = self.server_version
        if version.get("release"):
            return version["release"]
        if v2 := version.get("version_v2"):
            return f"{v2 >> 48}.{(v2 >> 32) & 0xFFFF}.{(v2 >> 16) & 0xFFFF}"
        if v1 := version.get("version_v1"):
            return f"{v1 >> 16}.{(v1 >> 8) & 0xFF}.{v1 & 0xFF}"
        return "unknown"

    @property
    def connected(self) -> bool:
        return self._writer is not None and self.session is not None

    # -- sending ----------------------------------------------------------

    def send(self, message_type: MessageType, message: dict[str, Any]) -> None:
        payload = protobuf.encode(SCHEMAS[message_type], message)
        self._send_raw(message_type, payload)

    def _send_raw(self, message_type: MessageType, payload: bytes) -> None:
        if self._writer is None:
            raise MumbleError("not connected")
        self._writer.write(_HEADER.pack(message_type, len(payload)) + payload)

    async def drain(self) -> None:
        if self._writer is not None:
            await self._writer.drain()

    def send_text(
        self,
        message: str,
        *,
        channel_ids: list[int] | None = None,
        sessions: list[int] | None = None,
        tree_ids: list[int] | None = None,
    ) -> None:
        """Mumble text messages are HTML; escape plain text before sending."""
        self.send(
            MessageType.TextMessage,
            {
                "message": message,
                "channel_id": channel_ids or None,
                "session": sessions or None,
                "tree_id": tree_ids or None,
            },
        )

    def send_audio(
        self,
        opus_data: bytes,
        *,
        target: int = AudioTarget.NORMAL,
        is_terminator: bool = False,
    ) -> None:
        """Sends one 20 ms Opus frame. End each transmission with ``is_terminator``.

        ``frame_number`` counts 10 ms of wall-clock time, as the official
        client's counter does (it keeps running through silence): receivers
        schedule playback by it, so after a pause it must jump ahead, or the
        new frames look late and get dropped.
        """
        now = time.monotonic()
        if self._audio_epoch is None:
            self._audio_epoch = now
        elapsed = int((now - self._audio_epoch) * 100)
        # Never backwards, and 2 per frame when frames are sent back to back.
        self._frame_number = max(self._frame_number, elapsed)
        packet = encode_audio(opus_data, self._frame_number, target, is_terminator)
        self._frame_number += 2
        self._send_raw(MessageType.UDPTunnel, packet)

    def set_voice_target(
        self,
        target_id: int,
        *,
        sessions: list[int] | None = None,
        channel_id: int | None = None,
        children: bool = False,
    ) -> None:
        """Registers whisper target ``target_id`` (1-30) for ``send_audio``."""
        if not 1 <= target_id <= 30:
            raise ValueError("voice target id must be within 1..30")
        target: dict[str, Any] = {}
        if sessions:
            target["session"] = sessions
        if channel_id is not None:
            target["channel_id"] = channel_id
            target["children"] = children or None
        self.send(MessageType.VoiceTarget, {"id": target_id, "targets": [target]})

    def join_channel(self, channel_id: int) -> None:
        self.send(
            MessageType.UserState, {"session": self.session, "channel_id": channel_id}
        )

    def set_self_state(
        self,
        *,
        self_mute: bool | None = None,
        self_deaf: bool | None = None,
        comment: str | None = None,
    ) -> None:
        self.send(
            MessageType.UserState,
            {
                "session": self.session,
                "self_mute": self_mute,
                "self_deaf": self_deaf,
                "comment": comment,
            },
        )

    # -- lookups ----------------------------------------------------------

    @property
    def me(self) -> User | None:
        return self.users.get(self.session) if self.session is not None else None

    def find_channel(self, name_or_path: str) -> Channel | None:
        """By name, or by ``a/b/c`` path from the root channel."""
        if "/" not in name_or_path:
            return next(
                (c for c in self.channels.values() if c.name == name_or_path), None
            )
        current = self.channels.get(0)
        for part in filter(None, name_or_path.split("/")):
            if current is None:
                return None
            current = next(
                (
                    c
                    for c in self.channels.values()
                    if c.parent == current.channel_id and c.name == part
                ),
                None,
            )
        return current

    def find_user(self, name: str) -> User | None:
        return next((u for u in self.users.values() if u.name == name), None)

    # -- receiving --------------------------------------------------------

    async def _read_loop(self) -> None:
        assert self._reader is not None
        error: Exception | None = None
        try:
            while True:
                header = await self._reader.readexactly(_HEADER.size)
                message_type, length = _HEADER.unpack(header)
                if length > MAX_MESSAGE_SIZE:
                    raise MumbleError(f"message of {length} bytes is too large")
                payload = await self._reader.readexactly(length)
                self._last_received = time.monotonic()
                self._dispatch(message_type, payload)
        except asyncio.CancelledError:
            raise
        except asyncio.IncompleteReadError:
            error = None if self._closing else MumbleError("server closed connection")
        except Exception as exc:  # noqa: BLE001 - reported to the owner
            error = exc
        if self._synced is not None and not self._synced.done():
            self._synced.set_exception(error or MumbleError("disconnected"))
        self.session = None
        if not self._closing:
            await self.close()
            if self.on_disconnected is not None:
                self.on_disconnected(error)

    def _dispatch(self, raw_type: int, payload: bytes) -> None:
        """Handles one message; a bad message or a failing callback is logged
        and skipped rather than taking the connection down."""
        try:
            message_type = MessageType(raw_type)
        except ValueError:
            return
        try:
            if message_type is MessageType.UDPTunnel:
                # The tunnel carries the raw voice packet, not a protobuf message.
                self._handle_voice(payload)
                return
            schema = SCHEMAS.get(message_type)
            if schema is None:
                return
            message = protobuf.decode(schema, payload)
            handler = getattr(self, f"_on_{message_type.name}", None)
            if handler is not None:
                handler(message)
        except protobuf.DecodeError as exc:
            logger.debug("Mumble: dropping malformed %s: %s", message_type.name, exc)
        except Exception:  # noqa: BLE001 - a callback bug must not disconnect
            logger.exception("Mumble: handling %s failed", message_type.name)

    def _handle_voice(self, packet: bytes) -> None:
        if not packet or packet[0] == UDP_PING:
            return
        voice = decode_audio(packet)
        if voice is not None and self.on_voice is not None:
            self.on_voice(voice)

    def _on_Version(self, message: dict[str, Any]) -> None:
        self.server_version = message

    def _on_Reject(self, message: dict[str, Any]) -> None:
        error = MumbleRejected(message.get("type"), message.get("reason", ""))
        if self._synced is not None and not self._synced.done():
            self._synced.set_exception(error)

    def _on_ServerSync(self, message: dict[str, Any]) -> None:
        self.session = message.get("session")
        self.max_bandwidth = message.get("max_bandwidth")
        self.welcome_text = message.get("welcome_text", "")
        if self._synced is not None and not self._synced.done():
            self._synced.set_result(None)

    def _on_ServerConfig(self, message: dict[str, Any]) -> None:
        self.server_config.update(message)

    def _on_Ping(self, message: dict[str, Any]) -> None:
        sent = message.get("timestamp")
        if sent:
            self.tcp_ping_ms = (time.monotonic_ns() // 1000 - sent) / 1000

    def _on_ChannelState(self, message: dict[str, Any]) -> None:
        channel_id = message.get("channel_id", 0)
        channel = self.channels.setdefault(channel_id, Channel(channel_id))
        for key in ("name", "parent", "description", "temporary", "position"):
            if key in message:
                setattr(channel, key, message[key])
        if "links" in message:
            channel.links = set(message["links"])
        channel.links.update(message.get("links_add", ()))
        channel.links.difference_update(message.get("links_remove", ()))

    def _on_ChannelRemove(self, message: dict[str, Any]) -> None:
        self.channels.pop(message.get("channel_id", -1), None)

    def _on_UserState(self, message: dict[str, Any]) -> None:
        session = message.get("session")
        if session is None:
            return
        user = self.users.setdefault(session, User(session))
        changed = set()
        for key in (
            "name",
            "user_id",
            "channel_id",
            "mute",
            "deaf",
            "suppress",
            "self_mute",
            "self_deaf",
            "comment",
            "hash",
        ):
            if key in message and getattr(user, key) != message[key]:
                setattr(user, key, message[key])
                changed.add(key)
        if changed and self.on_user_changed is not None:
            self.on_user_changed(user, changed)

    def _on_UserRemove(self, message: dict[str, Any]) -> None:
        user = self.users.pop(message.get("session", -1), None)
        if user is not None and self.on_user_removed is not None:
            self.on_user_removed(user, message)

    def _on_TextMessage(self, message: dict[str, Any]) -> None:
        if self.on_text is None:
            return
        self.on_text(
            TextMessage(
                actor=message.get("actor"),
                message=message.get("message", ""),
                sessions=message.get("session", []),
                channel_ids=message.get("channel_id", []),
                tree_ids=message.get("tree_id", []),
            )
        )

    def _on_PermissionDenied(self, message: dict[str, Any]) -> None:
        logger.warning("Mumble: permission denied: %s", message)

    async def _ping_loop(self) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL)
            if time.monotonic() - self._last_received > RECEIVE_TIMEOUT:
                logger.warning(
                    "Mumble: server silent for %.0fs, reconnecting", RECEIVE_TIMEOUT
                )
                if self._writer is not None:
                    # Aborting fails the pending read, which reports the
                    # disconnect through the normal path.
                    self._writer.transport.abort()
                return
            try:
                self.send(MessageType.Ping, {"timestamp": time.monotonic_ns() // 1000})
                await self.drain()
            except (MumbleError, OSError, ConnectionError):
                return
