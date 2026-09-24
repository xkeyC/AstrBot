"""WebRTC media through a proxy, over the peer's ICE-TCP candidates.

aiortc (aioice) only speaks ICE over UDP, which no HTTP or SOCKS proxy that
AstrBot is likely to be given can carry. The realtime peer is ICE-lite and
also offers passive TCP candidates on port 443, so media can instead ride one
TCP connection opened through the proxy.

``IceTcpRelay`` bridges the two: it listens on a local UDP port that is put
in the SDP answer as the peer's only candidate, and forwards every datagram
over the proxied TCP connection with RFC 4571 framing (a 2-byte length before
each packet), and back. STUN, DTLS and SRTP pass through untouched; ICE
checks authenticate with the ufrag/password, not addresses, so the peer does
not mind where the packets come from.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import re
import socket
import struct
from urllib.parse import unquote, urlsplit

from astrbot import logger

CONNECT_TIMEOUT = 10.0
_CANDIDATE = re.compile(
    r"^a=candidate:\S+ \d+ (?P<proto>udp|tcp) \d+ (?P<ip>\S+) (?P<port>\d+) typ host(?P<rest>.*)$",
    re.I,
)


def tcp_candidates(sdp: str) -> list[tuple[str, int]]:
    """The peer's passive ICE-TCP host candidates, in SDP order."""
    found = []
    for line in sdp.splitlines():
        match = _CANDIDATE.match(line.strip())
        if (
            match
            and match.group("proto").lower() == "tcp"
            and "tcptype passive" in match.group("rest")
        ):
            found.append((match.group("ip"), int(match.group("port"))))
    return found


def replace_candidates(sdp: str, host: str, port: int) -> str:
    """The SDP with the peer's candidates replaced by one UDP host candidate."""
    lines = [line for line in sdp.split("\r\n") if not line.startswith("a=candidate:")]
    candidate = f"a=candidate:1 1 udp 2130706431 {host} {port} typ host"
    out: list[str] = []
    for line in lines:
        out.append(line)
        # Candidates belong to the media section; add ours right after its m= line.
        if line.startswith("m="):
            out.append(candidate)
    return "\r\n".join(out)


async def open_proxied_tcp(
    proxy_url: str, host: str, port: int
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Opens a TCP connection to ``host:port`` through ``proxy_url``.

    Args:
        proxy_url: ``socks5://``, ``socks5h://`` or ``http://`` URL, optionally
            with ``user:password@``.
        host: Destination host (an IP address for ICE candidates).
        port: Destination port.

    Raises:
        ConnectionError: The proxy refused or failed the connection.
        ValueError: The proxy URL scheme is not supported.
    """
    url = urlsplit(proxy_url)
    scheme = url.scheme.lower()
    if scheme not in ("socks5", "socks5h", "http"):
        raise ValueError(f"unsupported proxy scheme for voice media: {url.scheme}")
    proxy_port = url.port or (1080 if scheme.startswith("socks") else 80)
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(url.hostname, proxy_port), CONNECT_TIMEOUT
    )
    user = unquote(url.username) if url.username else ""
    password = unquote(url.password) if url.password else ""
    try:
        if scheme == "http":
            auth = ""
            if user:
                token = base64.b64encode(f"{user}:{password}".encode()).decode()
                auth = f"Proxy-Authorization: Basic {token}\r\n"
            writer.write(
                f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n{auth}\r\n".encode()
            )
            await writer.drain()
            status = await asyncio.wait_for(reader.readline(), CONNECT_TIMEOUT)
            parts = status.split()
            if len(parts) < 2 or parts[1] != b"200":
                raise ConnectionError(
                    f"proxy CONNECT failed: {status.decode(errors='replace').strip()}"
                )
            while (await asyncio.wait_for(reader.readline(), CONNECT_TIMEOUT)) not in (
                b"\r\n",
                b"\n",
                b"",
            ):
                pass
        else:
            methods = b"\x00\x02" if user else b"\x00"
            writer.write(bytes([5, len(methods)]) + methods)
            await writer.drain()
            version, method = await asyncio.wait_for(
                reader.readexactly(2), CONNECT_TIMEOUT
            )
            if version != 5 or method == 0xFF:
                raise ConnectionError("SOCKS5 proxy offered no acceptable auth method")
            if method == 2:
                u, p = user.encode(), password.encode()
                writer.write(bytes([1, len(u)]) + u + bytes([len(p)]) + p)
                await writer.drain()
                _, ok = await asyncio.wait_for(reader.readexactly(2), CONNECT_TIMEOUT)
                if ok != 0:
                    raise ConnectionError("SOCKS5 proxy rejected the credentials")
            try:
                address = ipaddress.ip_address(host)
                target = (b"\x01" if address.version == 4 else b"\x04") + address.packed
            except ValueError:
                name = host.encode("idna")
                target = bytes([3, len(name)]) + name
            writer.write(b"\x05\x01\x00" + target + struct.pack(">H", port))
            await writer.drain()
            head = await asyncio.wait_for(reader.readexactly(4), CONNECT_TIMEOUT)
            if head[1] != 0:
                raise ConnectionError(f"SOCKS5 proxy connect failed (reply {head[1]})")
            skip = {1: 4, 4: 16}.get(head[3])
            if skip is None:
                skip = (await reader.readexactly(1))[0]
            await asyncio.wait_for(reader.readexactly(skip + 2), CONNECT_TIMEOUT)
    except BaseException:
        writer.close()
        raise
    return reader, writer


class IceTcpRelay(asyncio.DatagramProtocol):
    """A local UDP port that relays to the peer's ICE-TCP candidate."""

    def __init__(self, proxy_url: str, candidates: list[tuple[str, int]]) -> None:
        self.proxy_url = proxy_url
        self.candidates = candidates
        self.port = 0
        self._transport: asyncio.DatagramTransport | None = None
        self._local_peer: tuple[str, int] | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> int:
        """Connects through the proxy and binds the local UDP port.

        Returns:
            The local UDP port to advertise in the SDP answer.
        """
        loop = asyncio.get_running_loop()
        last_error: Exception | None = None
        for host, port in self.candidates:
            try:
                reader, writer = await open_proxied_tcp(self.proxy_url, host, port)
            except (OSError, ConnectionError, asyncio.TimeoutError) as exc:
                last_error = exc
                logger.warning(
                    "Mumble voice: ICE-TCP %s:%s via proxy failed: %s", host, port, exc
                )
                continue
            sock = writer.get_extra_info("socket")
            if sock is not None:
                # Media packets are small and time-critical: no Nagle batching.
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._writer = writer
            self._reader_task = asyncio.create_task(self._read(reader))
            break
        else:
            raise ConnectionError(
                f"no ICE-TCP candidate reachable through the proxy: {last_error}"
            )
        transport, _ = await loop.create_datagram_endpoint(
            lambda: self, local_addr=("0.0.0.0", 0)
        )
        self._transport = transport  # type: ignore[assignment]
        self.port = transport.get_extra_info("sockname")[1]
        return self.port

    def datagram_received(self, data: bytes, addr) -> None:
        # aioice checks every local candidate against us; answer the one that
        # talks, the pair ICE ends up nominating.
        self._local_peer = addr[:2]
        if self._writer is None or self._closed:
            return
        self._writer.write(struct.pack(">H", len(data)) + data)

    async def _read(self, reader: asyncio.StreamReader) -> None:
        try:
            while True:
                length = struct.unpack(">H", await reader.readexactly(2))[0]
                packet = await reader.readexactly(length)
                if self._transport is not None and self._local_peer is not None:
                    self._transport.sendto(packet, self._local_peer)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            if not self._closed:
                logger.warning(
                    "Mumble voice: ICE-TCP connection through the proxy closed"
                )
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._transport is not None:
            self._transport.close()
        if self._writer is not None:
            self._writer.close()
        if (
            self._reader_task is not None
            and self._reader_task is not asyncio.current_task()
        ):
            self._reader_task.cancel()
