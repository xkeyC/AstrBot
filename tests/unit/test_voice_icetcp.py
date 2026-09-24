import asyncio
import base64
import struct

import pytest

from astrbot.core.voice.icetcp import (
    IceTcpRelay,
    open_proxied_tcp,
    replace_candidates,
    tcp_candidates,
)

ANSWER = "\r\n".join(
    [
        "v=0",
        "a=ice-lite",
        "m=audio 9 UDP/TLS/RTP/SAVPF 96 0 8",
        "a=candidate:1 1 udp 2130706431 40.118.236.137 3478 typ host ufrag x",
        "a=candidate:2 1 tcp 1671430143 40.118.236.137 443 typ host tcptype passive ufrag x",
        "a=candidate:3 1 tcp 1671430143 20.168.48.117 443 typ host tcptype passive ufrag x",
        "a=setup:passive",
        "",
    ]
)


def test_candidates_are_found_and_replaced():
    assert tcp_candidates(ANSWER) == [("40.118.236.137", 443), ("20.168.48.117", 443)]
    rewritten = replace_candidates(ANSWER, "192.168.1.2", 5000)
    lines = rewritten.split("\r\n")
    assert [line for line in lines if line.startswith("a=candidate")] == [
        "a=candidate:1 1 udp 2130706431 192.168.1.2 5000 typ host"
    ]
    assert lines.index("a=candidate:1 1 udp 2130706431 192.168.1.2 5000 typ host") == (
        lines.index("m=audio 9 UDP/TLS/RTP/SAVPF 96 0 8") + 1
    )
    assert "a=setup:passive" in lines


async def framed_echo(reader, writer):
    """The far end: echoes RFC 4571 frames back."""
    try:
        while True:
            length = struct.unpack(">H", await reader.readexactly(2))[0]
            data = await reader.readexactly(length)
            writer.write(struct.pack(">H", len(data)) + data)
            await writer.drain()
    except asyncio.IncompleteReadError:
        writer.close()


async def fake_socks5(reader, writer, seen):
    greeting = await reader.readexactly(2)
    methods = await reader.readexactly(greeting[1])
    if b"\x02" in methods:
        writer.write(b"\x05\x02")
        _, ulen = await reader.readexactly(2)
        user = await reader.readexactly(ulen)
        plen = (await reader.readexactly(1))[0]
        password = await reader.readexactly(plen)
        seen["auth"] = (user, password)
        writer.write(b"\x01\x00")
    else:
        writer.write(b"\x05\x00")
    head = await reader.readexactly(4)
    address = await reader.readexactly(4 if head[3] == 1 else 16)
    port = struct.unpack(">H", await reader.readexactly(2))[0]
    seen["target"] = (".".join(map(str, address)), port)
    writer.write(b"\x05\x00\x00\x01" + bytes(4) + b"\x00\x00")
    await framed_echo(reader, writer)


async def fake_http_proxy(reader, writer, seen):
    request = b""
    while not request.endswith(b"\r\n\r\n"):
        request += await reader.read(1)
    seen["request"] = request.decode()
    writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
    await framed_echo(reader, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scheme", "handler", "userinfo"),
    [
        ("socks5", fake_socks5, ""),
        ("socks5", fake_socks5, "u:p@"),
        ("http", fake_http_proxy, "u:p@"),
    ],
)
async def test_relay_round_trips_datagrams_through_the_proxy(scheme, handler, userinfo):
    seen: dict = {}
    server = await asyncio.start_server(
        lambda r, w: handler(r, w, seen), "127.0.0.1", 0
    )
    proxy_port = server.sockets[0].getsockname()[1]
    relay = IceTcpRelay(
        f"{scheme}://{userinfo}127.0.0.1:{proxy_port}", [("40.118.236.137", 443)]
    )
    port = await relay.start()

    received: asyncio.Queue = asyncio.Queue()

    class Peer(asyncio.DatagramProtocol):
        def datagram_received(self, data, addr):
            received.put_nowait((data, addr))

    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        Peer, local_addr=("127.0.0.1", 0)
    )
    for packet in (b"stun-binding", b"\x16dtls-hello", b"\x80" + bytes(160)):
        transport.sendto(packet, ("127.0.0.1", port))
        data, addr = await asyncio.wait_for(received.get(), 5)
        assert data == packet
        assert addr[1] == port
    if scheme == "http":
        assert seen["request"].startswith("CONNECT 40.118.236.137:443 HTTP/1.1\r\n")
        token = base64.b64encode(b"u:p").decode()
        assert f"Proxy-Authorization: Basic {token}" in seen["request"]
    else:
        assert seen["target"] == ("40.118.236.137", 443)
        assert seen.get("auth") == ((b"u", b"p") if userinfo else None)
    transport.close()
    relay.close()
    server.close()


@pytest.mark.asyncio
async def test_unreachable_proxy_and_bad_scheme():
    with pytest.raises(ValueError):
        await open_proxied_tcp("https://127.0.0.1:1", "1.2.3.4", 443)
    relay = IceTcpRelay("socks5://127.0.0.1:1", [("1.2.3.4", 443)])
    with pytest.raises(ConnectionError):
        await relay.start()
