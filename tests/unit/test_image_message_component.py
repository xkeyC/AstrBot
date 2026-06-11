import base64

import pytest

from astrbot.core.message.components import Image
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.pipeline.respond.stage import RespondStage  # noqa: F401
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)


@pytest.mark.asyncio
async def test_image_convert_to_base64_uses_path_fallback(tmp_path):
    image_path = tmp_path / "agent-image.png"
    image_path.write_bytes(b"image-bytes")

    image = Image(file="", path=str(image_path))

    assert await image.convert_to_base64() == base64.b64encode(b"image-bytes").decode()


@pytest.mark.asyncio
async def test_image_convert_to_base64_keeps_file_priority_over_unreadable_path(tmp_path):
    image_path = tmp_path / "agent-image.png"
    image_path.write_bytes(b"image-bytes")
    missing_path = tmp_path / "remote-cache-name.png"

    image = Image(file=str(image_path), path=str(missing_path))

    assert await image.convert_to_base64() == base64.b64encode(b"image-bytes").decode()


@pytest.mark.asyncio
async def test_image_convert_to_base64_supports_file_uri_variants(tmp_path):
    image_path = tmp_path / "agent image.png"
    image_path.write_bytes(b"image-bytes")
    expected = base64.b64encode(b"image-bytes").decode()

    image = Image(file=image_path.as_uri())
    assert await image.convert_to_base64() == expected

    localhost_uri = f"file://localhost{image_path.as_posix()}"
    image = Image(file=localhost_uri)
    assert await image.convert_to_base64() == expected


def test_image_decode_file_uri_normalizes_windows_backslashes():
    assert Image._decode_file_uri(r"file:///C:\Users\bot\img.png") == (
        "C:/Users/bot/img.png"
    )


@pytest.mark.asyncio
async def test_image_convert_to_base64_supports_data_url():
    image = Image(file="data:image/png;base64,aW1hZ2UtYnl0ZXM=")

    assert await image.convert_to_base64() == "aW1hZ2UtYnl0ZXM="


@pytest.mark.asyncio
async def test_aiocqhttp_image_segment_sends_path_fallback_as_base64(tmp_path):
    image_path = tmp_path / "agent-image.png"
    image_path.write_bytes(b"image-bytes")

    data = await AiocqhttpMessageEvent._parse_onebot_json(
        MessageChain([Image(file="", path=str(image_path))])
    )

    assert data == [
        {
            "type": "image",
            "data": {"file": "base64://aW1hZ2UtYnl0ZXM="},
        }
    ]
