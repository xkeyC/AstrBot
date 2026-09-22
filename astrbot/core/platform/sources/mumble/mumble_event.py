from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import At, Image, Plain

from .text import image_html, markdown_to_html

if TYPE_CHECKING:
    from .mumble_adapter import MumblePlatformAdapter

# Mumble servers default to 5000 characters per text message.
DEFAULT_MESSAGE_LENGTH = 5000
DEFAULT_IMAGE_MESSAGE_LENGTH = 131072


async def chain_to_html(
    chain: MessageChain, message_length: int, image_length: int
) -> list[str]:
    """Renders a message chain as Mumble HTML messages within the size limits.

    Args:
        chain: The chain to render.
        message_length: Server limit for a text message, in characters.
        image_length: Server limit for a message carrying an image.

    Returns:
        HTML messages in sending order; text is split to fit ``message_length``.
    """
    messages: list[str] = []
    text = ""
    for component in chain.chain:
        if isinstance(component, Plain):
            text += component.text
        elif isinstance(component, At):
            text += f"@{component.name or component.qq}"
        elif isinstance(component, Image):
            try:
                path = await component.convert_to_file_path()
                data = Path(path).read_bytes()
            except Exception as exc:  # noqa: BLE001 - fall back to a note
                logger.warning("Mumble: cannot read image to send: %s", exc)
                text += "[image]"
                continue
            mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
            rendered = image_html(data, mime)
            if len(rendered) > image_length:
                text += "[image too large for this server]"
                continue
            if text.strip():
                messages.extend(_split_text(text, message_length))
                text = ""
            messages.append(rendered)
    if text.strip():
        messages.extend(_split_text(text, message_length))
    return messages


def _split_text(text: str, limit: int) -> list[str]:
    rendered = markdown_to_html(text)
    if len(rendered) <= limit:
        return [rendered]
    # Too long for one message: send plain escaped parts, split on lines, so no
    # Markdown span is cut in half. Escaping grows text at most fivefold.

    def plain(part: str) -> str:
        return html.escape(part, quote=False).replace("\n", "<br>")

    step = max(1, limit // 5)
    parts: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        pieces = (
            [line[i : i + step] for i in range(0, len(line), step)]
            if len(plain(line)) > limit
            else [line]
        )
        for piece in pieces:
            if current and len(plain(current + piece)) > limit:
                parts.append(plain(current))
                current = ""
            current += piece
    if current.strip():
        parts.append(plain(current))
    return parts


class MumbleMessageEvent(AstrMessageEvent):
    def __init__(
        self,
        message_str,
        message_obj,
        platform_meta,
        session_id,
        adapter: MumblePlatformAdapter,
    ) -> None:
        super().__init__(message_str, message_obj, platform_meta, session_id)
        self.adapter = adapter

    async def send(self, message: MessageChain) -> None:
        await self.adapter.send_chain(self.get_session_id(), message)
        await super().send(message)
