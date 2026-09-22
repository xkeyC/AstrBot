"""Mumble text message formatting.

Text messages are HTML. The official client sends Markdown rendered to HTML
(or escaped plain text with ``<br>`` and ``&nbsp;``) and embeds images as
``data:`` URIs; other clients send plain text or ``<p>``-wrapped HTML. There
is no mention primitive on the wire.
"""

from __future__ import annotations

import base64
import binascii
import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser


@dataclass
class InlineImage:
    mime: str
    data: bytes


@dataclass
class ParsedText:
    text: str
    images: list[InlineImage] = field(default_factory=list)


_BLOCK_TAGS = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre"}
_DATA_URI = re.compile(r"data:([\w/+.-]+);base64,(.*)", re.S)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.images: list[InlineImage] = []
        self._href: str | None = None
        self._link_text: list[str] = []
        self._skip = 0

    def _newline(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in ("style", "script", "head"):
            self._skip += 1
        elif tag == "br":
            self.parts.append("\n")
        elif tag in _BLOCK_TAGS:
            self._newline()
            if tag == "li":
                self.parts.append("- ")
        elif (
            tag in ("td", "th")
            and self.parts
            and not self.parts[-1].endswith(("\n", "\t"))
        ):
            self.parts.append("\t")
        elif tag == "img":
            self._image(attributes.get("src") or "")
        elif tag == "a":
            self._href = attributes.get("href")
            self._link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("style", "script", "head"):
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self._newline()
        elif tag == "a" and self._href:
            label = "".join(self._link_text).strip()
            if self._href not in (label, f"http://{label}", f"https://{label}"):
                self.parts.append(f" ({self._href})")
            self._href = None

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        data = data.replace("\xa0", " ")
        if self._href is not None:
            self._link_text.append(data)
        self.parts.append(data)

    def _image(self, src: str) -> None:
        match = _DATA_URI.match(src.strip())
        if match is None:
            if src:
                self.parts.append(f"[image: {src}]")
            return
        try:
            data = base64.b64decode(re.sub(r"\s+", "", match.group(2)), validate=True)
        except (binascii.Error, ValueError):
            return
        self.images.append(InlineImage(match.group(1), data))


def html_to_text(message: str) -> ParsedText:
    """Plain text and inline images of a received text message."""
    parser = _TextExtractor()
    parser.feed(message)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return ParsedText(text.strip(), parser.images)


def escape_text(text: str) -> str:
    """Plain text as Mumble HTML: keeps line breaks and runs of spaces."""
    escaped = html.escape(text.replace("\r\n", "\n").replace("\r", "\n"), quote=False)
    escaped = re.sub(r"  +", lambda m: " " + "&nbsp;" * (len(m.group()) - 1), escaped)
    return escaped.replace("\n", "<br>")


_FENCE = re.compile(r"```[^\n]*\n(.*?)(?:```\n?|\Z)", re.S)
# Spans that other rules must not touch, rendered first and stashed.
_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^()\s\"'<>]+)\)")
_URL = re.compile(r"https?://[^\s\"'<>]+")
_EMPHASIS = [
    (re.compile(r"\*\*([^*\n]+)\*\*"), r"<b>\1</b>"),
    (re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])"), r"<i>\1</i>"),
    (re.compile(r"~~([^~\n]+)~~"), r"<s>\1</s>"),
]
_HEADING = re.compile(r"^(#{1,6}) +(.*)$", re.M)


def markdown_to_html(text: str) -> str:
    """The Markdown subset the official client renders, for model replies."""
    out: list[str] = []
    pos = 0
    for match in _FENCE.finditer(text):
        out.append(_markdown_inline(text[pos : match.start()]))
        code = html.escape(match.group(1).rstrip("\n"))
        out.append(f"<pre>{code}</pre>")
        pos = match.end()
    out.append(_markdown_inline(text[pos:]))
    return "".join(out)


def _markdown_inline(text: str) -> str:
    if not text:
        return ""
    stash: list[str] = []

    def keep(rendered: str) -> str:
        stash.append(rendered)
        return f"\x00{len(stash) - 1}\x00"

    # On the raw text: code, links and URLs are rendered (escaped) and stashed
    # before escaping and emphasis, so neither can break or nest into them.
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = _CODE.sub(lambda m: keep(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = _LINK.sub(
        lambda m: keep(
            f'<a href="{html.escape(m.group(2))}">{html.escape(m.group(1))}</a>'
        ),
        text,
    )
    text = _URL.sub(
        lambda m: keep(
            f'<a href="{html.escape(m.group())}">{html.escape(m.group())}</a>'
        ),
        text,
    )
    escaped = html.escape(text)
    escaped = _HEADING.sub(lambda m: f"<b>{m.group(2)}</b>", escaped)
    for pattern, replacement in _EMPHASIS:
        escaped = pattern.sub(replacement, escaped)
    escaped = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], escaped)
    return escaped.replace("\n", "<br>")


def image_html(data: bytes, mime: str = "image/png") -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f'<img src="data:{mime};base64,{encoded}"/>'
