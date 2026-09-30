"""Strip hidden markup that models leak into chat replies.

Three families show up in Codex replies and are never meant for the user:

- ``<oai-mem-citation>...</oai-mem-citation>`` memory citations. Codex core
  already strips these; this is a safety net.
- ChatGPT content references framed by private-use characters, e.g.
  ``\\ue200cite\\ue202turn0search6\\ue202turn1search4\\ue201`` (also ``filecite``,
  ``navlist`` ...; an ``entity`` keeps its display name), plus the bare
  ``citeturn0search6`` left over when a transport drops the private-use
  characters, and ``【F:a.md†L1-L2】``.
- Codex App directives ``:name{key="value" ...}`` (one to three colons), e.g.
  ``:codex-file-citation{path="..."}`` and ``::git-push{cwd="..."}``. Only the
  Codex TUI understands them. ``::code-comment{...}`` is rewritten the way the
  TUI shows it; every other directive is removed.

``ModelMarkupStream`` works on streamed deltas (markers may be split across
chunks); ``strip_model_markup`` is the one-shot form. Fenced code blocks are
left alone except for the private-use and memory-citation markers, which are
never legitimate text.
"""

from __future__ import annotations

import json
import re

_MEM_OPEN = "<oai-mem-citation>"
_MEM_CLOSE = "</oai-mem-citation>"
_REF_OPEN = chr(0xE200)
_REF_CLOSE = chr(0xE201)
_REF_SEP = chr(0xE202)
_REF_CHARS_RE = re.compile(f"[{chr(0xE200)}-{chr(0xE2FF)}]")
# A content reference is short and single-line; past this it is not one.
_REF_MAX = 512

_BARE_CITE_RE = re.compile(r"(?:file)?cite(?:turn\d+[a-z_]*\d+)+")
# Text that a later chunk could still extend into (or past) a bare citation.
_BARE_CITE_OPEN_RE = re.compile(
    r"(?:file)?cite(?:turn\d+[a-z_]*\d+)*(?:t(?:u(?:r(?:n(?:\d+[a-z_]*\d*)?)?)?)?)?"
    r"|f(?:i(?:l(?:e(?:c(?:i(?:t)?)?)?)?)?)?|c(?:i(?:t)?)?"
)
_LEGACY_CITE_MAX = 200
_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
_ASCII_WORD = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)
_NAME_BYTES = _ASCII_WORD | {"-"}

_OK, _PARTIAL, _BAD = "ok", "partial", "bad"


def _longest_prefix_suffix(text: str, marker: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``marker``."""
    for k in range(min(len(text), len(marker) - 1), 0, -1):
        if text.endswith(marker[:k]):
            return k
    return 0


def _render_reference(body: str) -> str:
    """Visible text of a content reference: an entity keeps its display name."""
    name, _, args = body.partition(_REF_SEP)
    if name != "entity":
        return ""
    try:
        value = json.loads(args.split(_REF_SEP)[0])
    except ValueError:
        return ""
    if isinstance(value, list) and len(value) > 1 and isinstance(value[1], str):
        return value[1]
    return ""


class _HiddenTags:
    """Stage 1: drop memory citations and private-use content references."""

    def __init__(self) -> None:
        self._mode = "text"
        self._pending = ""

    def push(self, chunk: str) -> str:
        self._pending += chunk
        out: list[str] = []
        while self._pending:
            buf = self._pending
            if self._mode == "text":
                mem = buf.find(_MEM_OPEN)
                ref = buf.find(_REF_OPEN)
                starts = [i for i in (mem, ref) if i >= 0]
                if not starts:
                    keep = _longest_prefix_suffix(buf, _MEM_OPEN)
                    out.append(buf[: len(buf) - keep])
                    self._pending = buf[len(buf) - keep :]
                    break
                start = min(starts)
                out.append(buf[:start])
                if start == mem:
                    self._mode = "mem"
                    self._pending = buf[start + len(_MEM_OPEN) :]
                else:
                    self._mode = "ref"
                    self._pending = buf[start:]
            elif self._mode == "mem":
                end = buf.find(_MEM_CLOSE)
                if end < 0:
                    keep = _longest_prefix_suffix(buf, _MEM_CLOSE)
                    self._pending = buf[len(buf) - keep :]
                    break
                self._mode = "text"
                self._pending = buf[end + len(_MEM_CLOSE) :]
            else:  # "ref": buffer from the opener until the closer.
                end = buf.find(_REF_CLOSE)
                newline = buf.find("\n")
                if end >= 0 and (newline < 0 or end < newline):
                    out.append(_render_reference(buf[1:end]))
                    self._mode = "text"
                    self._pending = buf[end + 1 :]
                    continue
                if newline >= 0 or len(buf) > _REF_MAX:
                    # Not a content reference after all: give the text back
                    # without its private-use characters.
                    cut = newline if newline >= 0 else len(buf)
                    out.append(_REF_CHARS_RE.sub("", buf[:cut]))
                    self._mode = "text"
                    self._pending = buf[cut:]
                    continue
                break
        return _REF_CHARS_RE.sub("", "".join(out))

    def finish(self) -> str:
        # An unterminated memory citation or content reference runs to the
        # end of the reply and is dropped, as Codex does.
        tail = self._pending if self._mode == "text" else ""
        self._mode, self._pending = "text", ""
        return _REF_CHARS_RE.sub("", tail)


def _parse_directive(
    text: str, start: int, backslash: bool
) -> tuple[str, int, str, dict[str, str]]:
    """Parse ``:name{k=v ...}`` at ``start``; returns (status, end, name, attrs)."""
    n = len(text)
    i = start
    while i < n and text[i] == ":":
        i += 1
    if i - start > 3:
        return _BAD, 0, "", {}
    if i == n:
        return _PARTIAL, 0, "", {}
    if not ("a" <= text[i].lower() <= "z"):
        return _BAD, 0, "", {}
    name_start = i
    while i < n and text[i] in _NAME_BYTES:
        i += 1
    if i == n:
        return _PARTIAL, 0, "", {}
    name = text[name_start:i]
    if text[i] != "{":
        return _BAD, 0, "", {}
    i += 1
    attrs: dict[str, str] = {}
    while True:
        while i < n and text[i] in " \t":
            i += 1
        if i == n:
            return _PARTIAL, 0, "", {}
        if text[i] == "}":
            return _OK, i + 1, name, attrs
        key_start = i
        while i < n and text[i] in _NAME_BYTES:
            i += 1
        if i == n:
            return _PARTIAL, 0, "", {}
        key = text[key_start:i]
        if not key or key in attrs:
            return _BAD, 0, "", {}
        while i < n and text[i] in " \t":
            i += 1
        if i == n:
            return _PARTIAL, 0, "", {}
        if text[i] != "=":
            return _BAD, 0, "", {}
        i += 1
        while i < n and text[i] in " \t":
            i += 1
        if i == n:
            return _PARTIAL, 0, "", {}
        if text[i] in "\"'":
            quote = text[i]
            i += 1
            value: list[str] = []
            while True:
                if i == n:
                    return _PARTIAL, 0, "", {}
                ch = text[i]
                if ch in "\r\n":
                    return _BAD, 0, "", {}
                if ch == quote:
                    i += 1
                    break
                if backslash and ch == "\\" and i + 1 < n and text[i + 1] == quote:
                    value.append(quote)
                    i += 2
                    continue
                if backslash and ch == "\\" and i + 1 == n:
                    return _PARTIAL, 0, "", {}
                value.append(ch)
                i += 1
            attrs[key] = "".join(value)
        else:
            value_start = i
            while i < n and text[i] not in " \t}\r\n":
                i += 1
            if i == n:
                return _PARTIAL, 0, "", {}
            if i == value_start:
                return _BAD, 0, "", {}
            attrs[key] = text[value_start:i]


def _match_directive(text: str, start: int) -> tuple[str, int, str, dict[str, str]]:
    # Directives quote literally, review comments with backslash escapes;
    # take whichever reading parses (as the Codex TUI does).
    statuses = []
    for backslash in (False, True):
        result = _parse_directive(text, start, backslash)
        if result[0] == _OK:
            return result
        statuses.append(result[0])
    if _PARTIAL in statuses:
        return _PARTIAL, 0, "", {}
    return _BAD, 0, "", {}


def _render_directive(name: str, attrs: dict[str, str]) -> str:
    if name != "code-comment":
        return ""
    title = attrs.get("title", "").strip()
    body = attrs.get("body", "").strip()
    file = attrs.get("file", "").strip()
    if not (title and body and file):
        return ""

    def as_int(key: str) -> int | None:
        try:
            return int(attrs.get(key, "").strip().lstrip("Pp"))
        except ValueError:
            return None

    line_start = max(as_int("start") or 1, 1)
    line_end = max(as_int("end") or line_start, line_start)
    priority = as_int("priority")
    if not re.match(r"\[[Pp]\d\]", title) and priority in (0, 1, 2, 3):
        title = f"[P{priority}] {title}"
    file = file.replace("\\", "/")
    location = (
        f"{file}:{line_start}"
        if line_start == line_end
        else f"{file}:{line_start}-{line_end}"
    )
    return f"- {title} — {location}\n  {body}"


class _LineMarkers:
    """Stage 2: directives and bare citations, which never span lines."""

    def __init__(self) -> None:
        self._raw = ""  # raw text of the current line seen so far
        self._buf = ""  # part of the current line not yet emitted
        self._prev = ""  # last character emitted on this line
        self._skip_ws = False
        self._in_fence = False

    def push(self, text: str) -> str:
        out: list[str] = []
        while text:
            newline = text.find("\n")
            if newline < 0:
                out.append(self._feed(text, final=False))
                break
            out.append(self._feed(text[:newline], final=True))
            out.append("\n")
            text = text[newline + 1 :]
        return "".join(out)

    def finish(self) -> str:
        return self._feed("", final=True)

    def _feed(self, piece: str, final: bool) -> str:
        self._raw += piece
        if self._in_fence:
            emitted = self._buf + piece
            self._buf = ""
        else:
            self._buf += piece
            emitted = self._scan(final)
        if final:
            if _FENCE_RE.match(self._raw):
                self._in_fence = not self._in_fence
            self._raw = self._buf = self._prev = ""
            self._skip_ws = False
        return emitted

    def _emit(self, out: list[str], text: str) -> None:
        if self._skip_ws:
            stripped = text.lstrip(" \t")
            if not stripped:
                return
            text = stripped
            self._skip_ws = False
        if text:
            out.append(text)
            self._prev = text[-1]

    def _drop(self, out: list[str], replacement: str = "") -> None:
        if replacement:
            self._emit(out, replacement)
        elif not self._prev or self._prev in " \t":
            self._skip_ws = True

    def _scan(self, final: bool) -> str:
        buf = self._buf
        out: list[str] = []
        i = j = 0
        n = len(buf)
        while j < n:
            ch = buf[j]
            before = buf[j - 1] if j else self._prev
            if ch == ":" and before != ":" and before not in _ASCII_WORD:
                status, end, name, attrs = _match_directive(buf, j)
                if status == _OK:
                    self._emit(out, buf[i:j])
                    self._drop(out, _render_directive(name, attrs))
                    i = j = end
                    continue
                if status == _PARTIAL and not final:
                    break
            elif ch == "【":
                end = buf.find("】", j + 1)
                if end < 0:
                    if not final and n - j <= _LEGACY_CITE_MAX:
                        break
                elif end - j <= _LEGACY_CITE_MAX and "†" in buf[j:end]:
                    self._emit(out, buf[i:j])
                    self._drop(out)
                    i = j = end + 1
                    continue
            elif ch in "cf":
                if not final and _BARE_CITE_OPEN_RE.fullmatch(buf, j):
                    break
                match = _BARE_CITE_RE.match(buf, j)
                if match:
                    self._emit(out, buf[i:j])
                    self._drop(out)
                    i = j = match.end()
                    continue
            j += 1
        self._emit(out, buf[i:j])
        self._buf = buf[j:]
        return "".join(out)


class ModelMarkupStream:
    """Incrementally strip hidden model markup from streamed text."""

    def __init__(self) -> None:
        self._tags = _HiddenTags()
        self._lines = _LineMarkers()

    def push(self, chunk: str) -> str:
        return self._lines.push(self._tags.push(chunk))

    def finish(self) -> str:
        return self._lines.push(self._tags.finish()) + self._lines.finish()


def strip_model_markup(text: str) -> str:
    """Strip hidden model markup from a complete reply."""
    if not text:
        return text
    stream = ModelMarkupStream()
    cleaned = stream.push(text) + stream.finish()
    if cleaned == text:
        return text
    # Removed markers leave trailing blanks behind; the reply's own leading
    # indentation is kept.
    lines = cleaned.split("\n")
    original = set(text.split("\n"))
    lines = [line if line in original else line.rstrip() for line in lines]
    return "\n".join(lines).rstrip()


__all__ = ["ModelMarkupStream", "strip_model_markup"]
