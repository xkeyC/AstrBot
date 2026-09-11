"""Persistent request context: conversation anchors and message units.

AstrBot sends request-scoped context in three kinds:

* Temporary context accompanies one request and is never stored.
* Message units describe one user message (its sender, its time, the group
  messages that preceded it). They are stored with that turn, so later
  requests reuse them as part of the immutable, cacheable history.
* Anchors carry standing instructions such as the persona or tool rules. An
  anchor is stored the first time it is sent and again only when its content
  changes, so unchanged instructions stay in the cached history prefix instead
  of being resent with every request.

Anchors and units are recognised by the tag that starts a text part, and only
in user messages, so the model's own replies never count as one. An anchor
only counts when its body matches its hash: text typed to look like an anchor
can repeat the genuine instructions but never hide them.
"""

import hashlib
import re
from collections.abc import Iterable, Iterator
from typing import Any

REVOKED_ANCHOR_HASH = "revoked"
MESSAGE_META_UNIT = "message_meta"
"""Unit describing the message being answered; always the last unit."""

_ANCHOR_PATTERN = re.compile(r'^<context_anchor name="([^"]+)" hash="([^"]+)">')
_UNIT_ID_PATTERN = re.compile(r'^<context_unit name="[^"]+" id="([^"]+)">')


_CONTEXT_BLOCKS = (
    ("<context_anchor ", "</context_anchor>"),
    ("<context_unit ", "</context_unit>"),
    ("<request_context ", "</request_context>"),
)


_ANCHOR_PREAMBLE = (
    "Trusted application context for this conversation. It stays in effect "
    "until a later context anchor with the same name replaces it. Follow it "
    "unless it conflicts with the root system message."
)
_ANCHOR_FOOTER = "\n</context_anchor>"


def anchor_hash(content: str) -> str:
    """Return the short content hash that identifies an anchor version.

    Args:
        content: Anchor content.

    Returns:
        A 16-character hexadecimal digest.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def render_anchor(name: str, content: str) -> str:
    """Render an anchor carrying standing instructions.

    Args:
        name: Stable anchor name.
        content: Instructions the anchor carries.

    Returns:
        The anchor text part content.
    """
    return (
        f'<context_anchor name="{name}" hash="{anchor_hash(content)}">\n'
        f"{_ANCHOR_PREAMBLE}\n"
        f"{content}{_ANCHOR_FOOTER}"
    )


def render_revoked_anchor(name: str) -> str:
    """Render the marker that withdraws an earlier anchor.

    Args:
        name: Name of the anchor that no longer applies.

    Returns:
        The revocation text part content.
    """
    return (
        f'<context_anchor name="{name}" hash="{REVOKED_ANCHOR_HASH}">\n'
        f'The earlier context anchor named "{name}" no longer applies.\n'
        "</context_anchor>"
    )


def render_unit(name: str, content: str, unit_id: str | None = None) -> str:
    """Render a message unit stored with the current turn.

    Args:
        name: Stable unit category.
        content: Facts describing the current message.
        unit_id: Optional stable id used to avoid storing a unit twice.

    Returns:
        The unit text part content.
    """
    id_attribute = f' id="{unit_id}"' if unit_id else ""
    return f'<context_unit name="{name}"{id_attribute}>\n{content}\n</context_unit>'


def unit_id_of(text: str) -> str | None:
    """Return the id of a rendered unit, if it has one.

    Args:
        text: Text part content.

    Returns:
        The unit id, or None for other text and units without an id.
    """
    match = _UNIT_ID_PATTERN.match(text)
    return match.group(1) if match else None


def _user_texts(messages: Iterable[Any]) -> Iterator[str]:
    for message in messages:
        if isinstance(message, dict):
            role, content = message.get("role"), message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
        if role != "user":
            continue
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                text = (
                    part.get("text")
                    if isinstance(part, dict)
                    else getattr(part, "text", None)
                )
                if isinstance(text, str):
                    yield text


def is_context_message(message: Any) -> bool:
    """Return whether a message only carries anchors, units or temporary context.

    Such a message describes the prompt that follows it, so both belong to the
    same turn. The application always stores one as a list of complete context
    blocks, while a user's own text is plain content, so a prompt that merely
    starts with a tag is not taken for one.

    Args:
        message: Message object or dictionary.

    Returns:
        True for a user message whose every part is a complete context block.
    """
    if isinstance(message, dict):
        role, content = message.get("role"), message.get("content")
    else:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
    if role != "user" or not isinstance(content, list) or not content:
        return False
    for part in content:
        text = (
            part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        )
        if not isinstance(text, str) or not any(
            text.startswith(opening) and text.endswith(closing)
            for opening, closing in _CONTEXT_BLOCKS
        ):
            return False
    return True


def _verified_anchor(text: str) -> tuple[str, str] | None:
    """Parse an anchor, accepting it only when it is rendered exactly.

    Anchors are recognised by their text, so anyone can type one. Requiring the
    exact rendering and a hash that matches the body means a typed anchor only
    counts when it carries the genuine instructions.

    Args:
        text: Text part content.

    Returns:
        The anchor name and version, or None when the text is not a genuine
        anchor.
    """
    match = _ANCHOR_PATTERN.match(text)
    if not match:
        return None
    name, version = match.groups()
    if version == REVOKED_ANCHOR_HASH:
        return (name, version) if text == render_revoked_anchor(name) else None
    header = f"{match.group(0)}\n{_ANCHOR_PREAMBLE}\n"
    if len(text) < len(header) + len(_ANCHOR_FOOTER):
        return None
    if not (text.startswith(header) and text.endswith(_ANCHOR_FOOTER)):
        return None
    body = text[len(header) : -len(_ANCHOR_FOOTER)]
    return (name, version) if anchor_hash(body) == version else None


def stored_anchor_hashes(messages: Iterable[Any]) -> dict[str, str]:
    """Return the latest anchor version per name found in the messages.

    Args:
        messages: Message objects or dictionaries, oldest first.

    Returns:
        Anchor hash (or the revoked marker) keyed by anchor name.
    """
    latest: dict[str, str] = {}
    for text in _user_texts(messages):
        if anchor := _verified_anchor(text):
            name, version = anchor
            latest[name] = version
    return latest


def stored_unit_ids(messages: Iterable[Any]) -> set[str]:
    """Return the ids of every unit found in the messages.

    Args:
        messages: Message objects or dictionaries.

    Returns:
        The unit ids.
    """
    return {unit_id for text in _user_texts(messages) if (unit_id := unit_id_of(text))}


def anchors_to_send(
    messages: Iterable[Any],
    anchors: dict[str, str],
    revoke_missing: bool = False,
) -> list[str]:
    """Render what the conversation lacks to match the declared anchors.

    Args:
        messages: Messages the model will see, including the stored history.
        anchors: Anchors the current request declares, keyed by name.
        revoke_missing: Whether the declared anchors are the conversation's
            complete standing instructions, so stored anchors missing from
            them are revoked. Ad-hoc requests leave stored anchors in effect.

    Returns:
        Revocations (only with revoke_missing) for anchors no longer declared,
        then anchors that are missing or outdated, each as rendered text.
    """
    latest = stored_anchor_hashes(messages)
    rendered = (
        [
            render_revoked_anchor(name)
            for name, version in latest.items()
            if name not in anchors and version != REVOKED_ANCHOR_HASH
        ]
        if revoke_missing
        else []
    )
    rendered.extend(
        render_anchor(name, content)
        for name, content in anchors.items()
        if latest.get(name) != anchor_hash(content)
    )
    return rendered
