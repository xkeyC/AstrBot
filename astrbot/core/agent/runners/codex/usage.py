"""Per-turn usage of a Codex thread, in AstrBot's stats shapes.

Codex reports usage on ``token_count`` events, but re-sends the same numbers
on rate-limit and context updates, so adding those up double counts. The
thread's cumulative ``total_token_usage`` only grows when a model response
reports usage: its difference across a turn is exactly that turn's usage.
"""

from __future__ import annotations

import json
import typing as T

from astrbot.core import logger
from astrbot.core.provider.entities import TokenUsage

if T.TYPE_CHECKING:
    from .native import CodexEngine

JsonObject = dict[str, T.Any]

# Events that mean the model has started answering, for the time-to-first-token
# fallback when Codex does not report its own measurement.
FIRST_TOKEN_EVENTS = frozenset(
    {
        "agent_message_content_delta",
        "agent_message_delta",
        "agent_message",
        "agent_reasoning_delta",
        "agent_reasoning",
        "agent_reasoning_raw_content_delta",
        "reasoning_content_delta",
        "reasoning_raw_content_delta",
    }
)


async def thread_usage(engine: CodexEngine, thread_id: str) -> JsonObject | None:
    """`{"model", "model_provider", "total_token_usage"}`, or None if unknown.

    None also with a binding that predates ``thread_usage``.
    """
    get = getattr(engine.rt, "thread_usage", None)
    if get is None:
        return None
    try:
        info = json.loads(await get(thread_id))
    except Exception as e:  # noqa: BLE001 - stats must never break a turn
        logger.debug("Codex thread usage unavailable for %s: %s", thread_id, e)
        return None
    return info if isinstance(info, dict) else None


def _int(usage: JsonObject, key: str) -> int:
    try:
        return int(usage.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def to_token_usage(usage: JsonObject | None) -> TokenUsage | None:
    """Codex ``TokenUsage`` -> AstrBot's; Codex's input count includes cached."""
    if not isinstance(usage, dict):
        return None
    cached = _int(usage, "cached_input_tokens")
    return TokenUsage(
        input_other=max(_int(usage, "input_tokens") - cached, 0),
        input_cached=cached,
        output=_int(usage, "output_tokens"),
    )


def usage_between(
    before: JsonObject | None, after: JsonObject | None
) -> TokenUsage | None:
    """Usage spent between two ``thread_usage`` snapshots of one thread."""
    end = to_token_usage((after or {}).get("total_token_usage"))
    if end is None:
        return None
    start = to_token_usage((before or {}).get("total_token_usage")) or TokenUsage()
    return TokenUsage(
        input_other=max(end.input_other - start.input_other, 0),
        input_cached=max(end.input_cached - start.input_cached, 0),
        output=max(end.output - start.output, 0),
    )
