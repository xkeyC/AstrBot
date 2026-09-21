"""Per-turn usage of a Codex thread, in AstrBot's stats shapes.

Codex reports usage on ``token_count`` events, but re-sends the same numbers
on rate-limit and context updates, so adding those up double counts. The
thread's cumulative ``total_token_usage`` grows only when a model response
reports usage, so its growth across a turn is that turn's usage -- with one
exception: when the context window overflows, Codex replaces the total with a
placeholder (every count zero). ``UsageMeter`` follows the total through the
turn's events so usage spent before such a reset is kept.
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


def _grew(prev: TokenUsage, cur: TokenUsage) -> bool:
    return (
        cur.input_other >= prev.input_other
        and cur.input_cached >= prev.input_cached
        and cur.output >= prev.output
    )


class UsageMeter:
    """Adds up a thread's usage across one turn from its cumulative totals.

    Fed the ``thread_usage`` snapshot taken before the turn, then every total
    seen during it (``token_count`` events and the closing snapshot). Growth
    is added; a total that went down was reset, and becomes the new baseline.
    """

    def __init__(self) -> None:
        self._last: TokenUsage | None = None
        self._spent = TokenUsage()
        self.started = False

    def start(self, snapshot: JsonObject | None) -> None:
        """Baseline from the pre-turn snapshot; None (old binding) disables."""
        if snapshot is None:
            return
        self.started = True
        self._last = to_token_usage(snapshot.get("total_token_usage")) or TokenUsage()

    def observe(self, total: JsonObject | None) -> None:
        cur = to_token_usage(total)
        if not self.started or cur is None or self._last is None:
            return
        if _grew(self._last, cur):
            self._spent = self._spent + (cur - self._last)
        self._last = cur

    @property
    def spent(self) -> TokenUsage | None:
        """The turn's usage, or None when there was no baseline."""
        return self._spent if self.started else None


def usage_between(
    before: JsonObject | None, after: JsonObject | None
) -> TokenUsage | None:
    """Usage spent between two ``thread_usage`` snapshots of one thread."""
    meter = UsageMeter()
    meter.start(before if before is not None else {})
    meter.observe((after or {}).get("total_token_usage"))
    if (after or {}).get("total_token_usage") is None:
        return None
    return meter.spent
