"""Sliding-window rate limits from permission rules.

A rule (permission group) may cap how many requests an account sends to the
agent, e.g. 10 per minute and 100 per day; every window must hold. Each
request a user addresses to the bot (an @, a wake word, a private message)
counts once -- including one steered into a running turn or queued behind
another -- while scheduled tasks, background task wake-ups and the bot's own
active replies do not. A request stopped before the agent (a plugin hook, a
failure while preparing it) is given back. Accounts are ``<platform id>:<sender id>``, so one person
shares a single count across every group and private chat on that platform;
the limits applied are those of the group matched where the message was sent.
A group without limits is not counted at all.

Uses are kept in SQLite (``permission_usage``), so a restart does not reset a
daily quota. A refused request is not counted, and the account is told once
per refusal period rather than on every message.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sqlalchemy import text

from astrbot.core import logger
from astrbot.core.permission_rules import (
    CONFIG_KEY,
    MAX_WINDOW_S,
    PermissionPolicy,
    policy_for_event,
)

DEFAULT_REPLY = "请求太频繁了，请 {wait} 后再试（限制：{limit}）。"
# How often uses older than the longest allowed window are deleted.
PRUNE_INTERVAL_S = 600.0
# Event extra holding (limiter, use id) of the use this request counted as.
USE_EXTRA_KEY = "_rate_limit_use"
# Event extra marking a request the bot started itself (an active reply).
ACTIVE_REPLY_EXTRA = "_active_reply"
# Accounts remembered as already told about a refusal before a cleanup.
MAX_NOTIFIED = 4096


def format_duration(seconds: float) -> str:
    seconds = max(1, int(seconds + 0.999))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if days:
        parts.append(f"{days} 天")
    if hours:
        parts.append(f"{hours} 小时")
    if minutes and not days:
        parts.append(f"{minutes} 分钟")
    if secs and not (days or hours):
        parts.append(f"{secs} 秒")
    return " ".join(parts)


def format_limit(window: int, count: int) -> str:
    return f"每 {format_duration(window)} {count} 次"


class RateLimiter:
    """Counts accounts' uses in the database and refuses those over a limit."""

    def __init__(self, db: Any) -> None:
        self.db = db
        # Check and record as one step, so concurrent messages cannot both
        # take the last free slot.
        self._lock = asyncio.Lock()
        # (account, limit) -> time until which that refusal was announced.
        self._notified: dict[tuple[str, tuple[int, int]], float] = {}
        self._last_prune = 0.0

    async def acquire(
        self,
        account: str,
        limits: tuple[tuple[int, int], ...],
        *,
        now: float | None = None,
    ) -> tuple[int | None, tuple[float, tuple[int, int]] | None]:
        """Records one use, or refuses it.

        Args:
            account: Whose use this is.
            limits: (window seconds, max uses) pairs; all must hold.
            now: Current time (epoch seconds); for tests.

        Returns:
            (use id, None) when the use was recorded, else (None, refusal):
            the seconds until it would be allowed and the limit that is the
            last to clear.
        """
        if not limits:
            return None, None
        now = time.time() if now is None else now
        async with self._lock, self.db.get_db() as session:
            async with session.begin():
                counts = await self._counts(session, account, limits, now)
                waits = []
                for (window, count), used in zip(limits, counts):
                    if used < count:
                        continue
                    # The window frees a slot once enough of its oldest uses
                    # have aged out.
                    row = await session.execute(
                        text(
                            "SELECT used_at FROM permission_usage "
                            "WHERE account = :account AND used_at > :since "
                            "ORDER BY used_at ASC LIMIT 1 OFFSET :offset"
                        ),
                        {
                            "account": account,
                            "since": now - window,
                            "offset": used - count,
                        },
                    )
                    oldest = row.scalar()
                    wait = (oldest + window - now) if oldest is not None else 1.0
                    waits.append((max(wait, 1.0), (window, count)))
                if waits:
                    return None, max(waits)
                inserted = await session.execute(
                    text(
                        "INSERT INTO permission_usage (account, used_at) "
                        "VALUES (:account, :now)"
                    ),
                    {"account": account, "now": now},
                )
                # Keep whatever any rule may count, in every config file.
                if now - self._last_prune >= PRUNE_INTERVAL_S:
                    self._last_prune = now
                    await session.execute(
                        text("DELETE FROM permission_usage WHERE used_at <= :before"),
                        {"before": now - MAX_WINDOW_S},
                    )
        return inserted.lastrowid, None

    async def refund(self, use_id: int) -> None:
        """Gives back a use whose request never reached the agent."""
        async with self._lock, self.db.get_db() as session:
            async with session.begin():
                await session.execute(
                    text("DELETE FROM permission_usage WHERE id = :id"), {"id": use_id}
                )

    async def _counts(
        self,
        session: Any,
        account: str,
        limits: tuple[tuple[int, int], ...],
        now: float,
    ) -> list[int]:
        cases = ", ".join(
            f"COALESCE(SUM(CASE WHEN used_at > :since{i} THEN 1 ELSE 0 END), 0)"
            for i in range(len(limits))
        )
        params: dict[str, Any] = {
            f"since{i}": now - window for i, (window, _) in enumerate(limits)
        }
        params["account"] = account
        params["oldest"] = now - max(window for window, _ in limits)
        row = await session.execute(
            text(
                f"SELECT {cases} FROM permission_usage "
                "WHERE account = :account AND used_at > :oldest"
            ),
            params,
        )
        return [int(v or 0) for v in row.one()]

    def should_notify(
        self,
        account: str,
        limit: tuple[int, int],
        wait: float,
        now: float | None = None,
    ) -> bool:
        """Whether to tell the account about this refusal: once per period of
        each limit, so a longer limit met in another chat is still told."""
        now = time.time() if now is None else now
        key = (account, limit)
        if self._notified.get(key, 0.0) > now:
            return False
        if len(self._notified) >= MAX_NOTIFIED:
            self._notified = {k: t for k, t in self._notified.items() if t > now}
        self._notified[key] = now + wait
        return True


_limiter: RateLimiter | None = None


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        from astrbot.core import db_helper

        _limiter = RateLimiter(db_helper)
    return _limiter


def account_of(event: Any) -> str:
    sender = str(event.get_sender_id() or "")
    if not sender:
        return ""
    return f"{event.get_platform_id() or ''}:{sender}"


def is_counted(event: Any) -> bool:
    """Whether a user asked for this request.

    Scheduled and background-task wake-ups are synthetic, and an active reply
    is the bot joining a chat nobody asked it into. Everything else counts,
    including a plugin that answers a keyword without an @.
    """
    from astrbot.core.cron.events import CronMessageEvent

    if isinstance(event, CronMessageEvent):
        return False
    return event.get_extra(ACTIVE_REPLY_EXTRA) is not True


async def check_rate_limit(
    event: Any, config: dict, limiter: RateLimiter | None = None
) -> str | None:
    """Counts this request against the sender's group, or refuses it.

    Returns:
        None to let the request through (it was counted if limited at all,
        and ``refund_rate_limit`` gives the use back); otherwise the reply to
        send, which is empty when the account was already told during this
        refusal period.
    """
    if not is_counted(event):
        return None
    rules = config.get(CONFIG_KEY) or []
    policy: PermissionPolicy = policy_for_event(event, rules)
    account = account_of(event)
    if not policy.rate_limits or not account:
        return None
    limiter = limiter or get_limiter()
    try:
        use_id, refused = await limiter.acquire(account, policy.rate_limits)
    except Exception as e:  # noqa: BLE001 - never block chat on a counter failure
        logger.warning("Permission rate limit check failed for %s: %s", account, e)
        return None
    if refused is None:
        if use_id is not None:
            event.set_extra(USE_EXTRA_KEY, (limiter, use_id))
        return None
    wait, (window, count) = refused
    logger.info(
        "Rate limited %s (rule %s): %d per %ds, retry in %.0fs",
        account,
        policy.rule_name or "-",
        count,
        window,
        wait,
    )
    if not limiter.should_notify(account, (window, count), wait):
        return ""
    template = policy.rate_limit_reply or DEFAULT_REPLY
    try:
        return template.format(
            wait=format_duration(wait), limit=format_limit(window, count)
        )
    except Exception:  # noqa: BLE001 - a malformed template is sent as is
        return template


def keep_rate_limit_use(event: Any) -> None:
    """The request reached the agent: its use stays counted, whatever
    happens to the run afterwards."""
    set_extra = getattr(event, "set_extra", None)
    if callable(set_extra):
        set_extra(USE_EXTRA_KEY, None)


async def refund_rate_limit(event: Any) -> None:
    """Gives back the use this request counted as, since it never reached the
    agent. Safe to call again, or for a request that was never counted."""
    counted = event.get_extra(USE_EXTRA_KEY)
    if not isinstance(counted, tuple):
        return
    event.set_extra(USE_EXTRA_KEY, None)
    limiter, use_id = counted
    try:
        await limiter.refund(use_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not give back a rate-limited use: %s", e)
