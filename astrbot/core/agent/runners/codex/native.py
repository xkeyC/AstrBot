"""In-process Codex engine (``codex_astrbot`` pyo3 binding).

One engine per distinct option set. Each loaded Codex thread gets a pump task
that reads its events and routes them to the turn currently running on it;
dynamic tool calls are answered from their own tasks so tools run
concurrently with event streaming.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot.core import logger

JsonObject = dict[str, Any]
ToolCallHandler = Callable[[JsonObject], Awaitable[JsonObject]]
# (kind "exec" | "patch", request) -> (approved, reason)
ApprovalHandler = Callable[[str, JsonObject], Awaitable[tuple[bool, str]]]

APPROVAL_EVENTS = {
    "exec_approval_request": "exec",
    "apply_patch_approval_request": "patch",
}

TERMINAL_EVENTS = ("task_complete", "turn_complete", "turn_aborted")


class CodexEngineError(RuntimeError):
    pass


def _import_binding():
    try:
        import codex_astrbot  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise CodexEngineError(
            "codex_astrbot is not installed. Build it with `maturin develop` in "
            "codex-rs/astrbot-py of the codex_for_astrbot fork."
        ) from e
    return codex_astrbot


def find_code_mode_host(explicit: str = "") -> str | None:
    """Locate codex-code-mode-host: explicit path, next to `codex` on PATH, or none."""
    if explicit:
        return explicit if Path(explicit).is_file() else None
    exe = shutil.which("codex")
    if not exe:
        return None
    exe_path = Path(exe).resolve()
    names = ("codex-code-mode-host.exe", "codex-code-mode-host")
    candidates = [exe_path.parent / n for n in names]
    # scoop shims live in ~/scoop/shims; the real install is apps/codex/current/bin
    candidates += [
        exe_path.parent.parent / "apps" / "codex" / "current" / "bin" / n for n in names
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


@dataclass
class ActiveTurn:
    """A turn running for a session (UMO), used to steer same-sender follow-ups."""

    engine: CodexEngine
    thread_id: str
    turn_id: str
    sender_id: str
    message_id: str | None = None
    steered: int = 0
    aborted: bool = False
    steered_texts: list[str] = field(default_factory=list)


# umo -> running turn
ACTIVE_TURNS: dict[str, ActiveTurn] = {}


async def try_steer(
    umo: str, sender_id: str, turn_input: list[JsonObject], *, prompt: str = ""
) -> str | None:
    """Inject a follow-up from the same sender into the running turn.

    Returns the running turn's source message id (or "") when Codex accepted
    the input; the caller then produces no reply of its own. Other senders,
    stopped turns, or a turn that already finished return None and are handled
    as a new turn (queued).
    """
    active = ACTIVE_TURNS.get(umo)
    if (
        active is None
        or active.aborted
        or not sender_id
        or active.sender_id != sender_id
    ):
        return None
    try:
        result = await active.engine.submit_turn(
            active.thread_id,
            {"input": turn_input, "mode": "steer", "expected_turn_id": active.turn_id},
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("codex steer failed for %s: %s", umo, e)
        return None
    if result.get("status") != "steered":
        return None
    active.steered += 1
    if prompt:
        active.steered_texts.append(prompt)
    logger.info(
        "Follow-up from %s steered into running Codex turn (umo=%s)", sender_id, umo
    )
    return active.message_id or ""


@dataclass
class _TurnRoute:
    events: asyncio.Queue[JsonObject]
    tool_handler: ToolCallHandler | None
    approval_handler: ApprovalHandler | None = None


@dataclass
class ThreadPump:
    engine: CodexEngine
    thread_id: str
    route: _TurnRoute | None = None
    task: asyncio.Task | None = None
    tool_tasks: set[asyncio.Task] = field(default_factory=set)
    closed: bool = False

    def start(self) -> None:
        self.task = asyncio.create_task(
            self._run(), name=f"codex-pump-{self.thread_id}"
        )

    def open_turn(
        self,
        tool_handler: ToolCallHandler | None,
        approval_handler: ApprovalHandler | None = None,
    ) -> asyncio.Queue[JsonObject]:
        queue: asyncio.Queue[JsonObject] = asyncio.Queue()
        self.route = _TurnRoute(queue, tool_handler, approval_handler)
        return queue

    def close_turn(self) -> None:
        self.route = None

    async def _run(self) -> None:
        rt = self.engine.rt
        try:
            while True:
                raw = await rt.next_event(self.thread_id)
                if raw is None:  # thread terminated and drained
                    if self.route is not None:
                        self.route.events.put_nowait(
                            {"type": "_pump_closed", "message": "thread terminated"}
                        )
                    break
                event = json.loads(raw)
                msg = event.get("msg") or {}
                if msg.get("type") == "dynamic_tool_call_request":
                    task = asyncio.create_task(self._answer_tool(msg))
                    self.tool_tasks.add(task)
                    task.add_done_callback(self.tool_tasks.discard)
                    continue
                if (kind := APPROVAL_EVENTS.get(msg.get("type") or "")) is not None:
                    task = asyncio.create_task(self._answer_approval(kind, msg))
                    self.tool_tasks.add(task)
                    task.add_done_callback(self.tool_tasks.discard)
                if self.route is not None:
                    self.route.events.put_nowait(msg)
                if msg.get("type") == "shutdown_complete":
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - thread closed or engine failure
            logger.debug("codex pump %s stopped: %s", self.thread_id, e)
            if self.route is not None:
                self.route.events.put_nowait(
                    {"type": "_pump_closed", "message": str(e)}
                )
        finally:
            self.closed = True
            self.engine.pumps.pop(self.thread_id, None)

    async def _answer_approval(self, kind: str, msg: JsonObject) -> None:
        """Decide a native exec / patch approval; unattended requests are denied."""
        route = self.route
        approved, reason = False, "No AstrBot session is attached to this request."
        try:
            if route is not None and route.approval_handler is not None:
                approved, reason = await route.approval_handler(kind, msg)
        except Exception as e:  # noqa: BLE001
            logger.error("codex approval handler failed: %s", e, exc_info=True)
            approved, reason = False, f"approval failed: {e!s}"
        call_id = str(msg.get("call_id") or msg.get("callId") or "")
        request = {
            "kind": kind,
            "id": str(msg.get("approval_id") or msg.get("approvalId") or call_id),
            "turn_id": msg.get("turn_id") or msg.get("turnId") or None,
            "approved": approved,
            "reason": reason or None,
        }
        try:
            await self.engine.rt.review_decision(
                self.thread_id, json.dumps(request, ensure_ascii=False)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("codex approval response for %s failed: %s", call_id, e)

    async def _answer_tool(self, msg: JsonObject) -> None:
        call_id = msg.get("callId") or msg.get("call_id") or ""
        route = self.route
        try:
            if route is None or route.tool_handler is None:
                result = {
                    "contentItems": [
                        {
                            "type": "inputText",
                            "text": "No AstrBot session is attached to this call.",
                        }
                    ],
                    "success": False,
                }
            else:
                result = await route.tool_handler(msg)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "codex tool call %s failed: %s", msg.get("tool"), e, exc_info=True
            )
            result = {
                "contentItems": [{"type": "inputText", "text": f"error: {e!s}"}],
                "success": False,
            }
        try:
            await self.engine.rt.dynamic_tool_response(
                self.thread_id, call_id, json.dumps(result, ensure_ascii=False)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("codex tool response for %s failed: %s", call_id, e)


class CodexEngine:
    """Shared in-process Codex runtime for one option set."""

    _instances: dict[str, CodexEngine] = {}
    _lock: asyncio.Lock | None = None

    def __init__(self, rt: Any) -> None:
        self.rt = rt
        self.pumps: dict[str, ThreadPump] = {}
        self.thread_locks: dict[str, asyncio.Lock] = {}

    @classmethod
    async def get(cls, options: JsonObject) -> CodexEngine:
        key = json.dumps(options, sort_keys=True, ensure_ascii=False)
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        async with cls._lock:
            engine = cls._instances.get(key)
            if engine is None:
                binding = _import_binding()
                Path(options["codex_home"]).mkdir(parents=True, exist_ok=True)
                rt = await binding.Runtime.create(
                    json.dumps(options, ensure_ascii=False)
                )
                engine = cls(rt)
                cls._instances[key] = engine
                logger.info("Codex engine ready (codex_home=%s)", options["codex_home"])
            return engine

    @classmethod
    async def shutdown_all(cls) -> None:
        engines = list(cls._instances.values())
        cls._instances.clear()
        for engine in engines:
            with contextlib.suppress(Exception):
                await engine.rt.shutdown()

    def lock_for(self, thread_id: str) -> asyncio.Lock:
        return self.thread_locks.setdefault(thread_id, asyncio.Lock())

    def _pump(self, thread_id: str) -> ThreadPump:
        pump = self.pumps.get(thread_id)
        if pump is None or pump.closed:
            pump = ThreadPump(self, thread_id)
            self.pumps[thread_id] = pump
            pump.start()
        return pump

    async def open_thread(
        self, state: JsonObject | None, params: JsonObject
    ) -> tuple[JsonObject, bool]:
        """Reuse, resume or start a thread. Returns (info, started_new)."""
        thread_id = (state or {}).get("thread_id")
        if thread_id and await self.rt.is_loaded(thread_id):
            self._pump(thread_id)
            return {
                "thread_id": thread_id,
                "rollout_path": state.get("rollout_path"),
            }, False
        rollout = (state or {}).get("rollout_path")
        if thread_id and rollout and Path(rollout).exists():
            try:
                info = json.loads(
                    await self.rt.resume_thread(
                        json.dumps(
                            {**params, "rollout_path": rollout}, ensure_ascii=False
                        )
                    )
                )
                self._pump(info["thread_id"])
                logger.info("Codex thread resumed: %s", info["thread_id"])
                return info, False
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Resume codex thread %s failed, starting new: %s", thread_id, e
                )
        info = json.loads(
            await self.rt.start_thread(json.dumps(params, ensure_ascii=False))
        )
        self._pump(info["thread_id"])
        logger.info(
            "Codex thread started: %s model=%s", info["thread_id"], info.get("model")
        )
        return info, True

    def pump(self, thread_id: str) -> ThreadPump:
        return self._pump(thread_id)

    async def submit_turn(self, thread_id: str, request: JsonObject) -> JsonObject:
        return json.loads(
            await self.rt.submit_turn(
                thread_id, json.dumps(request, ensure_ascii=False)
            )
        )

    async def interrupt(self, thread_id: str) -> None:
        with contextlib.suppress(Exception):
            await self.rt.interrupt(thread_id)

    async def forget_thread(self, thread_id: str) -> None:
        with contextlib.suppress(Exception):
            await self.rt.shutdown_thread(thread_id)
