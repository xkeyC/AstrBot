"""Asyncio JSON-RPC client for ``codex app-server`` over stdio.

The official Python SDK answers server-initiated requests on its single stdout
reader thread, so a slow tool call would stall every other thread on the same
process. This client routes notifications and server requests per Codex thread
and answers server requests from their own tasks, so AstrBot tools can run
concurrently while other sessions keep streaming.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from astrbot.core import logger

JsonObject = dict[str, Any]
ServerRequestHandler = Callable[[str, JsonObject], Awaitable[JsonObject]]

CLIENT_NAME = "astrbot"
CLIENT_VERSION = "0.1.0"
_STREAM_LIMIT = 64 * 1024 * 1024


class CodexAppServerError(RuntimeError):
    """A JSON-RPC error returned by the app-server, or a dead connection."""

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class CodexLaunchOptions:
    codex_bin: str = ""
    codex_home: str = ""
    config_overrides: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    def key(self) -> tuple:
        return (
            self.codex_bin,
            self.codex_home,
            self.config_overrides,
            tuple(sorted(self.env.items())),
        )


@dataclass
class _ThreadRoute:
    """Where messages for one Codex thread go while a turn is running."""

    events: asyncio.Queue[JsonObject]
    request_handler: ServerRequestHandler | None = None


class CodexAppServerClient:
    """One long-lived ``codex app-server`` process shared by all sessions."""

    def __init__(self, options: CodexLaunchOptions) -> None:
        self.options = options
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[JsonObject]] = {}
        self._routes: dict[str, _ThreadRoute] = {}
        self._loaded_threads: set[str] = set()
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self.server_info: JsonObject = {}

    # ------------------------------------------------------------------ process

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def _resolve_bin(self) -> str:
        if self.options.codex_bin:
            return self.options.codex_bin
        found = shutil.which("codex")
        if not found:
            raise CodexAppServerError(
                "Cannot find the codex executable. Install Codex CLI or set codex_bin."
            )
        return found

    async def ensure_started(self) -> None:
        if self.alive:
            return
        async with self._start_lock:
            if self.alive:
                return
            await self._start()

    async def _start(self) -> None:
        args: list[str] = []
        for kv in self.options.config_overrides:
            args.extend(["--config", kv])
        args.extend(["app-server", "--listen", "stdio://"])
        env = os.environ.copy()
        env.update(self.options.env)
        if self.options.codex_home:
            env["CODEX_HOME"] = self.options.codex_home
        codex_bin = self._resolve_bin()
        logger.info("Starting codex app-server: %s %s", codex_bin, " ".join(args))
        self._proc = await asyncio.create_subprocess_exec(
            codex_bin,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=_STREAM_LIMIT,
        )
        self._loaded_threads.clear()
        self._reader_task = asyncio.create_task(self._read_loop(self._proc))
        self._stderr_task = asyncio.create_task(self._drain_stderr(self._proc))
        self.server_info = await self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": CLIENT_NAME,
                    "title": "AstrBot",
                    "version": CLIENT_VERSION,
                },
                # Dynamic tools and additionalContext are experimental fields.
                "capabilities": {"experimentalApi": True},
            },
        )
        await self.notify("initialized")
        logger.info("codex app-server ready: %s", self.server_info.get("userAgent"))

    async def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc and proc.returncode is None:
            with contextlib.suppress(Exception):
                if proc.stdin:
                    proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except asyncio.TimeoutError:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
        for task in (self._reader_task, self._stderr_task, *self._tasks):
            if task and not task.done():
                task.cancel()
        self._fail_all(CodexAppServerError("codex app-server closed"))

    def _fail_all(self, exc: Exception) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()
        for route in self._routes.values():
            route.events.put_nowait(
                {"method": "_connection/closed", "params": {"message": str(exc)}}
            )
        self._loaded_threads.clear()

    # ---------------------------------------------------------------- transport

    async def _write(self, obj: JsonObject) -> None:
        proc = self._proc
        if not proc or proc.returncode is not None or not proc.stdin:
            raise CodexAppServerError("codex app-server is not running")
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            proc.stdin.write(data)
            await proc.stdin.drain()

    async def request(
        self, method: str, params: JsonObject | None = None
    ) -> JsonObject:
        self._next_id += 1
        req_id = self._next_id
        fut: asyncio.Future[JsonObject] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._write({"id": req_id, "method": method, "params": params or {}})
            return await fut
        finally:
            self._pending.pop(req_id, None)

    async def notify(self, method: str, params: JsonObject | None = None) -> None:
        msg: JsonObject = {"method": method}
        if params is not None:
            msg["params"] = params
        await self._write(msg)

    async def _read_loop(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout
        try:
            while line := await proc.stdout.readline():
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "codex app-server sent non-JSON line: %r", line[:200]
                    )
                    continue
                self._dispatch(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("codex app-server reader failed: %s", exc, exc_info=True)
        finally:
            if self._proc is proc:
                logger.warning("codex app-server exited (code=%s)", proc.returncode)
                self._fail_all(CodexAppServerError("codex app-server exited"))

    async def _drain_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr
        with contextlib.suppress(asyncio.CancelledError):
            while line := await proc.stderr.readline():
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug("[codex] %s", text)

    def _dispatch(self, msg: JsonObject) -> None:
        if "id" in msg and "method" in msg:
            self._spawn(self._answer_server_request(msg))
        elif "id" in msg:
            fut = self._pending.get(msg["id"])
            if fut is None or fut.done():
                return
            if "error" in msg:
                err = msg["error"] or {}
                fut.set_exception(
                    CodexAppServerError(
                        str(err.get("message", err)), code=err.get("code")
                    )
                )
            else:
                fut.set_result(msg.get("result") or {})
        elif "method" in msg:
            thread_id = _thread_id_of(msg.get("params"))
            route = self._routes.get(thread_id) if thread_id else None
            if route is not None:
                route.events.put_nowait(msg)

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _answer_server_request(self, msg: JsonObject) -> None:
        method: str = msg["method"]
        params: JsonObject = msg.get("params") or {}
        route = self._routes.get(_thread_id_of(params) or "")
        try:
            if route and route.request_handler:
                result = await route.request_handler(method, params)
            else:
                result = default_server_request_response(method)
            await self._write({"id": msg["id"], "result": result})
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to answer codex request %s: %s", method, exc, exc_info=True
            )
            with contextlib.suppress(Exception):
                await self._write(
                    {"id": msg["id"], "error": {"code": -32000, "message": str(exc)}}
                )

    # ------------------------------------------------------------------ routing

    def open_route(
        self, thread_id: str, request_handler: ServerRequestHandler | None
    ) -> asyncio.Queue[JsonObject]:
        queue: asyncio.Queue[JsonObject] = asyncio.Queue()
        self._routes[thread_id] = _ThreadRoute(queue, request_handler)
        return queue

    def close_route(self, thread_id: str) -> None:
        self._routes.pop(thread_id, None)

    def is_loaded(self, thread_id: str) -> bool:
        return thread_id in self._loaded_threads

    def mark_loaded(self, thread_id: str) -> None:
        self._loaded_threads.add(thread_id)


def _thread_id_of(params: Any) -> str | None:
    if isinstance(params, dict):
        tid = params.get("threadId")
        if isinstance(tid, str):
            return tid
    return None


def default_server_request_response(method: str) -> JsonObject:
    """Answer for server requests that arrive with no active turn handler."""
    if method == "item/tool/call":
        return {
            "contentItems": [
                {
                    "type": "inputText",
                    "text": "No AstrBot session is attached to this call.",
                }
            ],
            "success": False,
        }
    if method in (
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
    ):
        return {"decision": "decline"}
    if method == "mcpServer/elicitation/request":
        return {"action": "decline"}
    return {}


_clients: dict[tuple, CodexAppServerClient] = {}
_clients_lock = asyncio.Lock()


async def get_shared_client(options: CodexLaunchOptions) -> CodexAppServerClient:
    """Return the app-server process for these launch options, starting it once."""
    async with _clients_lock:
        client = _clients.get(options.key())
        if client is None:
            client = CodexAppServerClient(options)
            _clients[options.key()] = client
    await client.ensure_started()
    return client


async def shutdown_shared_clients() -> None:
    async with _clients_lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        await client.close()
