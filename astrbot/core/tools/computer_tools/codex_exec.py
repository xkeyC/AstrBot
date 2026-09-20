"""Codex-style ``exec_command`` / ``write_stdin`` over AstrBot's runtimes.

Codex models are trained on this pair: a command runs in a session, the call
yields after ``yield_time_ms`` with whatever output arrived so far, and a
session id lets the model keep reading or send input. That beats one-shot
execution for anything interactive or slow (installers, REPLs, servers, test
watchers), so these replace the plain shell tools when Codex drives the turn.

The local runtime uses AstrBot's managed shell sessions. The Shipyard Neo SDK
only offers one-shot ``exec``, so a session is emulated inside the sandbox: the
command runs detached with its output in a log file, stdin arrives through a
FIFO held open by a sleeper process, and each poll reads the log from the last
offset.
"""

from __future__ import annotations

import shlex
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import FunctionTool
from astrbot.core import logger
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.computer.booters.local import LocalShellComponent
from astrbot.core.computer.computer_client import get_booter

from ..registry import builtin_tool
from .util import check_admin_permission, is_local_runtime, workspace_root_for_context

_COMPUTER_RUNTIME_TOOL_CONFIG = {
    "provider_settings.computer_use_runtime": ("local", "sandbox"),
}

DEFAULT_YIELD_TIME_MS = 10_000
MAX_YIELD_TIME_MS = 30_000
DEFAULT_MAX_OUTPUT_TOKENS = 10_000
# Codex counts tokens with a tokenizer; four bytes per token is close enough
# for a budget and never underestimates ASCII output.
CHARS_PER_TOKEN = 4
SANDBOX_SESSION_ROOT = "/tmp/astrbot-exec"
META_MARKER = "<<<ASTRBOT_EXEC_META"
TRUNCATION_MARKER = "[... omitted middle of output ...]"


def approx_token_count(text: str) -> int:
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN) if text else 0


def truncate_middle(text: str, max_tokens: int) -> tuple[str, bool]:
    """Keep the head and tail of ``text`` within the token budget."""
    budget = max(1, max_tokens) * CHARS_PER_TOKEN
    if len(text) <= budget:
        return text, False
    keep = max(1, (budget - len(TRUNCATION_MARKER) - 2) // 2)
    return f"{text[:keep]}\n{TRUNCATION_MARKER}\n{text[-keep:]}", True


def format_exec_response(
    *,
    output: str,
    wall_time_seconds: float,
    exit_code: int | None = None,
    session_id: str | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """Render one exec result the way Codex's own exec tools do."""
    original_tokens = approx_token_count(output)
    text, truncated = truncate_middle(output, max_output_tokens)
    sections = []
    if truncated:
        sections.append(f"Chunk ID: {uuid.uuid4().hex[:6]}")
    sections.append(f"Wall time: {wall_time_seconds:.4f} seconds")
    if exit_code is not None:
        sections.append(f"Process exited with code {exit_code}")
    if session_id is not None:
        sections.append(f"Process running with session ID {session_id}")
    if truncated:
        sections.append(f"Original token count: {original_tokens}")
    sections.append("Output:")
    return "\n".join(sections) + "\n" + text


def clamp_yield_time(value: Any, default: int = DEFAULT_YIELD_TIME_MS) -> int:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(ms, MAX_YIELD_TIME_MS))


def clamp_output_tokens(value: Any) -> int:
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_OUTPUT_TOKENS
    return max(256, min(tokens, DEFAULT_MAX_OUTPUT_TOKENS * 4))


# --------------------------------------------------------------- sandbox sessions


@dataclass
class SandboxSession:
    session_id: str
    directory: str
    cursor: int = 0


class SandboxSessions:
    """Sessions emulated inside a Shipyard Neo sandbox, keyed by chat."""

    _sessions: dict[str, dict[str, SandboxSession]] = {}

    @classmethod
    def get(cls, umo: str, session_id: str) -> SandboxSession | None:
        return cls._sessions.get(umo, {}).get(session_id)

    @classmethod
    def add(cls, umo: str, session: SandboxSession) -> None:
        cls._sessions.setdefault(umo, {})[session.session_id] = session

    @classmethod
    def drop(cls, umo: str, session_id: str) -> None:
        cls._sessions.get(umo, {}).pop(session_id, None)


def _poll_script(directory: str, cursor: int, yield_ms: int) -> str:
    """Wait inside the sandbox for new output or exit, then report both.

    The wait happens in the sandbox so one poll costs one round trip. The
    trailing metadata line carries the log size and exit code.
    """
    seconds = (max(0, yield_ms) + 999) // 1000
    return (
        f"d={shlex.quote(directory)}; c={cursor}; "
        f"end=$(( $(date +%s) + {seconds} )); "
        "while :; do "
        'size=$(wc -c < "$d/out.log" 2>/dev/null || echo 0); size=$((size + 0)); '
        '{ [ -f "$d/exit" ] || [ "$size" -gt "$c" ] || '
        '[ "$(date +%s)" -ge "$end" ]; } && break; '
        "sleep 0.2; done; "
        'tail -c +$((c + 1)) "$d/out.log" 2>/dev/null; '
        f'printf "\\n{META_MARKER} %s %s>>>\\n" '
        '"$(( $(wc -c < "$d/out.log" 2>/dev/null || echo 0) + 0 ))" '
        '"$(cat "$d/exit" 2>/dev/null || echo none)"'
    )


def _parse_poll(raw: str, cursor: int) -> tuple[str, int, int | None]:
    """Split a poll result into (output, new cursor, exit code)."""
    marker = raw.rfind(META_MARKER)
    end = raw.find(">>>", marker) if marker >= 0 else -1
    if marker < 0 or end < 0:
        return raw, cursor + len(raw.encode("utf-8", "replace")), None
    output = raw[:marker].rstrip("\n")
    meta = raw[marker + len(META_MARKER) : end].split()
    size = cursor + len(output.encode("utf-8", "replace"))
    exit_code: int | None = None
    if meta:
        try:
            size = int(meta[0])
        except ValueError:
            pass
    if len(meta) >= 2 and meta[1] != "none":
        try:
            exit_code = int(meta[1])
        except ValueError:
            exit_code = None
    return output, size, exit_code


async def _sandbox_exec(shell: Any, script: str, timeout: int) -> str:
    result = await shell.exec(script, timeout=timeout)
    if callable(getattr(result, "model_dump", None)):
        payload = result.model_dump()
    elif isinstance(result, dict):
        payload = result
    else:
        payload = getattr(result, "__dict__", {})
    output = str(payload.get("output") or payload.get("stdout") or "")
    error = str(payload.get("error") or payload.get("stderr") or "")
    return output + error


async def _start_sandbox_session(
    shell: Any, cmd: str, workdir: str | None, shell_binary: str | None
) -> SandboxSession:
    session_id = uuid.uuid4().hex[:8]
    directory = f"{SANDBOX_SESSION_ROOT}/{session_id}"
    runner = shell_binary or "sh"
    # The sleeper keeps a writer on the FIFO so the command does not see EOF
    # between write_stdin calls.
    script = (
        f"mkdir -p {shlex.quote(directory)} && cd {shlex.quote(workdir or '.')} && "
        f"mkfifo {shlex.quote(directory + '/stdin')} 2>/dev/null; "
        f": > {shlex.quote(directory + '/out.log')}; "
        f"(sleep 86400 > {shlex.quote(directory + '/stdin')} &) ; "
        f"( {runner} -c {shlex.quote(cmd)} "
        f"< {shlex.quote(directory + '/stdin')} "
        f"> {shlex.quote(directory + '/out.log')} 2>&1; "
        f"echo $? > {shlex.quote(directory + '/exit')} ) >/dev/null 2>&1 & "
        f"echo $! > {shlex.quote(directory + '/pid')}"
    )
    await _sandbox_exec(shell, script, timeout=30)
    return SandboxSession(session_id=session_id, directory=directory)


# -------------------------------------------------------------------- the tools


@builtin_tool(config=_COMPUTER_RUNTIME_TOOL_CONFIG)
@dataclass
class ExecCommandTool(FunctionTool):
    name: str = "exec_command"
    description: str = (
        "Runs a command in a shell session, returning output or a session ID for "
        "ongoing interaction. Use write_stdin to keep reading or to answer prompts."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to execute."},
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command. Defaults to the workspace root.",
                },
                "yield_time_ms": {
                    "type": "number",
                    "description": (
                        "Wait before yielding output. Defaults to 10000 ms; "
                        "effective range is 250-30000 ms. Commands that finish "
                        "sooner return immediately."
                    ),
                },
                "max_output_tokens": {
                    "type": "number",
                    "description": "Output token budget. Defaults to 10000 tokens.",
                },
                "shell": {
                    "type": "string",
                    "description": "Shell binary to launch. Defaults to the runtime's default shell.",
                },
            },
            "required": ["cmd"],
            "additionalProperties": False,
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        cmd: str,
        workdir: str | None = None,
        yield_time_ms: Any = DEFAULT_YIELD_TIME_MS,
        max_output_tokens: Any = DEFAULT_MAX_OUTPUT_TOKENS,
        shell: str | None = None,
    ) -> ToolExecResult:
        if permission_error := check_admin_permission(context, "Shell execution"):
            return permission_error
        event = context.context.event
        umo = event.unified_msg_origin
        wait_ms = clamp_yield_time(yield_time_ms)
        budget = clamp_output_tokens(max_output_tokens)
        booter = await get_booter(context.context.context, umo)
        started = time.monotonic()

        if is_local_runtime(context):
            if not isinstance(booter.shell, LocalShellComponent):
                return "Error: the local shell component is unavailable."
            creator_id = event.get_sender_id()
            if not creator_id:
                return "Error: sender identity is unavailable."
            cwd = workdir
            if not cwd:
                root = await workspace_root_for_context(context)
                root.mkdir(parents=True, exist_ok=True)
                cwd = str(root)
            try:
                result = await booter.shell.exec_managed(
                    cmd,
                    owner_id=umo,
                    creator_id=creator_id,
                    creator_is_admin=event.role == "admin",
                    sandboxed=False,
                    cwd=cwd,
                    env={},
                    timeout=None,
                    yield_time_ms=wait_ms,
                    max_output_chars=budget * CHARS_PER_TOKEN,
                )
            except (PermissionError, ValueError) as e:
                return f"Error: {e}"
            running = result.get("exit_code") is None
            return format_exec_response(
                output=str(result.get("stdout") or ""),
                wall_time_seconds=time.monotonic() - started,
                exit_code=result.get("exit_code"),
                session_id=str(result["session_id"]) if running else None,
                max_output_tokens=budget,
            )

        try:
            session = await _start_sandbox_session(booter.shell, cmd, workdir, shell)
        except Exception as e:  # noqa: BLE001 - sandbox or transport failure
            logger.error("exec_command failed to start a sandbox session: %s", e)
            return f"Error starting command: {e}"
        raw = await _sandbox_exec(
            booter.shell,
            _poll_script(session.directory, 0, wait_ms),
            timeout=max(30, wait_ms // 1000 + 15),
        )
        output, cursor, exit_code = _parse_poll(raw, 0)
        session.cursor = cursor
        if exit_code is None:
            SandboxSessions.add(umo, session)
        return format_exec_response(
            output=output,
            wall_time_seconds=time.monotonic() - started,
            exit_code=exit_code,
            session_id=session.session_id if exit_code is None else None,
            max_output_tokens=budget,
        )


@builtin_tool(config=_COMPUTER_RUNTIME_TOOL_CONFIG)
@dataclass
class WriteStdinTool(FunctionTool):
    name: str = "write_stdin"
    description: str = (
        "Writes characters to an existing exec_command session and returns recent "
        "output. Pass empty chars to poll a running command without writing."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Identifier of the running session, as reported by exec_command.",
                },
                "chars": {
                    "type": "string",
                    "description": (
                        "Bytes to write to stdin. Defaults to empty, which polls "
                        "without writing. Append a newline to submit a line."
                    ),
                },
                "yield_time_ms": {
                    "type": "number",
                    "description": "Wait before yielding output. Defaults to 5000 ms, capped at 30000 ms.",
                },
                "max_output_tokens": {
                    "type": "number",
                    "description": "Output token budget. Defaults to 10000 tokens.",
                },
            },
            "required": ["session_id"],
            "additionalProperties": False,
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        session_id: str,
        chars: str = "",
        yield_time_ms: Any = 5_000,
        max_output_tokens: Any = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> ToolExecResult:
        if permission_error := check_admin_permission(context, "Shell execution"):
            return permission_error
        event = context.context.event
        umo = event.unified_msg_origin
        wait_ms = clamp_yield_time(yield_time_ms, default=5_000)
        budget = clamp_output_tokens(max_output_tokens)
        booter = await get_booter(context.context.context, umo)
        started = time.monotonic()

        if is_local_runtime(context):
            if not isinstance(booter.shell, LocalShellComponent):
                return "Error: the local shell component is unavailable."
            requester_id = event.get_sender_id()
            is_admin = event.role == "admin"
            try:
                if chars:
                    await booter.shell.write_session(
                        owner_id=umo,
                        requester_id=requester_id,
                        requester_is_admin=is_admin,
                        session_id=session_id,
                        chars=chars,
                    )
                result = await booter.shell.poll_session(
                    owner_id=umo,
                    requester_id=requester_id,
                    requester_is_admin=is_admin,
                    session_id=session_id,
                    yield_time_ms=wait_ms,
                    max_output_chars=budget * CHARS_PER_TOKEN,
                )
            except ValueError as e:
                return f"Error: {e}"
            running = result.get("exit_code") is None
            return format_exec_response(
                output=str(result.get("stdout") or ""),
                wall_time_seconds=time.monotonic() - started,
                exit_code=result.get("exit_code"),
                session_id=session_id if running else None,
                max_output_tokens=budget,
            )

        session = SandboxSessions.get(umo, session_id)
        if session is None:
            return f"Error: unknown session {session_id}."
        if chars:
            await _sandbox_exec(
                booter.shell,
                f"printf %s {shlex.quote(chars)} > {shlex.quote(session.directory + '/stdin')}",
                timeout=30,
            )
        raw = await _sandbox_exec(
            booter.shell,
            _poll_script(session.directory, session.cursor, wait_ms),
            timeout=max(30, wait_ms // 1000 + 15),
        )
        output, cursor, exit_code = _parse_poll(raw, session.cursor)
        session.cursor = cursor
        if exit_code is not None:
            SandboxSessions.drop(umo, session_id)
        return format_exec_response(
            output=output,
            wall_time_seconds=time.monotonic() - started,
            exit_code=exit_code,
            session_id=session_id if exit_code is None else None,
            max_output_tokens=budget,
        )
