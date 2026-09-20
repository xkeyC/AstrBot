import asyncio
from types import SimpleNamespace

import pytest

from astrbot.core.tools.computer_tools import codex_exec
from astrbot.core.tools.computer_tools.codex_exec import (
    ExecCommandTool,
    SandboxSessions,
    WriteStdinTool,
    _parse_poll,
    _poll_script,
    format_exec_response,
    truncate_middle,
)


def _context(monkeypatch, booter, *, local: bool, role: str = "admin"):
    event = SimpleNamespace(
        unified_msg_origin="qq:FriendMessage:1",
        get_sender_id=lambda: "1",
        role=role,
    )
    ctx = SimpleNamespace(context=SimpleNamespace(event=event, context=None))

    async def fake_get_booter(_context, _umo):
        return booter

    monkeypatch.setattr(codex_exec, "get_booter", fake_get_booter)
    monkeypatch.setattr(codex_exec, "check_admin_permission", lambda *_: None)
    monkeypatch.setattr(codex_exec, "is_local_runtime", lambda _: local)
    return ctx


def test_response_format_matches_codex_shape():
    text = format_exec_response(output="hello", wall_time_seconds=1.25, exit_code=0)
    assert text.splitlines()[0] == "Wall time: 1.2500 seconds"
    assert "Process exited with code 0" in text
    assert text.endswith("Output:\nhello")

    running = format_exec_response(
        output="", wall_time_seconds=0.5, session_id="abc123"
    )
    assert "Process running with session ID abc123" in running
    assert "Process exited" not in running


def test_long_output_is_truncated_in_the_middle():
    body = "x" * 10_000
    kept, truncated = truncate_middle(body, 100)
    assert truncated and len(kept) < len(body)
    assert kept.startswith("x") and kept.endswith("x")

    text = format_exec_response(
        output=body, wall_time_seconds=0.1, exit_code=0, max_output_tokens=100
    )
    assert "Chunk ID: " in text
    assert "Original token count: 2500" in text
    assert "omitted middle of output" in text


def test_local_runtime_reports_session_while_running(monkeypatch):
    calls = {}

    class Shell(codex_exec.LocalShellComponent):
        async def exec_managed(self, command, **kwargs):
            calls["command"] = command
            calls["yield_time_ms"] = kwargs["yield_time_ms"]
            return {
                "session_id": "s-1",
                "stdout": "building...",
                "exit_code": None,
            }

        async def poll_session(self, **kwargs):
            calls["poll"] = kwargs
            return {"session_id": "s-1", "stdout": "done", "exit_code": 0}

        async def write_session(self, **kwargs):
            calls["write"] = kwargs["chars"]
            return {"status": "running"}

    booter = SimpleNamespace(shell=Shell())
    ctx = _context(monkeypatch, booter, local=True)

    out = asyncio.run(ExecCommandTool().call(ctx, "make", workdir="/w"))
    assert calls["command"] == "make" and calls["yield_time_ms"] == 10_000
    assert "Process running with session ID s-1" in out
    assert out.endswith("building...")

    out = asyncio.run(WriteStdinTool().call(ctx, "s-1", chars="y\n"))
    assert calls["write"] == "y\n"
    assert "Process exited with code 0" in out


def test_local_runtime_surfaces_blocked_commands(monkeypatch):
    class Shell(codex_exec.LocalShellComponent):
        async def exec_managed(self, command, **kwargs):
            raise PermissionError("Blocked unsafe shell command.")

    ctx = _context(monkeypatch, SimpleNamespace(shell=Shell()), local=True)
    assert "Blocked unsafe shell command." in asyncio.run(
        ExecCommandTool().call(ctx, "rm -rf /")
    )


class FakeSandboxShell:
    """Minimal stand-in for the Shipyard Neo one-shot shell capability."""

    def __init__(self, polls):
        self.polls = list(polls)
        self.scripts = []

    async def exec(self, command, *, timeout=30, **_kwargs):
        self.scripts.append(command)
        if "out.log" in command and "tail -c" in command:
            return SimpleNamespace(
                output=self.polls.pop(0), error=None, model_dump=None
            )
        return {"output": "", "error": None}


def test_sandbox_session_lifecycle(monkeypatch):
    marker = codex_exec.META_MARKER
    shell = FakeSandboxShell(
        [
            f"waiting for input\n{marker} 18 none>>>\n",
            f"got it\n{marker} 25 0>>>\n",
        ]
    )
    ctx = _context(monkeypatch, SimpleNamespace(shell=shell), local=False)

    out = asyncio.run(ExecCommandTool().call(ctx, "python repl.py", workdir="/w"))
    assert "Process running with session ID" in out
    assert out.endswith("waiting for input")
    start_script = shell.scripts[0]
    assert "mkfifo" in start_script and "sleep 86400" in start_script
    session_id = out.split("session ID ")[1].splitlines()[0]
    session = SandboxSessions.get("qq:FriendMessage:1", session_id)
    assert session is not None and session.cursor == 18

    out = asyncio.run(WriteStdinTool().call(ctx, session_id, chars="hi\n"))
    assert any("printf %s" in script for script in shell.scripts)
    assert "Process exited with code 0" in out
    # A finished session is forgotten.
    assert SandboxSessions.get("qq:FriendMessage:1", session_id) is None

    assert "unknown session" in asyncio.run(
        WriteStdinTool().call(ctx, "does-not-exist")
    )


@pytest.mark.parametrize(
    ("raw", "cursor", "expected"),
    [
        (f"out\n{codex_exec.META_MARKER} 12 0>>>\n", 0, ("out", 12, 0)),
        (f"{codex_exec.META_MARKER} 3 none>>>\n", 3, ("", 3, None)),
        ("no marker", 5, ("no marker", 14, None)),
    ],
)
def test_parse_poll(raw, cursor, expected):
    assert _parse_poll(raw, cursor) == expected


def test_poll_script_waits_inside_the_sandbox():
    script = _poll_script("/tmp/astrbot-exec/abc", 42, 3_000)
    assert "c=42" in script and "+ 3 ))" in script
    assert "tail -c +$((c + 1))" in script
    assert codex_exec.META_MARKER in script


def test_shipyard_mode_forces_sandbox_and_disables_native_exec(tmp_path):
    from astrbot.core.agent.runners.codex.codex_agent_runner import engine_options
    from astrbot.core.agent.runners.codex.skills import build_codex_skills_prompt
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.codex_request import (
        _build_config,
    )
    from astrbot.core.skills.skill_manager import SkillInfo

    plugin_context = SimpleNamespace(get_config=lambda: {"timezone": "UTC"})
    astrbot_config = {"provider_settings": {"computer_use_runtime": "none"}}
    config = _build_config(astrbot_config, {"shipyard_mode": True}, plugin_context)
    assert config.computer_use_runtime == "sandbox"
    assert config.provider_settings["_codex_skills"] == "sandbox"

    plain = _build_config(astrbot_config, {}, plugin_context)
    assert plain.computer_use_runtime == "none"
    assert plain.provider_settings["_codex_skills"] is True

    opts = engine_options(
        {
            "codex_home": str(tmp_path),
            "tool_mode": "direct",
            "native_exec_tools": True,
            "shipyard_mode": True,
        }
    )
    assert opts["config"]["features.shell_tool"] is False
    assert "approve_every_command" not in opts

    skills = [SkillInfo(name="demo", description="d", path="x/SKILL.md", active=True)]
    assert "skills/<name>/SKILL.md" in build_codex_skills_prompt(
        skills, in_sandbox=True
    )
    assert "astrbot__astrbot_read_skill" in build_codex_skills_prompt(skills)
