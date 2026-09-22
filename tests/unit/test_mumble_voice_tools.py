import pytest

from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.permission_rules import EVENT_EXTRA_KEY, PermissionPolicy
from astrbot.core.platform.sources.mumble import voice_tools
from astrbot.core.star.context import Context

UMO = "mumble_test:GroupMessage:server"


async def _echo(event, text: str = "") -> str:
    return text


def _tool(name: str) -> FunctionTool:
    return FunctionTool(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=_echo,
    )


class FakeContext(Context):
    def __init__(self) -> None:  # none of the real managers are needed
        pass

    def get_config(self, umo=None):
        return {"admins_id": ["someone"]}


@pytest.fixture
def prepared(monkeypatch):
    seen = {}

    async def fake_prepare(event, req, ctx, cfg, runner_cfg):
        seen["event"] = event
        seen["umo"] = req.session_id
        req.func_tool = ToolSet([_tool("web_fetch"), _tool("cron_manage")])
        event.set_extra(
            EVENT_EXTRA_KEY,
            PermissionPolicy(tools_allow=("web_fetch",), native_exec=False),
        )

    import astrbot.core.pipeline.process_stage.method.agent_sub_stages.codex_request as codex_request
    import astrbot.core.star.context as star_context

    monkeypatch.setattr(codex_request, "prepare_codex_request", fake_prepare)
    monkeypatch.setattr(star_context, "_current", FakeContext())
    return seen


@pytest.mark.asyncio
async def test_voice_agent_gets_the_member_tools_of_its_chat(prepared, monkeypatch):
    stored = {}

    async def get_async(scope, scope_id, key, default=None):
        return stored.get((scope_id, key), default)

    import astrbot.core as core

    monkeypatch.setattr(core.sp, "get_async", get_async)
    tools = await voice_tools.voice_agent_tools(UMO, {"native_exec_tools": True})
    event = prepared["event"]
    assert prepared["umo"] == UMO
    assert event.unified_msg_origin == UMO
    # Whoever talks, channel voice acts as an ordinary member.
    assert event.role == "member"
    assert event.get_sender_id() == voice_tools.VOICE_SENDER_ID
    assert tools.native_exec is True
    names = [t["name"] for t in tools.dynamic_tools[0]["tools"]]
    assert set(names) == {"web_fetch", "cron_manage"}
    # Denied tools are refused when called (the policy is checked per call).
    denied = await tools.tool_handler(
        {"callId": "c1", "tool": "cron_manage", "namespace": None, "arguments": {}}
    )
    assert denied["success"] is False
    # Native exec follows the member rule (native_exec: false here).
    approved, _reason = await tools.approval_handler("exec", {})
    assert approved is False


@pytest.mark.asyncio
async def test_no_core_context_means_no_tools(monkeypatch):
    import astrbot.core.star.context as star_context

    monkeypatch.setattr(star_context, "_current", None)
    assert await voice_tools.voice_agent_tools(UMO, {}) is None
