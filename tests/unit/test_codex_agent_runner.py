import asyncio
from types import SimpleNamespace

from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.codex.codex_agent_runner import (
    build_additional_context,
    build_turn_input,
    engine_options,
)
from astrbot.core.agent.runners.codex.tool_bridge import CodexToolBridge
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.provider.entities import ProviderRequest


async def _echo(event, text: str = ""):
    return f"echo:{text}"


def _tool(name: str, params: dict | None = None) -> FunctionTool:
    return FunctionTool(
        name=name,
        description=f"{name} tool",
        parameters=params
        if params is not None
        else {"type": "object", "properties": {"text": {"type": "string"}}},
        handler=_echo,
    )


def test_bridge_sanitizes_and_dedupes_names():
    bridge = CodexToolBridge(
        ToolSet([_tool("a.b"), _tool("a_b"), _tool("mcp__x"), _tool("ok", {})])
    )
    names = [spec["name"] for spec in bridge.specs]
    assert names == ["a_b", "a_b_2", "ext_mcp__x", "ok"]
    ok_spec = next(s for s in bridge.specs if s["name"] == "ok")
    assert ok_spec["inputSchema"] == {"type": "object", "properties": {}}
    [namespace] = bridge.dynamic_tools()
    assert namespace["type"] == "namespace"
    assert namespace["name"] == "astrbot"


def test_bridge_fingerprint_tracks_tool_set():
    a = CodexToolBridge(ToolSet([_tool("x")]))
    b = CodexToolBridge(ToolSet([_tool("x")]))
    c = CodexToolBridge(ToolSet([_tool("x"), _tool("y")]))
    assert a.fingerprint == b.fingerprint != c.fingerprint
    assert CodexToolBridge(None).dynamic_tools() == []


def test_bridge_call_runs_tool_and_hooks():
    bridge = CodexToolBridge(ToolSet([_tool("echo")]))
    event = SimpleNamespace(get_result=lambda: None)
    ctx = ContextWrapper(context=SimpleNamespace(event=event), tool_call_timeout=5)
    seen = []

    class Hooks(BaseAgentRunHooks):
        async def on_tool_start(self, run_context, tool, tool_args):
            seen.append(("start", tool.name, tool_args))

        async def on_tool_end(self, run_context, tool, tool_args, tool_result):
            seen.append(("end", tool.name))

    result = asyncio.run(
        bridge.call(
            {
                "namespace": "astrbot",
                "tool": "echo",
                "arguments": {"text": "hi", "junk": 1},
            },
            ctx,
            Hooks(),
        )
    )
    assert result == {
        "contentItems": [{"type": "inputText", "text": "echo:hi"}],
        "success": True,
    }
    assert seen == [("start", "echo", {"text": "hi"}), ("end", "echo")]

    missing = asyncio.run(
        bridge.call({"namespace": "astrbot", "tool": "nope"}, ctx, Hooks())
    )
    assert missing["success"] is False


def test_turn_input_orders_context_prompt_and_media():
    req = ProviderRequest(prompt="hello")
    req.add_temporary_context("kb", "retrieved facts")
    req.add_persistent_context("message_meta", "Sender: alice")
    req.image_urls = ["https://example.com/a.png", "C:/tmp/b.png"]
    items = build_turn_input(req)
    texts = [i.get("text", "") for i in items if i["type"] == "text"]
    assert "retrieved facts" in texts[0]
    assert "Sender: alice" in texts[1]
    assert texts[2] == "hello"
    assert items[-2] == {"type": "image", "image_url": "https://example.com/a.png"}
    assert items[-1] == {"type": "local_image", "path": "C:/tmp/b.png"}


def test_additional_context_maps_anchors_and_system_prompt():
    req = ProviderRequest(prompt="x", system_prompt=" be nice ")
    req.set_context_anchor("persona", "cat girl")
    assert build_additional_context(req) == {
        "astrbot_system_prompt": {"value": "be nice", "kind": "application"},
        "astrbot_persona": {"value": "cat girl", "kind": "application"},
    }


def test_bridge_defers_tools_in_code_mode():
    bridge = CodexToolBridge(ToolSet([_tool("x")]), defer=True)
    assert bridge.specs[0]["deferLoading"] is True
    assert "deferLoading" not in CodexToolBridge(ToolSet([_tool("x")])).specs[0]


def test_engine_options_default_to_lean_code_mode(tmp_path):
    opts = engine_options({"codex_home": str(tmp_path), "code_mode_host": ""})
    cfg = opts["config"]
    assert cfg["model_tool_mode"] == "code_mode_only"
    assert cfg["features.shell_tool"] is False
    assert cfg["web_search"] == "disabled"
    assert cfg["include_permissions_instructions"] is False
    assert cfg["features.code_mode.structured_dynamic_tool_results"] is True
    overridden = engine_options(
        {
            "codex_home": str(tmp_path),
            "tool_mode": "direct",
            "thread_config": {"web_search": "live"},
        }
    )
    assert overridden["config"]["model_tool_mode"] == "direct"
    assert overridden["config"]["web_search"] == "live"


def test_read_skill_file_stays_inside_skill(tmp_path):
    from astrbot.core.agent.runners.codex.skills import (
        build_codex_skills_prompt,
        read_skill_file,
    )
    from astrbot.core.skills.skill_manager import SkillInfo

    skill_dir = tmp_path / "demo"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: d\n---\nBODY", encoding="utf-8"
    )
    (skill_dir / "scripts" / "run.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    skills = [
        SkillInfo(
            name="demo", description="d", path=str(skill_dir / "SKILL.md"), active=True
        )
    ]

    assert "BODY" in read_skill_file(skills, "demo")
    assert read_skill_file(skills, "demo", "scripts/run.py") == "print(1)"
    assert "run.py" in read_skill_file(skills, "demo", "scripts")
    assert read_skill_file(skills, "demo", "../secret.txt").startswith("error")
    assert read_skill_file(skills, "missing").startswith("error: unknown skill")
    assert "astrbot__astrbot_read_skill" in build_codex_skills_prompt(skills)


def test_read_skill_rejects_absolute_and_unc(tmp_path):
    from astrbot.core.agent.runners.codex.skills import read_skill_file
    from astrbot.core.skills.skill_manager import SkillInfo

    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "SKILL.md").write_text("x", encoding="utf-8")
    skills = [
        SkillInfo(
            name="s", description="", path=str(tmp_path / "s" / "SKILL.md"), active=True
        )
    ]
    for bad in (
        r"\\attacker\share\x",
        r"C:\Windows\win.ini",
        "/etc/passwd",
        "../x",
        "C:x",
    ):
        assert read_skill_file(skills, "s", bad).startswith("error"), bad


def test_model_providers_map_to_codex_overrides(tmp_path):
    opts = engine_options(
        {
            "codex_home": str(tmp_path),
            "tool_mode": "direct",
            "model_provider": "my.relay",
            "model_providers": [
                {"id": "my.relay", "base_url": "https://r/v1", "api_key": "sk-1"},
                {"id": "no-url"},
                "junk",
            ],
        }
    )
    cfg = opts["config"]
    assert cfg["model_providers.my_relay.base_url"] == "https://r/v1"
    assert cfg["model_providers.my_relay.experimental_bearer_token"] == "sk-1"
    assert cfg["model_providers.my_relay.wire_api"] == "responses"
    assert not any(k.startswith("model_providers.no-url") for k in cfg)


def test_native_exec_approvals_follow_permission_rules(tmp_path):
    from astrbot.core.agent.runners.codex.codex_agent_runner import (
        native_exec_decision,
    )
    from astrbot.core.permission_rules import EVENT_EXTRA_KEY, PermissionPolicy

    opts = engine_options(
        {"codex_home": str(tmp_path), "tool_mode": "direct", "native_exec_tools": True}
    )
    assert opts["approve_every_command"] is True
    assert "approval_policy" not in opts["config"]
    plain = engine_options({"codex_home": str(tmp_path), "tool_mode": "direct"})
    assert "approve_every_command" not in plain
    assert plain["config"]["approval_policy"] == "never"

    def event(policy):
        return SimpleNamespace(
            get_extra=lambda key: policy if key == EVENT_EXTRA_KEY else None
        )

    assert native_exec_decision(event(None)) == (True, "")
    assert native_exec_decision(event(PermissionPolicy(native_exec=True)))[0] is True
    denied = native_exec_decision(event(PermissionPolicy(native_exec=False)))
    assert denied[0] is False and "not permitted" in denied[1]
    off = native_exec_decision(event(PermissionPolicy(native_exec=True)), False)
    assert off[0] is False and "turned off" in off[1]
    assert native_exec_decision(event(None), True) == (True, "")


def test_memory_thread_config_limits_global_to_permitted_private_chats(tmp_path):
    from astrbot.core.agent.runners.codex.codex_agent_runner import (
        memory_thread_config,
    )
    from astrbot.core.permission_rules import EVENT_EXTRA_KEY, PermissionPolicy

    def event(policy, group=""):
        return SimpleNamespace(
            get_extra=lambda key: policy if key == EVENT_EXTRA_KEY else None,
            get_group_id=lambda: group,
        )

    allowed = PermissionPolicy(global_memory=True)
    cfg = {"memory_auto_consolidate": False}
    conf = memory_thread_config(cfg, "qq:FriendMessage:1", event(allowed))
    assert conf["memories.scope_key"] == "qq:FriendMessage:1"
    assert conf["memories.may_write_global"] is True
    assert conf["memories.auto_consolidate"] is False
    assert conf["memories.extra_session_sources"] == ["astrbot"]
    assert (
        memory_thread_config(cfg, "u", event(allowed, group="9"))[
            "memories.may_write_global"
        ]
        is False
    )
    assert (
        memory_thread_config(cfg, "u", event(None))["memories.may_write_global"]
        is False
    )
    exe = tmp_path / "codex.exe"
    exe.write_bytes(b"")
    opts = engine_options(
        {
            "codex_home": str(tmp_path),
            "tool_mode": "direct",
            "memory_enabled": True,
            "codex_self_exe": str(exe),
        }
    )
    assert opts["codex_self_exe"] == str(exe)


def test_auto_approve_off_denies_explicit_approval_policies():
    from astrbot.core.agent.runners.codex.codex_agent_runner import (
        approvals_disabled,
    )

    assert approvals_disabled({}) is False
    assert approvals_disabled({"approval_policy": "never"}) is False
    assert approvals_disabled({"approval_policy": "on-request"}) is True
    assert (
        approvals_disabled({"approval_policy": "on-request", "auto_approve": True})
        is False
    )


def test_approval_is_answered_for_the_turn_it_arrived_in():
    from astrbot.core.agent.runners.codex.native import ThreadPump, _TurnRoute

    decisions: list[str] = []

    class Rt:
        async def review_decision(self, thread_id, request):
            decisions.append(request)

    async def approve(kind, msg):
        return True, ""

    async def run():
        import json

        pump = ThreadPump(SimpleNamespace(rt=Rt(), pumps={}), "t1")
        route = _TurnRoute(asyncio.Queue(), None, approve)
        msg = {"type": "exec_approval_request", "call_id": "c1", "turn_id": "u1"}
        await pump._answer_approval("exec", msg, route)
        await pump._answer_approval("exec", msg, None)
        first, second = (json.loads(d) for d in decisions)
        assert first["approved"] is True and first["id"] == "c1"
        assert second["approved"] is False and second["reason"]

    asyncio.run(run())


def test_persona_catalog_only_when_personas_alternate():
    from astrbot.core.agent.runners.codex.codex_agent_runner import persona_context

    def req(persona):
        r = ProviderRequest(prompt="hi")
        if persona:
            r.set_context_anchor("persona", persona)
        return r

    catalog, ctx, active = persona_context(req("cat"), {})
    assert active is None and ctx["astrbot_persona"]["value"] == "cat"
    same, ctx, active = persona_context(req("cat"), catalog)
    assert same == catalog and active is None

    both, ctx_dog, active = persona_context(req("dog"), catalog)
    assert len(both) == 2 and "astrbot_persona" not in ctx_dog
    assert "cat" in ctx_dog["astrbot_personas"]["value"]
    assert active["text"].startswith("<active_persona id=")
    # Alternating back does not change the standing context (cache-safe).
    again, ctx_cat, active_cat = persona_context(req("cat"), both)
    assert again == both and ctx_cat == ctx_dog and active_cat != active
    _, _, none_active = persona_context(req(""), both)
    assert "none" in none_active["text"]


def test_code_mode_hides_denied_tools_but_direct_mode_lists_them():
    from astrbot.core.permission_rules import PermissionPolicy

    tools = ToolSet([_tool("weather"), _tool("calc")])
    denied = PermissionPolicy(rule_name="r", tools_deny=("weather",))

    hidden = CodexToolBridge(tools, defer=True, policy=denied)
    assert [spec["name"] for spec in hidden.specs] == ["calc"]
    assert "weather" not in hidden.tools

    # Without deferral the tool set must stay identical for every sender, or
    # the prompt prefix (and its cache) would differ per user.
    listed = CodexToolBridge(tools, defer=False, policy=denied)
    assert [spec["name"] for spec in listed.specs] == ["calc", "weather"]

    # No policy: nothing is filtered in either mode.
    assert len(CodexToolBridge(tools, defer=True).specs) == 2


def test_hidden_tools_change_the_fingerprint_per_sender():
    from astrbot.core.permission_rules import PermissionPolicy

    tools = ToolSet([_tool("weather"), _tool("calc")])
    allowed = CodexToolBridge(tools, defer=True, policy=PermissionPolicy())
    denied = CodexToolBridge(
        tools, defer=True, policy=PermissionPolicy(tools_deny=("weather",))
    )
    # A different tool set means the runner sends a tools update for that turn.
    assert allowed.fingerprint != denied.fingerprint
