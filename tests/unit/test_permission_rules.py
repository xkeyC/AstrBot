from astrbot.core.permission_rules import (
    DEFAULT_POLICY,
    EventFacts,
    condition_matches,
    resolve_policy,
)

RULES = [
    {"name": "off", "enabled": False, "match": ["*"], "tools_deny": ["*"]},
    {"name": "vip", "match": "p_1\n", "tools_allow": ["*"], "persona_id": "vip"},
    {"name": "grp", "match": ["g_100"], "tools_deny": ["shell_*"], "mcp_deny": ["fs"]},
    {
        "name": "one",
        "match": ["200/2"],
        "tools_allow": ["weather"],
        "native_exec": "false",
    },
]


def facts(sender="9", group="", role="member"):
    return EventFacts(sender_id=sender, group_id=group, role=role)


def test_condition_syntax():
    assert condition_matches("p_1", facts("1"))
    assert condition_matches("g_100", facts("5", "100"))
    assert not condition_matches("g_100", facts("5", ""))
    assert condition_matches("200/2", facts("2", "200"))
    assert not condition_matches("200/2", facts("2", "201"))
    assert condition_matches("role:admin", facts(role="admin"))
    assert condition_matches("*", facts())


def test_first_enabled_match_wins_and_default():
    assert resolve_policy(RULES, facts("1", "100")).rule_name == "vip"
    assert resolve_policy(RULES, facts("5", "100")).rule_name == "grp"
    assert resolve_policy(RULES, facts("5")) is DEFAULT_POLICY
    assert resolve_policy(RULES, facts("1")).persona_id == "vip"


def test_tool_and_mcp_checks():
    grp = resolve_policy(RULES, facts("5", "100"))
    assert not grp.allows_tool("shell_exec")
    assert grp.allows_tool("weather")
    assert not grp.allows_tool("read", mcp_server="fs")
    assert grp.allows_tool("search", mcp_server="web")
    one = resolve_policy(RULES, facts("2", "200"))
    assert one.allows_tool("weather")
    assert not one.allows_tool("counter")
    assert one.native_exec is False
    assert "allowed tools: weather" in one.summary()
    assert DEFAULT_POLICY.is_default and not one.is_default


def test_tools_allow_combined_with_mcp_lists():
    from astrbot.core.permission_rules import PermissionPolicy

    both = PermissionPolicy(tools_allow=("weather",), mcp_allow=("web",))
    assert both.allows_tool("weather")
    assert not both.allows_tool("counter")
    assert both.allows_tool("search", mcp_server="web")
    assert not both.allows_tool("read", mcp_server="fs")
    only_tools = PermissionPolicy(tools_allow=("weather",))
    assert not only_tools.allows_tool("read", mcp_server="fs")
    deny = PermissionPolicy(tools_deny=("read",), mcp_allow=("fs",))
    assert not deny.allows_tool("read", mcp_server="fs")
    assert (
        resolve_policy([{"match": ["*"], "enabled": "false"}], facts())
        is DEFAULT_POLICY
    )


def test_dynamic_persona_bindings_migrate_once(tmp_path):
    import json

    from astrbot.core.utils.migra_helper import migrate_config_on_load

    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "astrbot_plugin_DynamicPersona_config.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "persona_bindings": [
                    {
                        "rule_enabled": True,
                        "rule_name": "vip",
                        "match_conditions": "p_1\n\n g_2 ",
                        "persona_id": "admin",
                        "provider_id": "gpt",
                    },
                    {"match_conditions": ""},
                ],
            }
        ),
        encoding="utf-8",
    )
    conf = {"agent_runner": {"runner_type": "codex", "config": {}}, "config_version": 3}
    migrate_config_on_load(conf, tmp_path / "cmd_config.json")
    assert conf["permission_rules"] == [
        {"name": "vip", "enabled": True, "match": ["p_1", "g_2"], "persona_id": "admin"}
    ]
    conf["permission_rules"][0]["persona_id"] = "edited"
    migrate_config_on_load(conf, tmp_path / "cmd_config.json")
    assert conf["permission_rules"][0]["persona_id"] == "edited"
    # Rules the user cleared afterwards are not imported again.
    conf["permission_rules"] = []
    migrate_config_on_load(conf, tmp_path / "cmd_config.json")
    assert conf["permission_rules"] == []
    other = {"agent_runner": {"runner_type": "codex", "config": {}}}
    migrate_config_on_load(other, tmp_path / "abconf_x.json")
    assert "permission_rules" not in other


def test_summary_names_the_sandbox_tools_that_still_work():
    from astrbot.core.permission_rules import PermissionPolicy

    denied = PermissionPolicy(native_exec=False)
    # Host execution is on for the bot, so the restriction is worth stating,
    # and the model is pointed at what it can still use.
    assert denied.summary(
        host_exec=True, sandbox_tools=("exec_command", "write_stdin")
    ) == (
        "no command execution on the bot host "
        "(the sandbox is unaffected: use exec_command, write_stdin)"
    )
    assert denied.summary(host_exec=True) == "no command execution on the bot host"
    # Nothing runs on the host anyway: saying so would only confuse the model.
    assert denied.summary(host_exec=False) == ""
    assert denied.summary(host_exec=False, sandbox_tools=("exec_command",)) == ""

    mixed = PermissionPolicy(tools_deny=("weather",), native_exec=False)
    assert mixed.summary(host_exec=False) == "denied tools: weather"
