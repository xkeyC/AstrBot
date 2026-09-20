"""Per-user permission rules (replaces the DynamicPersona plugin's bindings).

Rules are evaluated in order; the first enabled rule whose match conditions
hit the event wins. Matching syntax (compatible with DynamicPersona):

- ``<group_id>/<sender_id>``: that sender in that group
- ``p_<sender_id>``: that sender anywhere
- ``g_<group_id>``: everyone in that group
- ``role:admin`` / ``role:member``: by AstrBot role
- ``*``: everyone

Tool permissions are always enforced when a tool is called. Under code mode
the denied tools are also left out of the set the model can see, which costs
nothing because deferred tool specs never enter the prompt prefix; without
deferral they stay listed, so the prefix (and its cache) is the same for
everyone.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any

CONFIG_KEY = "permission_rules"
EVENT_EXTRA_KEY = "_permission_policy"


@dataclass(frozen=True)
class PermissionPolicy:
    rule_name: str = ""
    tools_allow: tuple[str, ...] = ()
    tools_deny: tuple[str, ...] = ()
    mcp_allow: tuple[str, ...] = ()
    mcp_deny: tuple[str, ...] = ()
    persona_id: str = ""
    model: str = ""
    # Advisory until the runner enforces it through Codex exec approvals:
    # native execution is off by default, so this only matters when an admin
    # enables native_exec_tools.
    native_exec: bool | None = None
    # Consumed by the scoped memory module (may_write_global).
    global_memory: bool | None = None

    @property
    def is_default(self) -> bool:
        return not (
            self.tools_allow
            or self.tools_deny
            or self.mcp_allow
            or self.mcp_deny
            or self.native_exec is False
        )

    def allows_tool(self, tool_name: str, mcp_server: str | None = None) -> bool:
        """Deny lists always win. MCP tools are decided by the MCP lists when
        either is set; otherwise, like plugin tools, by the tool lists."""
        if any(fnmatch.fnmatchcase(tool_name, p) for p in self.tools_deny):
            return False
        if mcp_server and (self.mcp_allow or self.mcp_deny):
            if any(fnmatch.fnmatchcase(mcp_server, p) for p in self.mcp_deny):
                return False
            if self.mcp_allow:
                return any(fnmatch.fnmatchcase(mcp_server, p) for p in self.mcp_allow)
            return True
        if self.tools_allow:
            return any(fnmatch.fnmatchcase(tool_name, p) for p in self.tools_allow)
        return True

    def summary(self, *, host_exec: bool = True, sandbox_tools: tuple = ()) -> str:
        """Short, model-facing description of the sender's restrictions.

        Args:
            host_exec: Whether commands can run on the bot host at all. With
                the native execution tools off there is nothing to restrict,
                so the clause is left out instead of suggesting otherwise.
            sandbox_tools: Execution tools that still work for this sender,
                named so the model reaches for them instead of giving up.

        Returns:
            One line, or an empty string when nothing is restricted.
        """
        parts = []
        if self.tools_allow:
            parts.append("allowed tools: " + ", ".join(self.tools_allow))
        if self.tools_deny:
            parts.append("denied tools: " + ", ".join(self.tools_deny))
        if self.mcp_allow:
            parts.append("allowed MCP servers: " + ", ".join(self.mcp_allow))
        if self.mcp_deny:
            parts.append("denied MCP servers: " + ", ".join(self.mcp_deny))
        if self.native_exec is False and host_exec:
            clause = "no command execution on the bot host"
            if sandbox_tools:
                clause += (
                    " (the sandbox is unaffected: use " + ", ".join(sandbox_tools) + ")"
                )
            parts.append(clause)
        return "; ".join(parts)


DEFAULT_POLICY = PermissionPolicy()


@dataclass
class EventFacts:
    sender_id: str
    group_id: str
    role: str


def _as_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        items = value.replace(",", "\n").splitlines()
    elif isinstance(value, list):
        items = [str(v) for v in value]
    else:
        items = []
    return tuple(i.strip() for i in items if i and i.strip())


def _as_opt_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def condition_matches(condition: str, facts: EventFacts) -> bool:
    cond = condition.strip()
    if not cond:
        return False
    if cond == "*":
        return True
    if cond.startswith("role:"):
        return facts.role == cond[len("role:") :]
    if cond.startswith("p_"):
        return facts.sender_id == cond[2:]
    if cond.startswith("g_"):
        return bool(facts.group_id) and facts.group_id == cond[2:]
    if "/" in cond:
        group, sender = cond.split("/", 1)
        return facts.group_id == group and facts.sender_id == sender
    return False


def resolve_policy(rules: list[dict] | None, facts: EventFacts) -> PermissionPolicy:
    for rule in rules or []:
        if (
            not isinstance(rule, dict)
            or _as_opt_bool(rule.get("enabled", True)) is False
        ):
            continue
        conditions = _as_list(rule.get("match"))
        if not any(condition_matches(c, facts) for c in conditions):
            continue
        return PermissionPolicy(
            rule_name=str(rule.get("name") or ""),
            tools_allow=_as_list(rule.get("tools_allow")),
            tools_deny=_as_list(rule.get("tools_deny")),
            mcp_allow=_as_list(rule.get("mcp_allow")),
            mcp_deny=_as_list(rule.get("mcp_deny")),
            persona_id=str(rule.get("persona_id") or ""),
            model=str(rule.get("model") or ""),
            native_exec=_as_opt_bool(rule.get("native_exec")),
            global_memory=_as_opt_bool(rule.get("global_memory")),
        )
    return DEFAULT_POLICY


def event_facts(event: Any) -> EventFacts:
    return EventFacts(
        sender_id=str(event.get_sender_id() or ""),
        group_id=str(event.get_group_id() or ""),
        role=str(getattr(event, "role", "") or "member"),
    )


def policy_for_event(event: Any, rules: list[dict] | None) -> PermissionPolicy:
    """Resolve and cache the policy on the event."""
    cached = event.get_extra(EVENT_EXTRA_KEY)
    if isinstance(cached, PermissionPolicy):
        return cached
    policy = resolve_policy(rules, event_facts(event))
    event.set_extra(EVENT_EXTRA_KEY, policy)
    return policy


def tool_mcp_server(tool: Any) -> str | None:
    """MCP server name of an AstrBot MCP tool, if it is one."""
    for attr in ("mcp_server_name", "server_name"):
        value = getattr(tool, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


DYNAMIC_PERSONA_CONFIG = "astrbot_plugin_DynamicPersona_config.json"


def rules_from_dynamic_persona(plugin_conf: Any) -> list[dict]:
    """Convert DynamicPersona ``persona_bindings`` into permission rules.

    Match lines and persona carry over; the plugin's per-rule chat provider
    has no Codex equivalent and is dropped (set ``model`` on the rule instead).
    """
    if not isinstance(plugin_conf, dict):
        return []
    rules: list[dict] = []
    for binding in plugin_conf.get("persona_bindings") or []:
        if not isinstance(binding, dict):
            continue
        match = [
            line.strip()
            for line in str(binding.get("match_conditions") or "").splitlines()
            if line.strip()
        ]
        if not match:
            continue
        rules.append(
            {
                "name": str(binding.get("rule_name") or "") or "DynamicPersona",
                "enabled": bool(binding.get("rule_enabled", True))
                and bool(plugin_conf.get("enabled", True)),
                "match": match,
                "persona_id": str(binding.get("persona_id") or ""),
            }
        )
    return rules
