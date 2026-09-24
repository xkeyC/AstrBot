"""Tools of the voice agent: what an ordinary member of its chat may use.

The voice agent runs in its own Codex thread, but gets the tools, execution
environment and approvals an ordinary (non-admin) member of the paired chat
would get there (e.g. a Mumble server group, a caller's private chat). A voice
conversation may mix speakers, so no one's own permissions apply; the member
rule does, whoever is talking.

Tool calls run against a synthetic event of that chat, the way scheduled
tasks do (see ``codex/wake.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astrbot import logger

VOICE_SENDER_ID = "voice"


@dataclass
class VoiceTools:
    """What a voice agent thread needs to offer and run tools."""

    dynamic_tools: list[dict]
    native_exec: bool
    tool_handler: Any
    approval_handler: Any


async def voice_agent_tools(umo: str, runner_cfg: dict) -> VoiceTools | None:
    """Builds the member tool set of ``umo`` for the voice agent.

    Args:
        umo: Unified message origin of the paired chat.
        runner_cfg: The Codex runner configuration.

    Returns:
        The tools and handlers, or None when the core context is not up (then
        the voice agent runs without tools).
    """
    from astrbot.core import sp
    from astrbot.core.agent.runners.codex.codex_agent_runner import (
        approvals_disabled,
        native_exec_decision,
    )
    from astrbot.core.agent.runners.codex.constants import NATIVE_EXEC_SESSION_KEY
    from astrbot.core.agent.runners.codex.tool_bridge import CodexToolBridge
    from astrbot.core.astr_agent_context import AgentContextWrapper, AstrAgentContext
    from astrbot.core.astr_agent_hooks import MAIN_AGENT_HOOKS
    from astrbot.core.cron.events import CronMessageEvent
    from astrbot.core.permission_rules import EVENT_EXTRA_KEY, PermissionPolicy
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.codex_request import (
        prepare_codex_request,
    )
    from astrbot.core.platform.message_session import MessageSession
    from astrbot.core.provider.entities import ProviderRequest
    from astrbot.core.star.context import current_context

    ctx = current_context()
    if ctx is None:
        logger.warning("Voice: no core context, the voice agent gets no tools")
        return None
    session = MessageSession.from_str(umo)
    event = CronMessageEvent(
        context=ctx,
        session=session,
        message="(voice conversation)",
        sender_id=VOICE_SENDER_ID,
        sender_name="Voice",
        message_type=session.message_type,
    )
    event.message_obj.sender.user_id = VOICE_SENDER_ID
    event.role = "member"
    cfg = ctx.get_config(umo=umo)
    req = ProviderRequest()
    req.session_id = umo
    req.prompt = ""
    await prepare_codex_request(event, req, ctx, cfg, runner_cfg)
    policy = event.get_extra(EVENT_EXTRA_KEY)
    # The voice thread runs in direct tool mode: every allowed tool is sent.
    bridge = CodexToolBridge(
        req.func_tool,
        defer=False,
        policy=policy if isinstance(policy, PermissionPolicy) else None,
    )
    run_context = AgentContextWrapper(
        context=AstrAgentContext(context=ctx, event=event),
        tool_call_timeout=int(runner_cfg.get("tool_call_timeout") or 120),
    )

    async def tool_handler(msg: dict) -> dict:
        return await bridge.call(msg, run_context, MAIN_AGENT_HOOKS)

    async def approval_handler(kind: str, msg: dict) -> tuple[bool, str]:
        if approvals_disabled(runner_cfg):
            return False, "Approval requests are disabled (auto_approve is off)."
        session_setting = await sp.get_async(
            scope="umo", scope_id=umo, key=NATIVE_EXEC_SESSION_KEY, default=None
        )
        return native_exec_decision(
            event,
            session_enabled=session_setting
            if isinstance(session_setting, bool)
            else None,
        )

    return VoiceTools(
        dynamic_tools=bridge.dynamic_tools(),
        native_exec=bool(runner_cfg.get("native_exec_tools")),
        tool_handler=tool_handler,
        approval_handler=approval_handler,
    )
