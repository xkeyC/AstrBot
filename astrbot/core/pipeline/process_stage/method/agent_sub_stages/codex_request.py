"""Build a full ProviderRequest for the Codex runner.

Third-party runners normally get a bare prompt. Codex runs AstrBot's plugin
tools itself, so it gets what the local agent would: conversation, persona,
message metadata, quoted messages, knowledge base results and the per-request
tool set. Only the provider-bound steps (model selection, context compression,
computer-use tools) are left out; Codex owns those.
"""

from __future__ import annotations

import os

from astrbot.core import logger
from astrbot.core.agent.message import TextPart
from astrbot.core.agent.tool import ToolSet
from astrbot.core.astr_main_agent import (
    MainAgentBuildConfig,
    _apply_kb,
    _apply_llm_safety_mode,
    _decorate_llm_request,
    _filter_tools_by_persona_scope,
    _get_session_conv,
    _plugin_tool_fix,
    _proactive_cron_job_tools,
)
from astrbot.core.message.components import File, Image, Record, Reply
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.message_type import MessageType
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.star.context import Context
from astrbot.core.tools.message_tools import (
    GetGroupMessageHistoryTool,
    SendMessageToUserTool,
)


def _build_config(
    astrbot_config: dict, runner_config: dict, plugin_context: Context
) -> MainAgentBuildConfig:
    settings = dict(astrbot_config.get("provider_settings", {}))
    # Codex reads images natively; do not caption them with another provider.
    settings["default_image_caption_provider_id"] = ""
    proactive_cfg = settings.get("proactive_capability", {}) or {}
    return MainAgentBuildConfig(
        tool_call_timeout=int(runner_config.get("tool_call_timeout") or 120),
        provider_settings=settings,
        kb_agentic_mode=astrbot_config.get("kb_agentic_mode", False),
        llm_safety_mode=bool(runner_config.get("safety_mode", False)),
        add_cron_tools=proactive_cfg.get("add_cron_tools", True),
        timezone=plugin_context.get_config().get("timezone"),
        max_quoted_fallback_images=settings.get("max_quoted_fallback_images", 20),
    )


async def _collect_media(event: AstrMessageEvent, req: ProviderRequest) -> None:
    """Replace the base64 images of the generic path with local file paths."""
    req.image_urls = []
    req.audio_urls = []
    for comp in event.message_obj.message:
        chains = [comp]
        if isinstance(comp, Reply) and comp.chain:
            chains = list(comp.chain)
        for c in chains:
            try:
                if isinstance(c, Image):
                    req.image_urls.append(await c.convert_to_file_path())
                elif isinstance(c, Record):
                    req.audio_urls.append(await c.convert_to_file_path())
                elif isinstance(c, File):
                    path = await c.get_file()
                    name = c.name or os.path.basename(path)
                    req.extra_user_content_parts.append(
                        TextPart(text=f"[File Attachment: name {name}, path {path}]")
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("Codex: failed to resolve attachment %s: %s", c, e)


async def prepare_codex_request(
    event: AstrMessageEvent,
    req: ProviderRequest,
    plugin_context: Context,
    astrbot_config: dict,
    runner_config: dict,
) -> None:
    config = _build_config(astrbot_config, runner_config, plugin_context)
    await _collect_media(event, req)

    req.conversation = await _get_session_conv(event, plugin_context)
    await _decorate_llm_request(event, req, plugin_context, config, provider=None)
    await _apply_kb(event, req, plugin_context, config)
    _plugin_tool_fix(event, req)

    if config.llm_safety_mode:
        _apply_llm_safety_mode(config, req)
    if config.add_cron_tools:
        _proactive_cron_job_tools(req, plugin_context)

    tmgr = plugin_context.get_llm_tool_manager()
    if req.func_tool is None:
        req.func_tool = ToolSet()
    if event.platform_meta.support_proactive_message:
        req.func_tool.add_tool(tmgr.get_builtin_tool(SendMessageToUserTool))
    ltm = plugin_context.get_config(umo=event.unified_msg_origin).get(
        "provider_ltm_settings", {}
    )
    if event.get_message_type() == MessageType.GROUP_MESSAGE and ltm.get(
        "group_message_history_enable", False
    ):
        req.func_tool.add_tool(tmgr.get_builtin_tool(GetGroupMessageHistoryTool))

    _filter_tools_by_persona_scope(event, req)
    req.context_anchors_complete = True
    if not req.prompt and (req.image_urls or req.extra_user_content_parts):
        req.prompt = "<attachment>"
