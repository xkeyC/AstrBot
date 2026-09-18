"""Scheduled-task wake-ups for the Codex runner (K12).

A cron run is delivered into the chat's own Codex thread (sharing its
context and prompt cache) as one message: a quote of the user's original
request plus the task to execute now. The wording makes clear this is one run
of an existing task, so the agent performs it instead of scheduling another.
"""

from __future__ import annotations

from typing import Any

from astrbot.core import logger
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.provider.entities import ProviderRequest


def build_cron_prompt(cron_job: dict, payload: dict) -> str:
    name = str(cron_job.get("name") or cron_job.get("id") or "task")
    started = str(cron_job.get("run_started_at") or "")
    note = str(
        payload.get("note") or cron_job.get("note") or cron_job.get("description") or ""
    )
    origin = str(payload.get("origin_message") or "").strip()
    lines = [
        f'<scheduled_task name="{name}" triggered_at="{started}">',
        "This is one scheduled run of a task the user set up earlier. Carry out the "
        "task now. Do not create, change or cancel scheduled tasks unless the task "
        "itself asks for it. Your final reply is sent to the chat; do not greet.",
    ]
    if origin:
        quoted = "\n".join(f"> {line}" for line in origin.splitlines())
        lines += ["User's original request (when the task was created):", quoted]
    lines += ["Task to execute:", note, "</scheduled_task>"]
    return "\n".join(lines)


async def run_codex_cron_job(
    ctx: Any,
    *,
    message: str,
    session_str: str,
    extras: dict,
    delivery_session_str: str = "",
) -> None:
    from astrbot.core.cron.events import CronMessageEvent

    try:
        session = MessageSession.from_str(session_str)
    except Exception as e:  # noqa: BLE001
        logger.error("Invalid session for cron job: %s", e)
        return

    extras = extras or {}
    payload = extras.get("cron_payload", {}) or {}
    cron_job = extras.get("cron_job", {}) or {}
    event = CronMessageEvent(
        context=ctx,
        session=session,
        message=message,
        extras=extras,
        message_type=session.message_type,
    )
    cfg = ctx.get_config(umo=event.unified_msg_origin)
    # Run with the permissions of the user who created the task.
    if sender_id := str(payload.get("sender_id") or ""):
        event.message_obj.sender.user_id = sender_id
    admin_ids = [str(a) for a in cfg.get("admins_id", [])]
    event.role = (
        "admin"
        if payload.get("origin") == "api" or sender_id in admin_ids
        else "member"
    )

    ok = await run_in_session_thread(
        ctx, event, cfg, build_cron_prompt(cron_job, payload), delivery_session_str
    )
    if not ok:
        logger.warning("Codex cron job %s produced no reply", cron_job.get("id"))


async def run_in_session_thread(
    ctx: Any, event: Any, cfg: dict, prompt: str, delivery_session_str: str
) -> bool:
    """Run one Codex turn with ``prompt`` in the event session's thread."""
    from astrbot.core.agent.runners.codex.codex_agent_runner import CodexAgentRunner
    from astrbot.core.astr_agent_context import AgentContextWrapper, AstrAgentContext
    from astrbot.core.astr_agent_hooks import MAIN_AGENT_HOOKS
    from astrbot.core.config.agent_runner import normalize_agent_runner
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.codex_request import (
        prepare_codex_request,
    )

    runner_cfg = normalize_agent_runner(cfg.get("agent_runner"))["config"]
    req = ProviderRequest()
    req.session_id = event.unified_msg_origin
    req.prompt = prompt
    await prepare_codex_request(event, req, ctx, cfg, runner_cfg)
    req.prompt = prompt

    runner = CodexAgentRunner()
    await runner.reset(
        request=req,
        run_context=AgentContextWrapper(
            context=AstrAgentContext(context=ctx, event=event),
            tool_call_timeout=int(runner_cfg.get("tool_call_timeout") or 120),
        ),
        agent_hooks=MAIN_AGENT_HOOKS,
        provider_config=runner_cfg,
        streaming=False,
    )
    async for _ in runner.step_until_done():
        pass
    resp = runner.get_final_llm_resp()
    if resp is None or resp.role != "assistant":
        return False
    text = (resp.completion_text or "").strip()
    if text and delivery_session_str:
        await ctx.send_message(delivery_session_str, MessageChain().message(text))
    return True


def build_background_prompt(task_result: dict, original_message: str) -> str:
    lines = [
        f'<background_task_result tool="{task_result.get("tool_name", "")}" '
        f'task_id="{task_result.get("task_id", "")}">',
        "A background task you started earlier has finished. Use its result to "
        "continue what the user asked. Your final reply is sent to the chat; if "
        "nothing needs to be said, reply with an empty message.",
    ]
    if original_message.strip():
        quoted = "\n".join(f"> {line}" for line in original_message.splitlines())
        lines += ["The request it belongs to:", quoted]
    lines += [
        "Result:",
        str(task_result.get("result") or ""),
        "</background_task_result>",
    ]
    return "\n".join(lines)


async def run_codex_background_wake(
    ctx: Any, origin_event: Any, task_result: dict
) -> None:
    from astrbot.core.cron.events import CronMessageEvent

    session = MessageSession.from_str(origin_event.unified_msg_origin)
    event = CronMessageEvent(
        context=ctx,
        session=session,
        message="background task result",
        extras={"background_task_result": task_result},
        message_type=session.message_type,
    )
    event.message_obj.sender.user_id = str(origin_event.get_sender_id() or "")
    event.role = origin_event.role
    cfg = ctx.get_config(umo=origin_event.unified_msg_origin) or {}
    prompt = build_background_prompt(task_result, origin_event.message_str or "")
    await run_in_session_thread(
        ctx, event, cfg, prompt, origin_event.unified_msg_origin
    )
