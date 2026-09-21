"""Agent runner that hands orchestration and execution to Codex.

AstrBot keeps message intake, persona, plugin hooks and tools; Codex (driven
in-process through the ``codex_astrbot`` binding) runs the agent loop. Plugin
tools are exposed as Codex dynamic tools — deferred and called from code mode
by default — and executed back in AstrBot with the triggering event.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import re
import shutil
import sys
import time
import typing as T
from pathlib import Path

from astrbot.core import db_helper, logger, sp
from astrbot.core.message.components import Json
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.permission_rules import EVENT_EXTRA_KEY as POLICY_EXTRA_KEY
from astrbot.core.permission_rules import PermissionPolicy
from astrbot.core.provider.entities import LLMResponse, ProviderRequest, TokenUsage

from ...hooks import BaseAgentRunHooks
from ...response import AgentResponseData, AgentStats
from ...run_context import ContextWrapper, TContext
from ..base import AgentResponse, AgentState, BaseAgentRunner
from .constants import (
    CODEX_RUNNER_TYPE,
    CODEX_THREAD_STATE_KEY,
    DEFAULT_SYSTEM_PROMPT,
    NATIVE_EXEC_SESSION_KEY,
)
from .native import (
    ACTIVE_TURNS,
    TERMINAL_EVENTS,
    ActiveTurn,
    CodexEngine,
    JsonObject,
    SessionBusy,
    find_code_mode_host,
    find_codex_exe,
    session_slot,
)
from .tool_bridge import CodexToolBridge
from .usage import FIRST_TOKEN_EVENTS, thread_usage, usage_between

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

_FINAL_PHASES = (None, "final_answer")
# A steered follow-up that reached Codex only as its turn ended is answered by
# a continuation turn carrying this note (B13).
CONTINUE_NOTE = (
    '<request_context name="follow_up">\n'
    "The user sent the message(s) above while you were finishing your previous "
    "reply. Reply to them now.\n"
    "</request_context>"
)
MAX_CONTINUATIONS = 2
# Shown instead of an answer when a chat already has too many turns waiting.
BUSY_NOTE = "我这边还在处理前面的消息，稍后再发一次吧。"
CODE_MODES = ("code_mode", "code_mode_only")


def _data_path() -> Path:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path

    return Path(get_astrbot_data_path())


def _default_cwd(umo: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in umo)
    path = _data_path() / "codex_workspaces" / safe
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def engine_options(cfg: dict) -> JsonObject:
    """Process-wide Codex options derived from the runner config."""
    codex_home = str(cfg.get("codex_home") or "") or str(_data_path() / "codex_home")
    tool_mode = cfg.get("tool_mode") or "code_mode_only"
    # Shipyard mode keeps every file operation inside the sandbox, so Codex
    # never gets its own shell on this host, whatever native_exec_tools says.
    native_exec = bool(cfg.get("native_exec_tools")) and not cfg.get("shipyard_mode")
    config: JsonObject = {
        # Chat-bot defaults: no coding-assistant scaffolding in the prompt.
        "include_permissions_instructions": False,
        "include_environment_context": False,
        "include_apps_instructions": False,
        "include_collaboration_mode_instructions": False,
        "skills.include_instructions": False,
        "project_doc_max_bytes": 0,
        "agents.enabled": False,
        "tools.experimental_request_user_input.enabled": False,
        "features.shell_tool": native_exec,
        "web_search": "live" if cfg.get("web_search") else "disabled",
        "model_tool_mode": tool_mode,
        "features.code_mode.structured_dynamic_tool_results": True,
        "features.code_mode.compact_exec_description": True,
        "features.code_mode.exec_as_function_tool": bool(
            cfg.get("exec_as_function_tool")
        ),
        "approval_policy": cfg.get("approval_policy") or "never",
        "sandbox_mode": cfg.get("sandbox") or "read-only",
        # Codex reports skill invocations (by name), MCP tool calls and thread
        # metadata to chatgpt.com whenever an account is signed in. A chat bot
        # runs other people's conversations, so this is off unless the operator
        # turns it back on through thread_config.
        "analytics.enabled": False,
        # Inert today (the binding builds no OTEL provider, so every metric is
        # a no-op) but the built-in default is a Statsig exporter, so pin it.
        "otel.metrics_exporter": "none",
    }
    if model := cfg.get("model"):
        config["model"] = model
    if provider := cfg.get("model_provider"):
        config["model_provider"] = provider
    if effort := cfg.get("reasoning_effort"):
        config["model_reasoning_effort"] = effort
    config.update(model_provider_overrides(cfg.get("model_providers")))
    config.update(dict(cfg.get("thread_config") or {}))
    options: JsonObject = {"codex_home": codex_home, "config": config}
    if tool_mode in CODE_MODES:
        host = find_code_mode_host(str(cfg.get("code_mode_host") or ""))
        if host:
            options["code_mode_host"] = host
        else:
            logger.warning(
                "codex-code-mode-host not found; set code_mode_host or install Codex CLI. "
                "code_mode_only turns will fail."
            )
    # The executable is what gives Codex a local execution environment, which
    # only native execution needs. Memory consolidation used to need it too;
    # it now maintains its files through the memories extension's file tools,
    # which run in-process.
    if native_exec:
        if exe := find_codex_exe(str(cfg.get("codex_self_exe") or "")):
            options["codex_self_exe"] = exe
        else:
            logger.warning(
                "No codex executable found, so native execution cannot run: "
                "without it Codex has no local execution environment. Chat and "
                "memory consolidation are unaffected. Set codex_self_exe, or "
                "reinstall the binding with CODEX_ASTRBOT_WITH_CODEX=1."
            )
    if native_exec and (cfg.get("approval_policy") or "never") == "never":
        # Every native command asks for approval; AstrBot answers it from the
        # sender's permission rule (native_exec_decision).
        options["approve_every_command"] = True
        config.pop("approval_policy", None)
    return options


def model_provider_overrides(providers: T.Any) -> JsonObject:
    """Map WebUI-managed providers to Codex `model_providers.<id>` overrides."""
    out: JsonObject = {}
    for p in providers if isinstance(providers, list) else []:
        if not isinstance(p, dict):
            continue
        pid = re.sub(r"[^A-Za-z0-9_-]", "_", str(p.get("id") or "").strip())
        base_url = str(p.get("base_url") or "").strip()
        if not pid or not base_url:
            continue
        prefix = f"model_providers.{pid}"
        out[f"{prefix}.name"] = str(p.get("name") or pid)
        out[f"{prefix}.base_url"] = base_url
        out[f"{prefix}.wire_api"] = str(p.get("wire_api") or "responses")
        if key := str(p.get("api_key") or "").strip():
            out[f"{prefix}.experimental_bearer_token"] = key
    return out


def memory_thread_config(cfg: dict, umo: str, event: T.Any) -> JsonObject:
    """Per-thread Codex memory settings (R16–R18).

    Each chat gets its own local store (scope = UMO). Only a private chat whose
    sender's rule grants ``global_memory`` may promote memories to the global
    store; group chats stay local because many people's details mix there.
    Whoever may write globally may also delete memories, so a wrong one can be
    taken back. Both flags are fixed when the thread starts (Codex keeps
    may_write_global sticky-false), so a rule change applies from the next
    thread, which ``/reset`` starts.
    """
    get_extra = getattr(event, "get_extra", None)
    policy = get_extra(POLICY_EXTRA_KEY) if callable(get_extra) else None
    is_private = False
    with contextlib.suppress(Exception):
        is_private = not event.get_group_id()
    trusted = isinstance(policy, PermissionPolicy) and policy.global_memory is True
    return {
        "features.memories": True,
        "memories.dedicated_tools": True,
        "memories.extra_session_sources": ["astrbot"],
        "memories.scope_key": umo,
        "memories.may_write_global": bool(is_private and trusted),
        "memories.may_delete": trusted,
        "memories.auto_consolidate": bool(cfg.get("memory_auto_consolidate", True)),
    }


def approvals_disabled(cfg: dict) -> bool:
    """An explicit non-"never" approval policy with auto_approve off denies
    every approval request, as the setting documents. With the default
    "never" policy and native exec on, approve_every_command is used and the
    permission rules decide instead."""
    policy = str(cfg.get("approval_policy") or "never")
    return policy != "never" and not cfg.get("auto_approve")


def native_exec_decision(
    event: T.Any, session_enabled: bool | None = None
) -> tuple[bool, str]:
    """Approve native execution unless this chat turned it off (K3) or the
    sender's rule sets native_exec: false. Deciding per command keeps the
    thread's tool set, history and prompt cache unchanged when toggled."""
    if session_enabled is False:
        return False, "Native command execution is turned off in this chat."
    get_extra = getattr(event, "get_extra", None)
    policy = get_extra(POLICY_EXTRA_KEY) if callable(get_extra) else None
    if isinstance(policy, PermissionPolicy) and policy.native_exec is False:
        logger.info("Codex native execution denied by rule %r", policy.rule_name)
        return False, "Native command execution is not permitted for this user."
    return True, ""


def system_prompt(cfg: dict) -> str:
    custom = str(cfg.get("base_instructions") or "").strip()
    return custom or DEFAULT_SYSTEM_PROMPT


# Codex saves a generated image under CODEX_HOME, on the host and outside every
# tool the model has -- the message tools refuse CODEX_HOME outright. So the
# image is copied into the chat's workspace and the model is told where, and it
# decides whether to send it or keep working on it. Both the hosted Responses
# item ("ImageGeneration") and the standalone extension item
# ("image_gen.generation") carry the same `saved_path`.
_IMAGE_ITEM_TYPES = ("ImageGeneration",)
_IMAGE_ITEM_KIND = "image_gen.generation"
#: Workspace-relative directory generated images are copied into. The same
#: relative path resolves for send_message_to_user in every runtime: the host
#: workspace is tried first, then the sandbox.
GENERATED_IMAGE_DIR = "generated_images"
#: Where Shipyard Neo mounts the sandbox workspace.
SANDBOX_WORKSPACE = "/workspace"


def generated_image_note(relative_path: str, where: str) -> str:
    """Tells the model where its generated image went and what to do with it."""
    return (
        f"The image was copied to {where}. It has NOT been sent to the user, and "
        "nothing will send it for you. To show it, call send_message_to_user with "
        f'{{"type": "image", "path": "{relative_path}"}}. You can also keep working '
        "with the file where it is."
    )


def generated_image_failure_note(error: str) -> str:
    """Tells the model its generated image could not be made reachable."""
    return (
        f"The image could not be copied into your workspace ({error}), so you "
        "cannot send or use it. Tell the user it could not be delivered."
    )


def request_context(name: str, text: str) -> str:
    """Wraps host text given to the model as input rather than as a tool result."""
    return f'<request_context name="{name}">\n{text}\n</request_context>'


def generated_image_path(item: JsonObject) -> str | None:
    """Path of a completed generated image, if this item is one."""
    is_image = (
        item.get("type") in _IMAGE_ITEM_TYPES or item.get("kind") == _IMAGE_ITEM_KIND
    )
    if not is_image or item.get("status") != "completed":
        return None
    path = item.get("saved_path")
    if not isinstance(path, str) or not path:
        return None
    return path if os.path.isfile(path) else None


def _part_to_input(part: T.Any) -> JsonObject | None:
    data = (
        part.model_dump_for_context()
        if hasattr(part, "model_dump_for_context")
        else part
    )
    if not isinstance(data, dict):
        return None
    if data.get("type") == "text" and data.get("text"):
        return {"type": "text", "text": data["text"], "text_elements": []}
    if data.get("type") == "image_url":
        url = (data.get("image_url") or {}).get("url")
        if url:
            return {"type": "image", "image_url": url}
    return None


def _image_input(ref: str) -> JsonObject:
    if ref.startswith(("http://", "https://", "data:")):
        return {"type": "image", "image_url": ref}
    if ref.startswith("file:///"):
        ref = ref[len("file:///") :]
    return {"type": "local_image", "path": ref}


def build_turn_input(req: ProviderRequest) -> list[JsonObject]:
    """Flatten a ProviderRequest's per-message content into Codex user input."""
    items: list[JsonObject] = []
    for part in [*req.dynamic_user_context_parts, *req.persistent_user_context_parts]:
        if item := _part_to_input(part):
            items.append(item)
    prompt = (req.prompt or "").strip()
    if prompt:
        items.append({"type": "text", "text": prompt, "text_elements": []})
    for part in req.extra_user_content_parts:
        if item := _part_to_input(part):
            items.append(item)
    for ref in req.image_urls:
        items.append(_image_input(ref))
    for ref in req.audio_urls:
        if ref.startswith(("http://", "https://", "data:")):
            items.append({"type": "audio", "audio_url": ref})
        elif os.path.exists(ref):
            items.append({"type": "local_audio", "path": ref})
    if not items:
        items.append({"type": "text", "text": "<empty message>", "text_elements": []})
    return items


def build_additional_context(req: ProviderRequest) -> dict[str, JsonObject]:
    """Standing instructions: re-sent by Codex only when their value changes."""
    ctx: dict[str, JsonObject] = {}
    if req.system_prompt and req.system_prompt.strip():
        ctx["astrbot_system_prompt"] = {
            "value": req.system_prompt.strip(),
            "kind": "application",
        }
    for name, content in req.context_anchors.items():
        ctx[f"astrbot_{name}"] = {"value": content, "kind": "application"}
    return ctx


PERSONA_ANCHORS = ("persona", "persona_examples", "default_persona")


def persona_context(
    req: ProviderRequest, catalog: dict[str, str]
) -> tuple[dict[str, str], dict[str, JsonObject], JsonObject | None]:
    """Cache-friendly personas when one thread serves several (B16).

    With a single persona per thread the anchors are sent as before. Once a
    second persona shows up (per-user persona rules in a group), all personas
    seen in the thread go out as one catalog that only changes when a new one
    appears, and each turn just names the active one, instead of re-sending
    the full persona text every time speakers alternate.

    Returns (updated catalog, additional_context, per-turn input item or None).
    """
    context = build_additional_context(req)
    block = "\n\n".join(
        req.context_anchors[name]
        for name in PERSONA_ANCHORS
        if req.context_anchors.get(name)
    )
    key = hashlib.sha1(block.encode("utf-8")).hexdigest()[:8] if block else ""
    updated = dict(catalog)
    if key:
        updated.setdefault(key, block)
    if len(updated) <= 1:
        return updated, context, None
    for name in PERSONA_ANCHORS:
        context.pop(f"astrbot_{name}", None)
    catalog_text = "\n\n".join(
        f'<persona id="{k}">\n{v}\n</persona>' for k, v in updated.items()
    )
    context["astrbot_personas"] = {
        "value": "Several personas are used in this chat; each user message names "
        "the one to use for that reply.\n\n" + catalog_text,
        "kind": "application",
    }
    active = (
        f'<active_persona id="{key}"/>'
        if key
        else "<active_persona>none: reply as the plain assistant</active_persona>"
    )
    return updated, context, {"type": "text", "text": active, "text_elements": []}


class CodexAgentRunner(BaseAgentRunner[TContext]):
    """In-process Codex agent runner."""

    @override
    async def reset(
        self,
        request: ProviderRequest,
        run_context: ContextWrapper[TContext],
        agent_hooks: BaseAgentRunHooks[TContext],
        provider_config: dict,
        **kwargs: T.Any,
    ) -> None:
        self.req = request
        self._additional_context: dict[str, JsonObject] = {}
        self._active_persona: JsonObject | None = None
        self.run_context = run_context
        self.agent_hooks = agent_hooks
        self.cfg = provider_config
        self.streaming = bool(kwargs.get("streaming", False))
        self.final_llm_resp: LLMResponse | None = None
        self._state = AgentState.IDLE
        self.umo = request.session_id or ""
        tool_mode = self.cfg.get("tool_mode") or "code_mode_only"
        event = getattr(getattr(run_context, "context", None), "event", None)
        get_extra = getattr(event, "get_extra", None)
        policy = get_extra(POLICY_EXTRA_KEY) if callable(get_extra) else None
        self.bridge = CodexToolBridge(
            request.func_tool,
            defer=tool_mode in CODE_MODES,
            policy=policy if isinstance(policy, PermissionPolicy) else None,
        )
        self._engine: CodexEngine | None = None
        self._thread_id: str | None = None
        self._turn_running = False
        self._aborted = False
        self._active: ActiveTurn | None = None
        # Usage of this run, in the shape the stats page and WebChat read.
        self.stats = AgentStats()
        self._usage_before: JsonObject | None = None
        self._stats_recorded = False

    # ------------------------------------------------------------------ public

    @override
    async def step(self) -> T.AsyncGenerator[AgentResponse, None]:
        if self._state == AgentState.IDLE:
            try:
                await self.agent_hooks.on_agent_begin(self.run_context)
            except Exception as e:  # noqa: BLE001
                logger.error("Error in on_agent_begin hook: %s", e, exc_info=True)
        self._transition_state(AgentState.RUNNING)
        try:
            async for resp in self._run_turn():
                yield resp
        except Exception as e:  # noqa: BLE001
            logger.error("Codex runner failed: %s", e, exc_info=True)
            msg = f"Codex 请求失败：{e!s}"
            self._transition_state(AgentState.ERROR)
            self.final_llm_resp = LLMResponse(role="err", completion_text=msg)
            await self._record_stats("error")
            yield AgentResponse(
                type="err", data=AgentResponseData(chain=MessageChain().message(msg))
            )

    @override
    async def step_until_done(
        self, max_step: int = 30
    ) -> T.AsyncGenerator[AgentResponse, None]:
        while not self.done():
            async for resp in self.step():
                yield resp

    @override
    def done(self) -> bool:
        return self._state in (AgentState.DONE, AgentState.ERROR)

    @override
    def get_final_llm_resp(self) -> LLMResponse | None:
        return self.final_llm_resp

    def request_stop(self) -> None:
        self._aborted = True
        if self._active is not None:
            self._active.aborted = True
        if self._turn_running and self._engine and self._thread_id:
            asyncio.ensure_future(self._engine.interrupt(self._thread_id))

    def was_aborted(self) -> bool:
        return self._aborted

    async def close(self) -> None:
        if self._turn_running and self._engine and self._thread_id:
            await self._engine.interrupt(self._thread_id)

    def _message_id(self) -> str | None:
        event = getattr(getattr(self.run_context, "context", None), "event", None)
        message_obj = getattr(event, "message_obj", None)
        mid = getattr(message_obj, "message_id", None)
        return str(mid) if mid else None

    def _sender_id(self) -> str:
        event = getattr(getattr(self.run_context, "context", None), "event", None)
        try:
            return str(event.get_sender_id()) if event is not None else ""
        except Exception:  # noqa: BLE001
            return ""

    # ---------------------------------------------------------------- threads

    def _thread_params(self) -> JsonObject:
        cwd = str(self.cfg.get("cwd") or "") or _default_cwd(self.umo)
        params: JsonObject = {
            "cwd": cwd,
            "base_instructions": system_prompt(self.cfg),
            "dynamic_tools": self.bridge.dynamic_tools(),
            "no_environment": not self.cfg.get("native_exec_tools"),
        }
        if extra := str(self.cfg.get("developer_instructions") or "").strip():
            params["developer_instructions"] = extra
        if self.cfg.get("memory_enabled"):
            event = getattr(getattr(self.run_context, "context", None), "event", None)
            params["config"] = memory_thread_config(self.cfg, self.umo, event)
        return params

    async def _open_thread(self, engine: CodexEngine) -> tuple[str, list | None]:
        """Return the thread id and the tool set to send if it changed."""
        state = await sp.get_async(
            scope="umo", scope_id=self.umo, key=CODEX_THREAD_STATE_KEY, default={}
        )
        state = state if isinstance(state, dict) else {}
        info, started_new = await engine.open_thread(state, self._thread_params())
        catalog = {} if started_new else state.get("personas") or {}
        catalog, self._additional_context, self._active_persona = persona_context(
            self.req, catalog if isinstance(catalog, dict) else {}
        )
        tools_update = None
        if not started_new and state.get("tools_fp") != self.bridge.fingerprint:
            # Replace the tool set in place; history and cache prefix are kept.
            tools_update = self.bridge.dynamic_tools()
            logger.info(
                "AstrBot tool set changed for umo=%s; updating thread tools.", self.umo
            )
        await sp.put_async(
            scope="umo",
            scope_id=self.umo,
            key=CODEX_THREAD_STATE_KEY,
            value={
                "thread_id": info["thread_id"],
                "rollout_path": info.get("rollout_path") or state.get("rollout_path"),
                "tools_fp": self.bridge.fingerprint,
                "personas": catalog,
            },
        )
        return info["thread_id"], tools_update

    # ------------------------------------------------------------------- turn

    async def _handle_tool_call(self, msg: JsonObject) -> JsonObject:
        return await self.bridge.call(msg, self.run_context, self.agent_hooks)

    async def _handle_approval(self, kind: str, msg: JsonObject) -> tuple[bool, str]:
        """Native exec / patch approvals follow the sender's permission rule (B15)."""
        if approvals_disabled(self.cfg):
            return False, "Approval requests are disabled (auto_approve is off)."
        session_setting = await sp.get_async(
            scope="umo", scope_id=self.umo, key=NATIVE_EXEC_SESSION_KEY, default=None
        )
        return native_exec_decision(
            getattr(getattr(self.run_context, "context", None), "event", None),
            session_enabled=session_setting
            if isinstance(session_setting, bool)
            else None,
        )

    def _turn_request(self, tools_update: list | None) -> JsonObject:
        turn_input = build_turn_input(self.req)
        if self._active_persona is not None:
            turn_input.insert(0, self._active_persona)
        request: JsonObject = {
            "input": turn_input,
            "mode": "start_or_steer",
            "additional_context": self._additional_context,
        }
        if tools_update is not None:
            request["dynamic_tools"] = tools_update
        if self.req.model:
            request["model"] = self.req.model
        return request

    def _computer_runtime(self) -> str:
        """Execution environment this chat's tools run in, as codex_request sets it."""
        if self.cfg.get("shipyard_mode"):
            return "sandbox"
        try:
            ctx = self.run_context.context.context  # type: ignore[attr-defined]
            settings = ctx.get_config(umo=self.umo).get("provider_settings") or {}
        except Exception:  # noqa: BLE001 - no config means no execution runtime
            return "none"
        return str(settings.get("computer_use_runtime") or "none")

    async def _place_generated_image(self, path: str) -> str:
        """Copies a generated image into the chat's workspace.

        The sandbox workspace when tools run there, the host workspace
        otherwise -- the same place a relative path given to
        send_message_to_user resolves to.

        Returns:
            The note to give the model: where the image is, or why it is not
            anywhere it can reach.
        """
        from astrbot.core.computer.computer_client import get_booter
        from astrbot.core.tools.computer_tools.util import (
            workspace_root,
            workspace_root_for_context,
        )

        relative = f"{GENERATED_IMAGE_DIR}/{os.path.basename(path)}"
        runtime = self._computer_runtime()
        try:
            if runtime == "sandbox":
                ctx = self.run_context.context.context  # type: ignore[attr-defined]
                booter = await get_booter(ctx, self.umo)
                await booter.upload_file(path, relative)
                where = (
                    f"`{relative}` in your sandbox workspace "
                    f"(`{SANDBOX_WORKSPACE}/{relative}`)"
                )
            else:
                root = (
                    await workspace_root_for_context(self.run_context)  # type: ignore[arg-type]
                    if runtime == "local"
                    else workspace_root(self.umo)
                )
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(shutil.copyfile, path, target)
                where = f"`{relative}` in your workspace"
        except Exception as e:  # noqa: BLE001 - sandbox or filesystem failure
            logger.warning("Could not place generated image %s: %s", path, e)
            return generated_image_failure_note(str(e))
        logger.info("Generated image placed at %s (%s runtime)", relative, runtime)
        return generated_image_note(relative, where)

    async def _on_saved_image(self, call_id: str, saved_path: str) -> str | None:
        """Places an image Codex just saved; the result is the tool's output."""
        if not os.path.isfile(saved_path):
            return None
        self._hooked_images.add(saved_path)
        return await self._place_generated_image(saved_path)

    async def _steer_note(
        self, engine: CodexEngine, thread_id: str, active: ActiveTurn | None, note: str
    ) -> bool:
        """Adds a note to the running turn; False when the turn is already over."""
        if active is None or not active.turn_id:
            return False
        try:
            result = await engine.submit_turn(
                thread_id,
                {
                    "input": [{"type": "text", "text": note, "text_elements": []}],
                    "mode": "steer",
                    "expected_turn_id": active.turn_id,
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not steer a note into %s: %s", thread_id, e)
            return False
        return result.get("status") == "steered"

    async def _run_turn(self) -> T.AsyncGenerator[AgentResponse, None]:
        engine = await CodexEngine.get(engine_options(self.cfg))
        self._engine = engine
        show_commentary = bool(self.cfg.get("show_commentary"))
        timeout = float(self.cfg.get("turn_timeout") or 600)
        max_queued = int(self.cfg.get("max_queued_turns") or 0)

        try:
            async with session_slot(engine, self.umo, max_queued):
                # Opening the thread belongs in the critical section: it
                # is what decides the thread id, so two senders arriving
                # together on a new chat would otherwise each make one.
                thread_id, tools_update = await self._open_thread(engine)
                self._thread_id = thread_id
                # Images Codex saves during this turn are placed through the
                # engine's hook, so the model reads where the image went in the
                # tool result itself. The event-stream path below only covers
                # an image the hook did not handle.
                self._hooked_images = set()
                engine.saved_image_handlers[thread_id] = self._on_saved_image
                pump = engine.pump(thread_id)
                queue = pump.open_turn(self._handle_tool_call, self._handle_approval)
                phases: dict[str, str | None] = {}
                final_texts: list[str] = []
                commentary: list[str] = []
                reasoning: list[str] = []
                # Generated images already handled, and notes about them that
                # could not be steered in because the turn was ending.
                placed_images: set[str] = set()
                pending_notes: list[str] = []
                usage: JsonObject | None = None
                error_msg: str | None = None
                end = "task_complete"
                started = time.monotonic()
                active: ActiveTurn | None = None
                event_seq = last_user_seq = last_agent_seq = 0
                codex_ttft = 0.0
                continuations = 0
                try:
                    # Registered before the submit, not after: a same-sender
                    # follow-up arriving during that round trip should wait for
                    # this turn and be steered into it, not become a second one.
                    active = ActiveTurn(
                        engine,
                        thread_id,
                        "",
                        self._sender_id(),
                        message_id=self._message_id(),
                    )
                    self._active = active
                    ACTIVE_TURNS[self.umo] = active
                    self._usage_before = await thread_usage(engine, thread_id)
                    self.stats.start_time = time.time()
                    try:
                        sub = await engine.submit_turn(
                            thread_id, self._turn_request(tools_update)
                        )
                        if sub.get("status") == "not_submitted":
                            raise RuntimeError(
                                f"Codex did not accept the turn: {sub.get('reason')}"
                            )
                        active.turn_id = str(sub.get("turn_id") or "")
                    except BaseException:
                        active.aborted = True
                        raise
                    finally:
                        # Release a follow-up waiting on the turn id, including
                        # when the submit failed.
                        active.ready.set()
                    self._turn_running = True
                    while True:
                        remaining = timeout - (time.monotonic() - started)
                        if remaining <= 0:
                            await engine.interrupt(thread_id)
                            error_msg = f"Codex turn timed out after {timeout:.0f}s"
                            break
                        try:
                            msg = await asyncio.wait_for(queue.get(), remaining)
                        except asyncio.TimeoutError:
                            continue
                        kind = msg.get("type")
                        event_seq += 1
                        if (
                            not self.stats.time_to_first_token
                            and kind in FIRST_TOKEN_EVENTS
                        ):
                            # Floored: 0 means "not measured" to the stats.
                            self.stats.time_to_first_token = max(
                                time.time() - self.stats.start_time, 0.001
                            )
                        if kind == "user_message":
                            last_user_seq = event_seq
                        if kind == "item_started":
                            item = msg.get("item") or {}
                            if item.get("type") == "AgentMessage":
                                phases[item.get("id", "")] = item.get("phase")
                        elif kind == "item_completed":
                            path = generated_image_path(msg.get("item") or {})
                            if (
                                path
                                and path not in placed_images
                                and path not in self._hooked_images
                            ):
                                placed_images.add(path)
                                note = request_context(
                                    "generated_image",
                                    await self._place_generated_image(path),
                                )
                                if not await self._steer_note(
                                    engine, thread_id, active, note
                                ):
                                    pending_notes.append(note)
                        elif kind == "agent_message_content_delta":
                            phase = phases.get(msg.get("item_id", ""))
                            if self.streaming and (
                                phase in _FINAL_PHASES or show_commentary
                            ):
                                yield AgentResponse(
                                    type="streaming_delta",
                                    data=AgentResponseData(
                                        chain=MessageChain().message(
                                            msg.get("delta", "")
                                        )
                                    ),
                                )
                        elif kind == "agent_message":
                            last_agent_seq = event_seq
                            text = msg.get("message") or ""
                            if msg.get("phase") in _FINAL_PHASES:
                                final_texts.append(text)
                            else:
                                commentary.append(text)
                                if self.streaming and show_commentary:
                                    yield AgentResponse(
                                        type="streaming_delta",
                                        data=AgentResponseData(
                                            chain=MessageChain().message("\n\n")
                                        ),
                                    )
                        elif kind == "agent_reasoning":
                            reasoning.append(msg.get("text") or "")
                        elif kind == "token_count":
                            info = msg.get("info") or {}
                            usage = info.get("last_token_usage") or usage
                            # The request that last filled the context.
                            self.stats.current_context_tokens = int(
                                (usage or {}).get("input_tokens") or 0
                            )
                        elif kind == "error":
                            error_msg = msg.get("message") or str(msg)
                        elif kind in TERMINAL_EVENTS:
                            end = kind
                            # Codex's own measure, from the turn's first model
                            # request; continuations do not replace it.
                            ttft_ms = msg.get("time_to_first_token_ms")
                            if (
                                not codex_ttft
                                and isinstance(ttft_ms, int | float)
                                and ttft_ms > 0
                            ):
                                codex_ttft = ttft_ms / 1000
                                self.stats.time_to_first_token = codex_ttft
                            # A follow-up steered in as the turn ended was recorded
                            # but never answered (B13): answer it in one more turn.
                            unanswered = (
                                active.steered > 0 and last_user_seq > last_agent_seq
                            )
                            # An image note that missed the turn gets the same
                            # treatment: without it the model never learns
                            # where its image went.
                            notes = [CONTINUE_NOTE] if unanswered else []
                            notes += pending_notes
                            pending_notes.clear()
                            if (
                                kind != "turn_aborted"
                                and notes
                                and not self._aborted
                                and continuations < MAX_CONTINUATIONS
                            ):
                                continuations += 1
                                again = await engine.submit_turn(
                                    thread_id,
                                    {
                                        "input": [
                                            {
                                                "type": "text",
                                                "text": note,
                                                "text_elements": [],
                                            }
                                            for note in notes
                                        ],
                                        "mode": "start_if_idle",
                                    },
                                )
                                if again.get("status") == "started":
                                    active.turn_id = str(again.get("turn_id") or "")
                                    continue
                            break
                        elif kind == "_pump_closed":
                            raise RuntimeError(
                                msg.get("message") or "Codex thread closed"
                            )
                finally:
                    self._turn_running = False
                    if (
                        engine.saved_image_handlers.get(thread_id)
                        == self._on_saved_image
                    ):
                        engine.saved_image_handlers.pop(thread_id, None)
                    if active is not None and ACTIVE_TURNS.get(self.umo) is active:
                        ACTIVE_TURNS.pop(self.umo, None)
                    pump.close_turn()

        except SessionBusy as busy:
            logger.info(
                "Codex session %s is busy; %d turns already queued.",
                self.umo,
                busy.waiting,
            )
            chain = MessageChain().message(BUSY_NOTE)
            self.final_llm_resp = LLMResponse(role="assistant", result_chain=chain)
            self._transition_state(AgentState.DONE)
            yield AgentResponse(type="llm_result", data=AgentResponseData(chain=chain))
            return

        text = "\n\n".join(t for t in final_texts if t.strip())
        if show_commentary and commentary:
            text = "\n\n".join([*commentary, text]) if text else "\n\n".join(commentary)
        if not text and (commentary or final_texts):
            text = (commentary or final_texts)[-1]
        if error_msg and not text:
            raise RuntimeError(error_msg)
        if end == "turn_aborted" and not text:
            text = "（已中断）" if self._aborted else ""

        chain = MessageChain().message(text)
        self.final_llm_resp = LLMResponse(
            role="assistant",
            result_chain=chain,
            reasoning_content="\n".join(r for r in reasoning if r) or None,
            usage=_token_usage(usage),
        )
        self._transition_state(AgentState.DONE)
        await self._record_stats("aborted" if end == "turn_aborted" else "completed")
        await self._sync_history(text)
        yield AgentResponse(
            type="agent_stats",
            data=AgentResponseData(
                chain=MessageChain(
                    type="agent_stats", chain=[Json(data=self.stats.to_dict())]
                )
            ),
        )
        try:
            await self.agent_hooks.on_agent_done(self.run_context, self.final_llm_resp)
        except Exception as e:  # noqa: BLE001
            logger.error("Error in on_agent_done hook: %s", e, exc_info=True)
        yield AgentResponse(type="llm_result", data=AgentResponseData(chain=chain))

    async def _record_stats(self, status: str) -> None:
        """Closes this run's stats and stores them for the stats page.

        Once per run, and never raising: stats must not break a reply. Rows are
        written with agent type ``codex``, which is what the stats page reads.
        """
        if self._stats_recorded or not self.stats.start_time:
            return
        self._stats_recorded = True
        self.stats.end_time = time.time()
        after: JsonObject | None = None
        if self._engine is not None and self._thread_id:
            after = await thread_usage(self._engine, self._thread_id)
        spent = usage_between(self._usage_before, after)
        if spent is not None:
            self.stats.token_usage = spent
        elif self.final_llm_resp and self.final_llm_resp.usage:
            # Older binding: the last request's usage is the best there is.
            self.stats.token_usage = self.final_llm_resp.usage
        model = (after or {}).get("model") or self.req.model or self.cfg.get("model")
        provider = (after or {}).get("model_provider") or self.cfg.get("model_provider")
        try:
            conv = self.req.conversation
            await db_helper.insert_provider_stat(
                umo=self.umo,
                conversation_id=conv.cid if conv else None,
                provider_id=str(provider or CODEX_RUNNER_TYPE),
                provider_model=str(model) if model else None,
                status=status,
                stats=self.stats.to_dict(),
                agent_type=CODEX_RUNNER_TYPE,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Persist codex stats failed: %s", e, exc_info=True)

    async def _sync_history(self, text: str) -> None:
        """Mirror the exchange into AstrBot's conversation for the WebUI."""
        conv = self.req.conversation
        if not conv or not self.cfg.get("sync_history", True):
            return
        try:
            ctx = self.run_context.context.context  # type: ignore[attr-defined]
            steered = self._active.steered_texts if self._active else []
            user_text = chr(10).join(t for t in [self.req.prompt or "", *steered] if t)
            await ctx.conversation_manager.add_message_pair(
                conv.cid,
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": text},
            )
            # What the conversation list shows as the context size.
            if self.stats.current_context_tokens:
                await ctx.conversation_manager.update_conversation(
                    self.umo,
                    conv.cid,
                    token_usage=self.stats.current_context_tokens,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to mirror codex exchange into history: %s", e)


def _token_usage(usage: JsonObject | None) -> TokenUsage | None:
    if not isinstance(usage, dict):
        return None
    try:
        cached = int(usage.get("cached_input_tokens") or 0)
        return TokenUsage(
            input_other=int(usage.get("input_tokens") or 0) - cached,
            input_cached=cached,
            output=int(usage.get("output_tokens") or 0),
        )
    except Exception:  # noqa: BLE001
        return None
