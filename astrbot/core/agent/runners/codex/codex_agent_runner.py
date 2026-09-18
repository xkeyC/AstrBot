"""Agent runner that hands orchestration and execution to Codex.

AstrBot keeps message intake, persona, plugin hooks and tools; Codex (driven
in-process through the ``codex_astrbot`` binding) runs the agent loop. Plugin
tools are exposed as Codex dynamic tools — deferred and called from code mode
by default — and executed back in AstrBot with the triggering event.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import typing as T
from pathlib import Path

from astrbot.core import logger, sp
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse, ProviderRequest, TokenUsage

from ...hooks import BaseAgentRunHooks
from ...response import AgentResponseData
from ...run_context import ContextWrapper, TContext
from ..base import AgentResponse, AgentState, BaseAgentRunner
from .constants import CODEX_THREAD_STATE_KEY, DEFAULT_SYSTEM_PROMPT
from .native import (
    ACTIVE_TURNS,
    TERMINAL_EVENTS,
    ActiveTurn,
    CodexEngine,
    JsonObject,
    find_code_mode_host,
)
from .tool_bridge import CodexToolBridge

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
    native_exec = bool(cfg.get("native_exec_tools"))
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
    if native_exec and (exe := cfg.get("codex_self_exe")):
        options["codex_self_exe"] = exe
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


def system_prompt(cfg: dict) -> str:
    custom = str(cfg.get("base_instructions") or "").strip()
    return custom or DEFAULT_SYSTEM_PROMPT


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
        self.run_context = run_context
        self.agent_hooks = agent_hooks
        self.cfg = provider_config
        self.streaming = bool(kwargs.get("streaming", False))
        self.final_llm_resp: LLMResponse | None = None
        self._state = AgentState.IDLE
        self.umo = request.session_id or ""
        tool_mode = self.cfg.get("tool_mode") or "code_mode_only"
        self.bridge = CodexToolBridge(request.func_tool, defer=tool_mode in CODE_MODES)
        self._engine: CodexEngine | None = None
        self._thread_id: str | None = None
        self._turn_running = False
        self._aborted = False
        self._active: ActiveTurn | None = None

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
        return params

    async def _open_thread(self, engine: CodexEngine) -> tuple[str, list | None]:
        """Return the thread id and the tool set to send if it changed."""
        state = await sp.get_async(
            scope="umo", scope_id=self.umo, key=CODEX_THREAD_STATE_KEY, default={}
        )
        state = state if isinstance(state, dict) else {}
        info, started_new = await engine.open_thread(state, self._thread_params())
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
            },
        )
        return info["thread_id"], tools_update

    # ------------------------------------------------------------------- turn

    async def _handle_tool_call(self, msg: JsonObject) -> JsonObject:
        return await self.bridge.call(msg, self.run_context, self.agent_hooks)

    def _turn_request(self, tools_update: list | None) -> JsonObject:
        request: JsonObject = {
            "input": build_turn_input(self.req),
            "mode": "start_or_steer",
            "additional_context": build_additional_context(self.req),
        }
        if tools_update is not None:
            request["dynamic_tools"] = tools_update
        if self.req.model:
            request["model"] = self.req.model
        return request

    async def _run_turn(self) -> T.AsyncGenerator[AgentResponse, None]:
        engine = await CodexEngine.get(engine_options(self.cfg))
        self._engine = engine
        thread_id, tools_update = await self._open_thread(engine)
        self._thread_id = thread_id
        show_commentary = bool(self.cfg.get("show_commentary"))
        timeout = float(self.cfg.get("turn_timeout") or 600)

        async with engine.lock_for(thread_id):
            pump = engine.pump(thread_id)
            queue = pump.open_turn(self._handle_tool_call)
            phases: dict[str, str | None] = {}
            final_texts: list[str] = []
            commentary: list[str] = []
            reasoning: list[str] = []
            usage: JsonObject | None = None
            error_msg: str | None = None
            end = "task_complete"
            started = time.monotonic()
            active: ActiveTurn | None = None
            event_seq = last_user_seq = last_agent_seq = 0
            continuations = 0
            try:
                sub = await engine.submit_turn(
                    thread_id, self._turn_request(tools_update)
                )
                if sub.get("status") == "not_submitted":
                    raise RuntimeError(
                        f"Codex did not accept the turn: {sub.get('reason')}"
                    )
                self._turn_running = True
                active = ActiveTurn(
                    engine,
                    thread_id,
                    str(sub.get("turn_id") or ""),
                    self._sender_id(),
                    message_id=self._message_id(),
                )
                self._active = active
                ACTIVE_TURNS[self.umo] = active
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
                    if kind == "user_message":
                        last_user_seq = event_seq
                    if kind == "item_started":
                        item = msg.get("item") or {}
                        if item.get("type") == "AgentMessage":
                            phases[item.get("id", "")] = item.get("phase")
                    elif kind == "agent_message_content_delta":
                        phase = phases.get(msg.get("item_id", ""))
                        if self.streaming and (
                            phase in _FINAL_PHASES or show_commentary
                        ):
                            yield AgentResponse(
                                type="streaming_delta",
                                data=AgentResponseData(
                                    chain=MessageChain().message(msg.get("delta", ""))
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
                        usage = (msg.get("info") or {}).get("last_token_usage") or usage
                    elif kind == "error":
                        error_msg = msg.get("message") or str(msg)
                    elif kind in TERMINAL_EVENTS:
                        end = kind
                        # A follow-up steered in as the turn ended was recorded
                        # but never answered (B13): answer it in one more turn.
                        unanswered = (
                            active.steered > 0 and last_user_seq > last_agent_seq
                        )
                        if (
                            kind != "turn_aborted"
                            and unanswered
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
                                            "text": CONTINUE_NOTE,
                                            "text_elements": [],
                                        }
                                    ],
                                    "mode": "start_if_idle",
                                },
                            )
                            if again.get("status") == "started":
                                active.turn_id = str(again.get("turn_id") or "")
                                continue
                        break
                    elif kind == "_pump_closed":
                        raise RuntimeError(msg.get("message") or "Codex thread closed")
            finally:
                self._turn_running = False
                if active is not None and ACTIVE_TURNS.get(self.umo) is active:
                    ACTIVE_TURNS.pop(self.umo, None)
                pump.close_turn()

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
        await self._sync_history(text)
        try:
            await self.agent_hooks.on_agent_done(self.run_context, self.final_llm_resp)
        except Exception as e:  # noqa: BLE001
            logger.error("Error in on_agent_done hook: %s", e, exc_info=True)
        yield AgentResponse(type="llm_result", data=AgentResponseData(chain=chain))

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
