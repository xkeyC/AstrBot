"""Agent runner that hands orchestration and execution to Codex (app-server).

AstrBot keeps message intake, persona, plugin hooks and tools; Codex runs the
agent loop. Plugin tools are exposed as Codex dynamic tools and executed back in
AstrBot with the triggering event, so they behave as in the local agent.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import typing as T
from collections import defaultdict
from pathlib import Path

from astrbot.core import logger, sp
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse, ProviderRequest, TokenUsage

from ...hooks import BaseAgentRunHooks
from ...response import AgentResponseData
from ...run_context import ContextWrapper, TContext
from ..base import AgentResponse, AgentState, BaseAgentRunner
from .app_server_client import (
    CodexAppServerClient,
    CodexAppServerError,
    CodexLaunchOptions,
    JsonObject,
    get_shared_client,
)
from .constants import CODEX_BRIDGE_INSTRUCTIONS, CODEX_THREAD_STATE_KEY
from .tool_bridge import CodexToolBridge

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

# One turn at a time per Codex thread; later messages of the session wait.
_thread_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

_APPROVAL_METHODS = (
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
)
_FINAL_PHASES = (None, "final_answer")


def _launch_options(cfg: dict) -> CodexLaunchOptions:
    overrides = tuple(
        str(kv) for kv in (cfg.get("codex_cli_overrides") or []) if str(kv).strip()
    )
    return CodexLaunchOptions(
        codex_bin=str(cfg.get("codex_bin") or ""),
        codex_home=str(cfg.get("codex_home") or ""),
        config_overrides=overrides,
    )


def _default_cwd(umo: str) -> str:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path

    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in umo)
    path = Path(get_astrbot_data_path()) / "codex_workspaces" / safe
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


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
            return {"type": "image", "url": url}
    return None


def _image_input(ref: str) -> JsonObject:
    if ref.startswith(("http://", "https://", "data:")):
        return {"type": "image", "url": ref}
    if ref.startswith("file:///"):
        ref = ref[len("file:///") :]
    return {"type": "localImage", "path": ref}


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
            items.append({"type": "audio", "url": ref})
        elif os.path.exists(ref):
            items.append({"type": "localAudio", "path": ref})
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
    """Codex app-server agent runner."""

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
        self.bridge = CodexToolBridge(request.func_tool)
        self._client: CodexAppServerClient | None = None
        self._thread_id: str | None = None
        self._turn_id: str | None = None
        self._turn_running = False
        self._aborted = False

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
        if self._turn_running:
            asyncio.ensure_future(self._interrupt())

    def was_aborted(self) -> bool:
        return self._aborted

    async def close(self) -> None:
        if self._turn_running:
            await self._interrupt()

    # ---------------------------------------------------------------- threads

    def _thread_params(self, cwd: str) -> JsonObject:
        cfg = self.cfg
        developer = CODEX_BRIDGE_INSTRUCTIONS
        if extra := str(cfg.get("developer_instructions") or "").strip():
            developer = f"{developer}\n\n{extra}"
        params: JsonObject = {
            "cwd": cwd,
            "approvalPolicy": cfg.get("approval_policy") or "never",
            "sandbox": cfg.get("sandbox") or "read-only",
            "developerInstructions": developer,
        }
        if model := cfg.get("model"):
            params["model"] = model
        if provider := cfg.get("model_provider"):
            params["modelProvider"] = provider
        if base := str(cfg.get("base_instructions") or "").strip():
            params["baseInstructions"] = base
        thread_config = dict(cfg.get("thread_config") or {})
        if effort := cfg.get("reasoning_effort"):
            thread_config.setdefault("model_reasoning_effort", effort)
        if thread_config:
            params["config"] = thread_config
        return params

    async def _ensure_thread(self, client: CodexAppServerClient) -> str:
        cwd = str(self.cfg.get("cwd") or "") or _default_cwd(self.umo)
        params = self._thread_params(cwd)
        state = await sp.get_async(
            scope="umo", scope_id=self.umo, key=CODEX_THREAD_STATE_KEY, default={}
        )
        thread_id = state.get("thread_id") if isinstance(state, dict) else None
        if thread_id and state.get("tools_fp") == self.bridge.fingerprint:
            if client.is_loaded(thread_id):
                return thread_id
            try:
                await client.request("thread/resume", {"threadId": thread_id, **params})
                client.mark_loaded(thread_id)
                logger.info("Codex thread resumed: %s (umo=%s)", thread_id, self.umo)
                return thread_id
            except CodexAppServerError as e:
                logger.warning(
                    "Resume codex thread %s failed, starting new: %s", thread_id, e
                )
        elif thread_id:
            # Dynamic tools are fixed at thread/start, so a changed tool set
            # needs a fresh thread.
            logger.info(
                "AstrBot tool set changed for umo=%s, starting a new Codex thread.",
                self.umo,
            )

        start = dict(params)
        if tools := self.bridge.dynamic_tools():
            start["dynamicTools"] = tools
        start["serviceName"] = "astrbot"
        resp = await client.request("thread/start", start)
        thread_id = resp["thread"]["id"]
        client.mark_loaded(thread_id)
        await sp.put_async(
            scope="umo",
            scope_id=self.umo,
            key=CODEX_THREAD_STATE_KEY,
            value={"thread_id": thread_id, "tools_fp": self.bridge.fingerprint},
        )
        logger.info(
            "Codex thread started: %s model=%s tools=%d (umo=%s)",
            thread_id,
            resp.get("model"),
            len(self.bridge.specs),
            self.umo,
        )
        return thread_id

    # ------------------------------------------------------------------- turn

    async def _handle_server_request(
        self, method: str, params: JsonObject
    ) -> JsonObject:
        if method == "item/tool/call":
            return await self.bridge.call(params, self.run_context, self.agent_hooks)
        if method in _APPROVAL_METHODS:
            decision = "accept" if self.cfg.get("auto_approve") else "decline"
            logger.info("Codex approval %s -> %s", method, decision)
            return {"decision": decision}
        if method == "mcpServer/elicitation/request":
            return {"action": "decline"}
        logger.warning("Unhandled codex server request: %s", method)
        return {}

    async def _interrupt(self) -> None:
        if not (self._client and self._thread_id and self._turn_id):
            return
        try:
            await self._client.request(
                "turn/interrupt", {"threadId": self._thread_id, "turnId": self._turn_id}
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("codex turn/interrupt failed: %s", e)

    def _turn_params(self, thread_id: str) -> JsonObject:
        params: JsonObject = {
            "threadId": thread_id,
            "input": build_turn_input(self.req),
        }
        if extra := build_additional_context(self.req):
            params["additionalContext"] = extra
        if model := self.req.model or self.cfg.get("model"):
            params["model"] = model
        if effort := self.cfg.get("reasoning_effort"):
            params["effort"] = effort
        return params

    async def _run_turn(self) -> T.AsyncGenerator[AgentResponse, None]:
        client = await get_shared_client(_launch_options(self.cfg))
        self._client = client
        thread_id = await self._ensure_thread(client)
        self._thread_id = thread_id
        show_commentary = bool(self.cfg.get("show_commentary"))
        timeout = float(self.cfg.get("turn_timeout") or 600)

        async with _thread_locks[thread_id]:
            queue = client.open_route(thread_id, self._handle_server_request)
            phases: dict[str, str | None] = {}
            final_texts: list[str] = []
            commentary: list[str] = []
            reasoning: list[str] = []
            usage: JsonObject | None = None
            error_msg: str | None = None
            status = "inProgress"
            started = time.monotonic()
            try:
                resp = await client.request("turn/start", self._turn_params(thread_id))
                self._turn_id = resp["turn"]["id"]
                self._turn_running = True
                while True:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        await self._interrupt()
                        error_msg = f"Codex turn timed out after {timeout:.0f}s"
                        break
                    try:
                        msg = await asyncio.wait_for(queue.get(), remaining)
                    except asyncio.TimeoutError:
                        continue
                    method = msg.get("method")
                    p = msg.get("params") or {}
                    if method == "item/started":
                        item = p.get("item") or {}
                        if item.get("type") == "agentMessage":
                            phases[item["id"]] = item.get("phase")
                    elif method == "item/agentMessage/delta":
                        phase = phases.get(p.get("itemId", ""))
                        if self.streaming and (
                            phase in _FINAL_PHASES or show_commentary
                        ):
                            yield AgentResponse(
                                type="streaming_delta",
                                data=AgentResponseData(
                                    chain=MessageChain().message(p.get("delta", ""))
                                ),
                            )
                    elif method == "item/completed":
                        item = p.get("item") or {}
                        kind = item.get("type")
                        if kind == "agentMessage":
                            text = item.get("text") or ""
                            if item.get("phase") in _FINAL_PHASES:
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
                        elif kind == "reasoning":
                            reasoning.extend(item.get("summary") or [])
                    elif method == "thread/tokenUsage/updated":
                        usage = p.get("tokenUsage") or usage
                    elif method == "error":
                        if not p.get("willRetry"):
                            err = p.get("error") or {}
                            error_msg = err.get("message") or str(err)
                    elif method == "turn/completed":
                        turn = p.get("turn") or {}
                        status = turn.get("status", "completed")
                        if turn.get("error") and not error_msg:
                            error_msg = (turn["error"] or {}).get("message")
                        break
                    elif method == "_connection/closed":
                        raise CodexAppServerError(p.get("message", "connection closed"))
            finally:
                self._turn_running = False
                client.close_route(thread_id)

        text = "\n\n".join(t for t in final_texts if t.strip())
        if show_commentary and commentary:
            text = "\n\n".join([*commentary, text]) if text else "\n\n".join(commentary)
        if not text and (commentary or final_texts):
            text = (commentary or final_texts)[-1]

        if status == "failed" or (error_msg and not text):
            raise CodexAppServerError(error_msg or f"turn {status}")
        if status == "interrupted" and not text:
            text = "（已中断）" if self._aborted else ""

        chain = MessageChain().message(text)
        self.final_llm_resp = LLMResponse(
            role="assistant",
            result_chain=chain,
            reasoning_content="\n".join(reasoning) or None,
        )
        self.final_llm_resp.usage = _token_usage(usage)
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
            await ctx.conversation_manager.add_message_pair(
                conv.cid,
                {"role": "user", "content": self.req.prompt or ""},
                {"role": "assistant", "content": text},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to mirror codex exchange into history: %s", e)


def _token_usage(usage: JsonObject | None) -> TokenUsage | None:
    last = (usage or {}).get("last") if usage else None
    if not isinstance(last, dict):
        return None
    try:
        cached = int(last.get("cachedInputTokens") or 0)
        return TokenUsage(
            input_other=int(last.get("inputTokens") or 0) - cached,
            input_cached=cached,
            output=int(last.get("outputTokens") or 0),
        )
    except Exception:  # noqa: BLE001
        return None
