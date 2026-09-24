"""Full-duplex voice through Codex realtime over WebRTC.

Each voice conversation has its own Codex thread (the "voice agent"), apart
from the chat threads, persisted per conversation key and resumed the next
time. It gets what an ordinary member of its paired chat gets there (tools,
execution environment, approvals; see ``tools``). The realtime model listens
and speaks; tasks it hands off run on that thread and its results are fed
back to the model by Codex.

The platform supplies the audio as a ``VoiceMedia``: Mumble mixes Opus
streams, a phone bridge carries raw PCM (see ``pcm``).
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)

from astrbot import logger
from astrbot.core import astrbot_config, sp
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .icetcp import IceTcpRelay, replace_candidates, tcp_candidates

VOICE_THREAD_KEY = "voice_thread"
# Voices of realtime v1/v3, which subscription (WebRTC) sessions use.
REALTIME_VOICES = (
    "juniper",
    "maple",
    "spruce",
    "ember",
    "vale",
    "breeze",
    "arbor",
    "sol",
    "cove",
)
# Per-thread overrides of the voice agent, on top of the runner's engine
# config (model, provider, effort, web search, sandbox, approvals and
# agents.enabled=false all come from there, as for chat threads).
VOICE_THREAD_CONFIG = {
    # Tools are sent directly: no code mode, so no code-mode host needed.
    "model_tool_mode": "direct",
    # ChatGPT apps (connectors) would expose the account's connected data.
    "features.apps": False,
    # Memories: off unless the runner enables them; see VoiceSession._start.
    "features.memories": False,
    # Nothing could deliver a generated image from a voice conversation.
    "features.image_generation": False,
    # No sub-agents, whatever thread_config says (multi_agent_v2 would take
    # precedence over agents.enabled).
    "agents.enabled": False,
    "features.multi_agent_v2": False,
    # Chat turns carry their send time in AstrBot's message metadata; voice
    # handoffs carry nothing, so let Codex state today's date and timezone
    # every turn. Without it the model searches for "latest" news as of its
    # training data.
    "include_environment_context": True,
}
SDP_TIMEOUT = 30.0
CONNECT_TIMEOUT = 15.0
# Pause between WebRTC connecting and handing the model the audio kept
# meanwhile. Core reports the conversation started before the SDP answer, so
# there is no later signal to wait for; sent right away, the start is lost.
READY_DELAY = 0.5
# How long a close waits for a start in progress to reach a safe point.
CLOSE_WAIT = 10.0

VOICE_AGENT_INSTRUCTIONS = """You are the backend of {name}, a voice assistant.
Requests reach you from the voice model, which reads your answers aloud.
Answer in plain spoken language: short, no Markdown, no tables, no code blocks
unless asked, and in the language of the request."""

# Appended to every realtime prompt: the realtime model has no clock of its own.
TIME_PROMPT = """Today is {date} ({weekday}), time zone {timezone}; the current time was {time} when this conversation started. Your own knowledge is older than that: anything about current events, news, prices, weather, schedules or other recent or changing information must be delegated to the backend, never answered from memory."""


def _patch_ice_candidates() -> None:
    """Skips host candidates that cannot carry WebRTC media.

    Proxy TUN adapters (fake-IP range 198.18.0.0/15) and link-local
    addresses get nominated first on machines that have them, and the DTLS
    handshake then stalls, so they are left out of ICE gathering.
    """
    import aioice.ice as ice

    if getattr(ice.get_host_addresses, "_astrbot_filtered", False):
        return
    original = ice.get_host_addresses
    fake_ip = ipaddress.ip_network("198.18.0.0/15")

    def filtered(use_ipv4: bool, use_ipv6: bool) -> list[str]:
        addresses = []
        for address in original(use_ipv4, use_ipv6):
            ip = ipaddress.ip_address(address)
            if not ip.is_link_local and ip not in fake_ip:
                addresses.append(address)
        return addresses

    filtered._astrbot_filtered = True  # type: ignore[attr-defined]
    ice.get_host_addresses = filtered


_patch_ice_candidates()


def _local_address() -> str:
    """A host address of ours that aioice also uses, for the relay candidate."""
    import aioice.ice as ice

    addresses = list(ice.get_host_addresses(True, False))
    return addresses[0] if addresses else "127.0.0.1"


# aioice logs every connectivity check at INFO.
logging.getLogger("aioice").setLevel(logging.WARNING)


def _disable_consent_expiry() -> None:
    """Keeps aioice from dropping a working call on a slow path.

    aioice checks consent (RFC 7675) every ~5 s with a single STUN request
    that waits about 0.5 s and is never retransmitted, and closes the ICE
    connection after 6 misses. On a path with a round trip above that (a
    proxy, a distant network) every check misses and a healthy call is torn
    down after ~30 s. Consent freshness guards browsers against being used to
    send traffic; here the end of a call is known from the peer closing DTLS,
    from Codex reporting the realtime session closed, and from standby.
    """
    import aioice.ice as ice

    ice.CONSENT_FAILURES = 1_000_000_000


_disable_consent_expiry()


class VoiceMedia(Protocol):
    """The platform side of a voice session's audio."""

    # What the model hears, served at real-time pace.
    track: MediaStreamTrack

    async def play(self, track: MediaStreamTrack) -> None:
        """Consumes the model's audio track until it ends."""

    def start(self) -> None:
        """The model listens now: hand over what was held back meanwhile."""

    def stop(self) -> None:
        """The session is closing: drop pending audio, end speech in progress."""

    def flush(self) -> None:
        """Drops the model's audio not played yet (its speech was cut)."""


@dataclass
class VoiceOptions:
    name: str
    aliases: list[str]
    voice: str = ""
    model: str = ""
    extra_prompt: str = ""
    agent_instructions: str = ""


def _runner_config() -> dict:
    from astrbot.core.config.agent_runner import normalize_agent_runner

    return normalize_agent_runner(astrbot_config.get("agent_runner"))["config"]


async def _codex_engine(realtime: bool = True):
    """The Codex engine the chat runner uses: same runtime, same account.

    Args:
        realtime: Whether the binding must support Codex realtime.
    """
    from astrbot.core.agent.runners.codex.codex_agent_runner import engine_options
    from astrbot.core.agent.runners.codex.native import CodexEngine

    engine = await CodexEngine.get(engine_options(_runner_config()))
    if realtime and not hasattr(engine.rt, "realtime_start"):
        raise RuntimeError(
            "codex_astrbot binding has no realtime support; update codex-astrbot"
        )
    return engine


def voice_thread_config(memory_scope: str | None) -> dict:
    """Thread overrides for a voice agent whose paired chat is ``memory_scope``.

    With memories enabled on the runner, the voice agent reads the global
    memories and those of its paired chat (the UMO of the server group or of
    the whisperer's private chat), like that chat's own thread does, but may
    never write global memories or delete any.
    """
    from astrbot.core.agent.runners.codex.codex_agent_runner import (
        memory_thread_config,
    )

    config = dict(VOICE_THREAD_CONFIG)
    cfg = _runner_config()
    if cfg.get("memory_enabled") and memory_scope:
        # No turn scopes: nothing global is writable and nothing can be
        # deleted, whoever speaks.
        config.update(memory_thread_config(cfg, memory_scope, None, turn_scopes=False))
    return config


class _SessionClosed(Exception):
    """The session was closed while it was still starting."""


class VoiceSession:
    """One realtime conversation bound to its own voice agent thread.

    Lifecycle: ``launch()`` starts it in the background; ``close()`` may be
    called at any time, from any path (standby, mute, disconnect, a failure),
    and runs once. A close during start lets the start stop at its next step
    and then releases whatever it had created, so nothing outlives the
    session (in particular no realtime call keeps running unowned).

    Another voice model replaces the transport (``_connect``,
    ``_release_transport``, ``say``) and keeps the voice agent thread.
    """

    # Whether the Codex binding must support realtime (the transport here).
    NEEDS_REALTIME = True

    def __init__(
        self,
        key: str,
        scope_id: str,
        prompt: str,
        options: VoiceOptions,
        media: VoiceMedia,
        on_closed: Callable[[VoiceSession], None],
        memory_scope: str | None = None,
        label: str = "voice",
        thread_key: str = VOICE_THREAD_KEY,
    ) -> None:
        """
        Args:
            key: Conversation key within the platform, e.g. ``server``.
            scope_id: Storage scope of the persisted voice thread.
            prompt: Instructions for the realtime model.
            options: Voice settings of the platform.
            media: The platform's audio in and out.
            on_closed: Called once the session has ended, for any reason.
            memory_scope: UMO of the paired chat, whose memories (with the
                global ones) and member tools the voice agent gets.
            label: Names the session in logs and task names, e.g. ``Mumble``.
            thread_key: Storage key of the persisted voice thread.
        """
        self.key = key
        self.scope_id = scope_id
        self.prompt = prompt
        self.options = options
        self.memory_scope = memory_scope
        self.media = media
        self.label = label
        self.thread_key = thread_key
        self._on_closed = on_closed
        self._engine = None
        self._thread_id: str | None = None
        self._events_queue: asyncio.Queue | None = None
        self._realtime_requested = False
        self._thread_released = False
        self._pc: RTCPeerConnection | None = None
        self._relay: IceTcpRelay | None = None
        self._tasks: list[asyncio.Task] = []
        self._start_task: asyncio.Task | None = None
        self._close_task: asyncio.Task | None = None
        self._closed_event = asyncio.Event()
        self._closed = False
        self.created_at = time.monotonic()
        # Set once the model listens; standby only counts from then.
        self.ready = False
        self.started_at = 0.0
        self.last_transcript_at = 0.0

    @property
    def last_activity(self) -> float:
        """Start or latest speech recognised by the model, for standby."""
        return max(self.started_at, self.last_transcript_at)

    def launch(self, on_failed: Callable[[Exception], None]) -> None:
        """Starts the session in the background.

        Args:
            on_failed: Called when starting fails (not when it is closed).
        """

        async def run() -> None:
            try:
                await self._start()
            except _SessionClosed:
                return
            except Exception as exc:  # noqa: BLE001 - reported to the owner
                if self._closed:
                    return
                on_failed(exc)
                # Not awaited: the close waits for this very task to end.
                self._request_close(f"start failed: {exc}")

        self._start_task = asyncio.create_task(
            run(), name=f"{self.label}-voice-{self.key}-start"
        )

    def _phase(self, name: str) -> None:
        logger.debug(
            "%s voice %s: %s after %.1fs",
            self.label,
            self.key,
            name,
            time.monotonic() - self.created_at,
        )

    def _check_open(self) -> None:
        if self._closed:
            raise _SessionClosed

    async def _wait_open(self, awaitable, timeout: float):
        """Awaits ``awaitable``, giving up as soon as the session is closed.

        Raises:
            _SessionClosed: The session was closed first.
            asyncio.TimeoutError: ``timeout`` passed first.
        """
        waiter = asyncio.ensure_future(awaitable)
        closed = asyncio.ensure_future(self._closed_event.wait())
        try:
            done, _ = await asyncio.wait(
                {waiter, closed}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            closed.cancel()
        if waiter in done:
            return waiter.result()
        waiter.cancel()
        if self._closed:
            raise _SessionClosed
        raise asyncio.TimeoutError

    async def _start(self) -> None:
        await self._open_agent()
        await self._connect()

    async def _open_agent(self) -> None:
        """Opens (or resumes) the voice agent thread and routes its events to
        ``self._events_queue``."""
        engine = await _codex_engine(self.NEEDS_REALTIME)
        self._check_open()
        self._engine = engine
        state = await sp.get_async(
            scope="umo", scope_id=self.scope_id, key=self.thread_key, default={}
        )
        self._check_open()
        runner_cfg = _runner_config()
        tools = None
        if self.memory_scope:
            from .tools import voice_agent_tools

            tools = await voice_agent_tools(self.memory_scope, runner_cfg)
            self._check_open()
        if runner_cfg.get("cwd"):
            workspace = Path(str(runner_cfg["cwd"]))
        elif self.memory_scope:
            from astrbot.core.agent.runners.codex.codex_agent_runner import (
                _default_cwd,
            )

            # The paired chat's workspace, as its own thread uses.
            workspace = Path(_default_cwd(self.memory_scope))
        else:
            workspace = Path(get_astrbot_data_path()) / "voice"
        workspace.mkdir(parents=True, exist_ok=True)
        params = {
            "cwd": str(workspace),
            "base_instructions": (
                self.options.agent_instructions
                or VOICE_AGENT_INSTRUCTIONS.format(name=self.options.name)
            ),
            "dynamic_tools": tools.dynamic_tools if tools else [],
            "no_environment": not (tools and tools.native_exec),
            "config": voice_thread_config(self.memory_scope),
        }
        # Opening and unloading this key's thread are serialised: a session
        # closed while its open was still running unloads the thread before
        # anyone else may open it, so it can never unload a newer session's
        # (the same thread id is resumed for the same key).
        async with engine.session_lock(self.scope_id):
            info, started_new = await engine.open_thread(state or None, params)
            self._thread_id = info["thread_id"]
            self._phase("thread opened")
            logger.info(
                "%s voice agent thread for %s: %s (%s)",
                self.label,
                self.key,
                self._thread_id,
                "new" if started_new else "resumed",
            )
            if self._closed:
                self._thread_released = True
                await engine.forget_thread(self._thread_id)
                raise _SessionClosed
        if started_new or state.get("thread_id") != self._thread_id:
            await sp.put_async(
                scope="umo",
                scope_id=self.scope_id,
                key=self.thread_key,
                value={
                    "thread_id": info["thread_id"],
                    "rollout_path": info.get("rollout_path"),
                },
            )
        self._check_open()
        # This thread only ever carries the voice conversation, so its pump
        # route stays open for the whole session and sees every event.
        events = engine.pump(self._thread_id).open_turn(
            tools.tool_handler if tools else None,
            tools.approval_handler if tools else None,
        )
        self._events_queue = events

    async def _connect(self) -> None:
        """Starts the realtime conversation over WebRTC on the agent thread."""
        engine, events = self._engine, self._events_queue
        # No STUN: the far end offers public host candidates and we connect
        # out to them. aiortc's default Google STUN server only adds a
        # multi-second wait while gathering.
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self._pc = pc
        pc.addTrack(self.media.track)
        pc.createDataChannel("oai-events")

        @pc.on("track")
        def on_track(track) -> None:
            if track.kind == "audio":
                self._spawn(self.media.play(track), "outbound")

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            if pc.connectionState in ("failed", "closed"):
                self._request_close(f"webrtc {pc.connectionState}")

        await pc.setLocalDescription(await pc.createOffer())
        self._phase("offer ready")
        self._check_open()
        request: dict = {
            "transport": {"type": "webrtc", "sdp": pc.localDescription.sdp},
            # Subscription (AVAS) calls only accept the frameless protocol.
            "version": "v3",
            "include_startup_context": False,
            "prompt": self.prompt,
        }
        if self.options.voice:
            request["voice"] = self.options.voice
        if self.options.model:
            request["model"] = self.options.model
        answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._spawn(self._events(events, answer), "events")
        self._realtime_requested = True
        await engine.rt.realtime_start(self._thread_id, json.dumps(request))
        sdp = await self._wait_open(answer, SDP_TIMEOUT)
        self._phase("answer received")
        if proxy := str(_runner_config().get("proxy") or "").strip():
            # Media cannot take a proxy over UDP: go through the peer's
            # ICE-TCP candidates, one TCP connection opened via the proxy.
            candidates = tcp_candidates(sdp)
            if not candidates:
                raise RuntimeError(
                    "proxy set, but the peer offered no ICE-TCP candidate"
                )
            self._relay = IceTcpRelay(proxy, candidates)
            port = await self._wait_open(self._relay.start(), CONNECT_TIMEOUT)
            sdp = replace_candidates(sdp, _local_address(), port)
            self._phase("media relay through proxy ready")
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))
        deadline = time.monotonic() + CONNECT_TIMEOUT
        while pc.connectionState != "connected":
            self._check_open()
            if time.monotonic() > deadline or pc.connectionState in (
                "failed",
                "closed",
            ):
                raise RuntimeError(f"WebRTC did not connect ({pc.connectionState})")
            await asyncio.sleep(0.1)
        self._phase("webrtc connected")
        with contextlib.suppress(asyncio.TimeoutError):
            await self._wait_open(asyncio.sleep(READY_DELAY), READY_DELAY + 1)
        self.media.start()
        self.started_at = time.monotonic()
        self.ready = True
        logger.info(
            "%s voice session %s started in %.1fs (thread %s)",
            self.label,
            self.key,
            self.started_at - self.created_at,
            self._thread_id,
        )

    async def say(self, text: str) -> None:
        """Gives the realtime model a text input, which it answers aloud.

        Args:
            text: What the model is told, e.g. why the bot placed a call.

        Raises:
            RuntimeError: The session is not ready.
        """
        if not self.ready or self._closed or self._engine is None:
            raise RuntimeError("voice session is not ready")
        await self._engine.rt.realtime_append_text(self._thread_id, text)

    def _spawn(self, coro, name: str) -> None:
        task = asyncio.create_task(coro, name=f"{self.label}-voice-{self.key}-{name}")
        self._tasks.append(task)

    async def _events(self, events: asyncio.Queue, answer: asyncio.Future) -> None:
        while True:
            msg = await events.get()
            kind = msg.get("type")
            if kind == "realtime_conversation_sdp":
                if not answer.done():
                    answer.set_result(msg["sdp"])
            elif kind == "realtime_conversation_closed":
                reason = msg.get("reason") or "closed"
                if not answer.done():
                    answer.set_exception(RuntimeError(f"realtime closed: {reason}"))
                self._realtime_requested = False
                self._request_close(f"realtime {reason}")
                return
            elif kind == "realtime_conversation_realtime":
                payload = msg.get("payload")
                if isinstance(payload, dict) and "Error" in payload:
                    logger.warning(
                        "%s voice %s: realtime error: %s",
                        self.label,
                        self.key,
                        payload["Error"],
                    )
                    if not answer.done():
                        answer.set_exception(RuntimeError(str(payload["Error"])))
                elif isinstance(payload, dict) and "InputTranscriptDelta" in payload:
                    self.last_transcript_at = time.monotonic()
                elif isinstance(payload, dict) and "InputTranscriptDone" in payload:
                    self.last_transcript_at = time.monotonic()
                    logger.debug(
                        "%s voice %s heard: %s",
                        self.label,
                        self.key,
                        payload["InputTranscriptDone"].get("text"),
                    )
            elif kind == "_pump_closed":
                if not answer.done():
                    answer.set_exception(RuntimeError("voice thread closed"))
                self._realtime_requested = False
                self._request_close("voice thread closed")
                return

    @property
    def closing(self) -> bool:
        return self._closed

    def _request_close(self, reason: str) -> asyncio.Task:
        """Starts closing (once) and returns the task doing it."""
        if self._close_task is None:
            self._closed = True
            self._closed_event.set()
            # Model audio still arriving is dropped, speech in progress ends.
            self.media.stop()
            self._close_task = asyncio.create_task(
                self._close(reason), name=f"{self.label}-voice-{self.key}-close"
            )
        return self._close_task

    async def close(self, reason: str = "") -> None:
        """Closes the session and waits until it is released.

        The release runs in its own task, so cancelling a caller does not stop
        it half way; every caller waits for the same release.
        """
        task = self._request_close(reason)
        if task is not asyncio.current_task():
            await asyncio.shield(task)

    async def _close(self, reason: str) -> None:
        try:
            start = self._start_task
            if start is not None and not start.done():
                # A start stops at its next step once closed; let it, so that
                # everything it created is known here and the realtime stop is
                # sent after its start.
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.shield(start), CLOSE_WAIT)
                # A start still inside a slow call is covered: one opening the
                # thread unloads it itself on return, and the realtime start
                # is queued on the thread ahead of the stop sent below.
            logger.info("%s voice session %s closed: %s", self.label, self.key, reason)
            await self._release()
        finally:
            self._on_closed(self)

    async def _release_transport(self) -> None:
        """Stops the realtime conversation and closes WebRTC (each once)."""
        engine, thread_id = self._engine, self._thread_id
        if self._realtime_requested and engine is not None and thread_id:
            self._realtime_requested = False
            with contextlib.suppress(Exception):
                await engine.rt.realtime_stop(thread_id)
        pc, self._pc = self._pc, None
        if pc is not None:
            with contextlib.suppress(Exception):
                await pc.close()
        relay, self._relay = self._relay, None
        if relay is not None:
            relay.close()

    async def _release(self) -> None:
        """Releases what exists now; each resource only once, so it can run
        again for what a late start created afterwards."""
        engine, thread_id = self._engine, self._thread_id
        await self._release_transport()
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current:
                task.cancel()
        if engine is not None and thread_id is not None and not self._thread_released:
            self._thread_released = True
            pump = engine.pumps.get(thread_id)
            if (
                pump is not None
                and pump.route is not None
                and pump.route.events is self._events_queue
            ):
                pump.close_turn()
            # Unload the thread; the next session resumes it from its rollout.
            async with engine.session_lock(self.scope_id):
                await engine.forget_thread(thread_id)


def time_prompt() -> str:
    """The current date for the realtime model, in AstrBot's configured zone."""
    import datetime
    import zoneinfo

    now = None
    if zone := astrbot_config.get("timezone"):
        with contextlib.suppress(Exception):
            now = datetime.datetime.now(zoneinfo.ZoneInfo(zone))
    if now is None:
        now = datetime.datetime.now().astimezone()
    return TIME_PROMPT.format(
        date=now.strftime("%Y-%m-%d"),
        weekday=now.strftime("%A"),
        timezone=now.strftime("%Z") or now.strftime("%z"),
        time=now.strftime("%H:%M"),
    )
