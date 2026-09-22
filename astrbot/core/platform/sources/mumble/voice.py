"""Full-duplex voice through Codex realtime over WebRTC.

Each voice conversation has its own Codex thread (the "voice agent"), kept
apart from the chat threads: it has no AstrBot tools and no execution
environment, so what is said by voice cannot reach other conversations. The
thread is persisted per conversation key and resumed the next time. The
realtime model listens and speaks; tasks it hands off run on that thread and
its results are fed back to the model by Codex.
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

from aiortc import RTCPeerConnection, RTCSessionDescription

from astrbot import logger
from astrbot.core import astrbot_config, sp
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .audio import InboundMixer, MixerTrack, OutboundVoice

VOICE_THREAD_KEY = "mumble_voice_thread"
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
SDP_TIMEOUT = 30.0
CONNECT_TIMEOUT = 15.0
# Pause between the model reporting it started and sending it audio.
READY_DELAY = 0.5

VOICE_AGENT_INSTRUCTIONS = """You are the backend of {name}, a voice assistant in a Mumble voice chat.
Requests reach you from the voice model, which reads your answers aloud.
Answer in plain spoken language: short, no Markdown, no tables, no code blocks
unless asked, and in the language of the request."""

CHANNEL_PROMPT = """Your name is {name}.

You are listening to a voice chat room where several people talk with each other. Almost everything you hear is people talking to each other, not to you.

The one rule that matters most: speak ONLY when the speaker says your name{aliases} to you in that utterance, or is directly continuing an exchange with you from a few seconds ago. In every other case produce no audio and no text at all - complete silence. Do not acknowledge, do not react, do not say "mm", do not comment, do not delegate.

When you are addressed, answer briefly in the speaker's language. Delegate real tasks (anything needing facts, lookups or work) to the backend and tell the speaker the result briefly."""

WHISPER_PROMPT = """Your name is {name}. You are talking privately, one to one, with {speaker} in a Mumble voice chat. Everything you hear is meant for you.

Answer briefly in the speaker's language. Delegate real tasks (anything needing facts, lookups or work) to the backend and tell the speaker the result briefly."""


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
# aioice logs every connectivity check at INFO.
logging.getLogger("aioice").setLevel(logging.WARNING)


@dataclass
class VoiceOptions:
    name: str
    aliases: list[str]
    voice: str = ""
    model: str = ""
    extra_prompt: str = ""
    agent_instructions: str = ""


async def _codex_engine():
    """The Codex engine the chat runner uses: same runtime, same account."""
    from astrbot.core.agent.runners.codex.codex_agent_runner import engine_options
    from astrbot.core.agent.runners.codex.native import CodexEngine
    from astrbot.core.config.agent_runner import normalize_agent_runner

    cfg = normalize_agent_runner(astrbot_config.get("agent_runner"))["config"]
    engine = await CodexEngine.get(engine_options(cfg))
    if not hasattr(engine.rt, "realtime_start"):
        raise RuntimeError(
            "codex_astrbot binding has no realtime support; update codex-astrbot"
        )
    return engine


class _SessionClosed(Exception):
    """The session was closed while it was still starting."""


class VoiceSession:
    """One realtime conversation bound to its own voice agent thread.

    Lifecycle: ``launch()`` starts it in the background; ``close()`` may be
    called at any time, from any path (standby, mute, disconnect, a failure),
    and runs once. A close during start lets the start stop at its next step
    and then releases whatever it had created, so nothing outlives the
    session (in particular no realtime call keeps running unowned).
    """

    def __init__(
        self,
        key: str,
        scope_id: str,
        prompt: str,
        options: VoiceOptions,
        send_audio: Callable[[bytes, bool], None],
        on_closed: Callable[[VoiceSession], None],
    ) -> None:
        """
        Args:
            key: Conversation key within the platform, e.g. ``server``.
            scope_id: Storage scope of the persisted voice thread.
            prompt: Instructions for the realtime model.
            options: Voice settings of the platform.
            send_audio: Sends one Opus frame to Mumble: ``(frame, terminator)``.
            on_closed: Called once the session has ended, for any reason.
        """
        self.key = key
        self.scope_id = scope_id
        self.prompt = prompt
        self.options = options
        self.mixer = InboundMixer()
        self.outbound = OutboundVoice(send_audio)
        self._on_closed = on_closed
        self._engine = None
        self._thread_id: str | None = None
        self._events_queue: asyncio.Queue | None = None
        self._realtime_requested = False
        self._pc: RTCPeerConnection | None = None
        self._tasks: list[asyncio.Task] = []
        self._start_task: asyncio.Task | None = None
        self._started = asyncio.Event()
        self._closed = False
        self.started_at = time.monotonic()
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
                await self.close(f"start failed: {exc}")

        self._start_task = asyncio.create_task(
            run(), name=f"mumble-voice-{self.key}-start"
        )

    def _check_open(self) -> None:
        if self._closed:
            raise _SessionClosed

    async def _start(self) -> None:
        engine = await _codex_engine()
        self._check_open()
        self._engine = engine
        state = await sp.get_async(
            scope="umo", scope_id=self.scope_id, key=VOICE_THREAD_KEY, default={}
        )
        self._check_open()
        workspace = Path(get_astrbot_data_path()) / "mumble_voice"
        workspace.mkdir(parents=True, exist_ok=True)
        params = {
            "cwd": str(workspace),
            "base_instructions": (
                self.options.agent_instructions
                or VOICE_AGENT_INSTRUCTIONS.format(name=self.options.name)
            ),
            "no_environment": True,
            # No tools to orchestrate, so no code mode (and no host needed).
            "config": {"model_tool_mode": "direct"},
        }
        info, started_new = await engine.open_thread(state or None, params)
        self._thread_id = info["thread_id"]
        if started_new or state.get("thread_id") != self._thread_id:
            await sp.put_async(
                scope="umo",
                scope_id=self.scope_id,
                key=VOICE_THREAD_KEY,
                value={
                    "thread_id": info["thread_id"],
                    "rollout_path": info.get("rollout_path"),
                },
            )
        self._check_open()
        # This thread only ever carries the voice conversation, so its pump
        # route stays open for the whole session and sees every event.
        events = engine.pump(self._thread_id).open_turn(None, None)
        self._events_queue = events

        pc = RTCPeerConnection()
        self._pc = pc
        pc.addTrack(MixerTrack(self.mixer))
        pc.createDataChannel("oai-events")

        @pc.on("track")
        def on_track(track) -> None:
            if track.kind == "audio":
                self._spawn(self.outbound.run(track), "outbound")

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            if pc.connectionState in ("failed", "closed"):
                await self.close(f"webrtc {pc.connectionState}")

        await pc.setLocalDescription(await pc.createOffer())
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
        sdp = await asyncio.wait_for(answer, SDP_TIMEOUT)
        self._check_open()
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
        # Hand over the audio kept while connecting only once the model is
        # listening; sent earlier, its start is lost.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._started.wait(), CONNECT_TIMEOUT)
        await asyncio.sleep(READY_DELAY)
        self._check_open()
        self.mixer.holding = False
        logger.info(
            "Mumble voice session %s started (thread %s)", self.key, self._thread_id
        )

    def _spawn(self, coro, name: str) -> None:
        task = asyncio.create_task(coro, name=f"mumble-voice-{self.key}-{name}")
        self._tasks.append(task)

    async def _events(self, events: asyncio.Queue, answer: asyncio.Future) -> None:
        while True:
            msg = await events.get()
            kind = msg.get("type")
            if kind == "realtime_conversation_started":
                self._started.set()
            elif kind == "realtime_conversation_sdp":
                if not answer.done():
                    answer.set_result(msg["sdp"])
            elif kind == "realtime_conversation_closed":
                reason = msg.get("reason") or "closed"
                if not answer.done():
                    answer.set_exception(RuntimeError(f"realtime closed: {reason}"))
                self._realtime_requested = False
                self._spawn(self.close(f"realtime {reason}"), "close")
                return
            elif kind == "realtime_conversation_realtime":
                payload = msg.get("payload")
                if isinstance(payload, dict) and "Error" in payload:
                    logger.warning(
                        "Mumble voice %s: realtime error: %s",
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
                        "Mumble voice %s heard: %s",
                        self.key,
                        payload["InputTranscriptDone"].get("text"),
                    )
            elif kind == "_pump_closed":
                if not answer.done():
                    answer.set_exception(RuntimeError("voice thread closed"))
                self._realtime_requested = False
                self._spawn(self.close("voice thread closed"), "close")
                return

    async def close(self, reason: str = "") -> None:
        if self._closed:
            return
        self._closed = True
        start = self._start_task
        current = asyncio.current_task()
        if start is not None and not start.done() and start is not current:
            # Let the start stop at its next step, so everything it created
            # is known here and the realtime stop is sent after its start.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.shield(start), SDP_TIMEOUT + CONNECT_TIMEOUT
                )
        logger.info("Mumble voice session %s closed: %s", self.key, reason)
        self.outbound.finish()
        engine, thread_id = self._engine, self._thread_id
        if self._realtime_requested and engine is not None and thread_id is not None:
            with contextlib.suppress(Exception):
                await engine.rt.realtime_stop(thread_id)
        if self._pc is not None:
            with contextlib.suppress(Exception):
                await self._pc.close()
        for task in self._tasks:
            if task is not current:
                task.cancel()
        if engine is not None and thread_id is not None:
            pump = engine.pumps.get(thread_id)
            if (
                pump is not None
                and pump.route is not None
                and (pump.route.events is self._events_queue)
            ):
                pump.close_turn()
            # Unload the thread; the next session resumes it from its rollout.
            await engine.forget_thread(thread_id)
        self.mixer.clear()
        self._on_closed(self)


def channel_prompt(options: VoiceOptions) -> str:
    aliases = [a for a in options.aliases if a and a != options.name]
    alias_text = (
        f' ("{options.name}"' + "".join(f', "{a}"' for a in aliases) + ")"
        if aliases
        else f' "{options.name}"'
    )
    prompt = CHANNEL_PROMPT.format(name=options.name, aliases=alias_text)
    if options.extra_prompt:
        prompt += "\n\n" + options.extra_prompt
    return prompt


def whisper_prompt(options: VoiceOptions, speaker: str) -> str:
    prompt = WHISPER_PROMPT.format(name=options.name, speaker=speaker)
    if options.extra_prompt:
        prompt += "\n\n" + options.extra_prompt
    return prompt
