"""The chat a voice conversation shares its agent with.

A voice conversation has no agent of its own: what the voice model hands off
runs as a turn of the paired chat's Codex thread, through the chat runner,
like a message of that chat. So voice and text share one context, persona,
tool set, memory and approvals, and queue behind each other on the chat's
session lock. The answer comes back to be spoken; nothing is posted.

Identity: in a private conversation (a call, a whisper) the turn runs as the
person talking, with their own permissions. In a group (a voice channel) the
speakers cannot be told apart, so it runs as one fixed voice user, locked to
the member role.
"""

from __future__ import annotations

from dataclasses import dataclass

from astrbot import logger

# The sender of group voice turns (never an admin, whatever is configured).
VOICE_SENDER_ID = "voice"
VOICE_SENDER_NAME = "Voice"

REQUEST_PROMPT = """<voice_request via="{via}" speaker="{speaker}">
{body}
</voice_request>
This came from a live voice conversation. Your final reply is read aloud there, not posted to the chat: answer in plain spoken language, briefly, without Markdown, lists, links or code, in the speaker's language."""

TASK_BODY = """What was said: {heard}
Task: {task}"""

OPENING_BODY = """The voice conversation just started. Give the words to open it with. Purpose: {purpose}"""


@dataclass
class VoiceChat:
    """The paired chat of a voice conversation, and who talks in it.

    Attributes:
        umo: Unified message origin of the chat.
        private: One person with the bot (a call, a whisper): turns run as
            ``sender_id`` with their permissions. Otherwise (a channel) they
            run as the voice user, a member.
        sender_id: The person talking, for a private chat.
        sender_name: Their display name.
        via: What the conversation is, for the agent, e.g. ``QQ voice call``.
    """

    umo: str
    private: bool
    sender_id: str = VOICE_SENDER_ID
    sender_name: str = VOICE_SENDER_NAME
    via: str = "voice"

    def busy(self) -> bool:
        """Whether a turn runs in the chat now (a request would queue)."""
        from astrbot.core.agent.runners.codex.native import ACTIVE_TURNS

        return self.umo in ACTIVE_TURNS

    async def voice_persona(self) -> str:
        """The voice persona of the chat's active persona: short instructions
        for the voice model, or empty when there are none.

        The persona is resolved as for the chat's turns (a persona forced on
        the session, else the conversation's, else the configured default).
        """
        from astrbot.core.star.context import current_context

        ctx = current_context()
        if ctx is None:
            return ""
        try:
            conversation_id = await ctx.conversation_manager.get_curr_conversation_id(
                self.umo
            )
            conversation = (
                await ctx.conversation_manager.get_conversation(
                    self.umo, conversation_id
                )
                if conversation_id
                else None
            )
            _, persona, _, _ = await ctx.persona_manager.resolve_selected_persona(
                umo=self.umo,
                conversation_persona_id=conversation.persona_id
                if conversation
                else None,
                platform_name=self.umo.split(":", 1)[0],
                provider_settings=ctx.get_config(umo=self.umo),
            )
        except Exception as exc:  # noqa: BLE001 - the platform default applies
            logger.warning("Voice: persona of %s not resolved: %s", self.umo, exc)
            return ""
        return str((persona or {}).get("voice_prompt") or "").strip()

    async def ask(self, body: str) -> str | None:
        """Runs ``body`` as a turn of the chat and returns the answer.

        Args:
            body: The request (see TASK_BODY, OPENING_BODY).

        Returns:
            The answer to speak, or None when the turn was stopped, failed or
            the core is not up.
        """
        from astrbot.core.agent.runners.codex.wake import run_turn_in_session
        from astrbot.core.cron.events import CronMessageEvent
        from astrbot.core.platform.message_session import MessageSession
        from astrbot.core.star.context import current_context

        ctx = current_context()
        if ctx is None:
            logger.warning("Voice: no core context, a request is not answered")
            return None
        session = MessageSession.from_str(self.umo)
        sender_id = self.sender_id if self.private else VOICE_SENDER_ID
        sender_name = self.sender_name if self.private else VOICE_SENDER_NAME
        prompt = REQUEST_PROMPT.format(via=self.via, speaker=sender_name, body=body)
        event = CronMessageEvent(
            context=ctx,
            session=session,
            message=prompt,
            sender_id=sender_id,
            sender_name=sender_name,
            message_type=session.message_type,
        )
        event.message_obj.sender.user_id = sender_id
        if not self.private:
            event.message_obj.group_id = session.session_id
        cfg = ctx.get_config(umo=self.umo)
        admins = {str(a) for a in cfg.get("admins_id", [])}
        event.role = "admin" if self.private and sender_id in admins else "member"
        try:
            return await run_turn_in_session(ctx, event, cfg, prompt)
        except Exception as exc:  # noqa: BLE001 - reported; the voice goes on
            logger.warning("Voice: request to %s failed: %s", self.umo, exc)
            return None
