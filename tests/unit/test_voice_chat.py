import pytest

from astrbot.core.agent.runners.codex import native, wake
from astrbot.core.star import context as star_context
from astrbot.core.star.context import Context
from astrbot.core.voice.chat import VOICE_SENDER_ID, VoiceChat


class FakeContext(Context):
    def __init__(self) -> None:  # none of the real managers are needed
        pass

    def get_config(self, umo=None):
        return {"admins_id": ["42", VOICE_SENDER_ID]}


@pytest.fixture
def turns(monkeypatch):
    seen = []

    async def run_turn(ctx, event, cfg, prompt):
        seen.append(event)
        return "answer"

    monkeypatch.setattr(wake, "run_turn_in_session", run_turn)
    monkeypatch.setattr(star_context, "_current", FakeContext())
    return seen


@pytest.mark.asyncio
async def test_a_private_voice_turn_runs_as_the_speaker(turns):
    chat = VoiceChat(
        umo="qq:FriendMessage:42", private=True, sender_id="42", sender_name="Alice"
    )
    assert await chat.ask("Task: x") == "answer"
    (event,) = turns
    assert event.get_sender_id() == "42"
    assert event.role == "admin"  # the speaker's own permissions
    assert event.unified_msg_origin == "qq:FriendMessage:42"
    assert "Task: x" in event.message_str and "read aloud" in event.message_str


@pytest.mark.asyncio
async def test_a_group_voice_turn_runs_as_the_voice_member(turns):
    chat = VoiceChat(umo="mumble:GroupMessage:server", private=False, sender_id="42")
    await chat.ask("Task: x")
    (event,) = turns
    assert event.get_sender_id() == VOICE_SENDER_ID
    # Locked to member, even if "voice" were listed as an admin.
    assert event.role == "member"
    assert event.get_group_id() == "server"


@pytest.mark.asyncio
async def test_no_core_context_means_no_answer(monkeypatch):
    monkeypatch.setattr(star_context, "_current", None)
    assert await VoiceChat(umo="qq:FriendMessage:1", private=True).ask("x") is None


def test_busy_follows_the_chats_active_turn(monkeypatch):
    chat = VoiceChat(umo="qq:FriendMessage:1", private=True)
    assert not chat.busy()
    monkeypatch.setitem(native.ACTIVE_TURNS, "qq:FriendMessage:1", object())
    assert chat.busy()
