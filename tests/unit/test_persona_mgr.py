from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from astrbot.core.persona_mgr import PersonaManager


@pytest.mark.asyncio
async def test_resolve_selected_persona_prefers_forced_session_persona():
    acm = MagicMock()
    acm.default_conf = {"provider_settings": {}}
    manager = PersonaManager(MagicMock(), acm)
    manager.personas_v3 = [
        {"name": "event-persona", "prompt": "event"},
        {"name": "session-persona", "prompt": "session"},
    ]

    with patch(
        "astrbot.core.persona_mgr.sp.get_async",
        AsyncMock(return_value={"persona_id": "session-persona"}),
    ):
        (
            persona_id,
            persona,
            force_applied_persona_id,
            use_webchat_default,
        ) = await manager.resolve_selected_persona(
            umo="platform:FriendMessage:user",
            conversation_persona_id="conversation-persona",
            platform_name="aiocqhttp",
            provider_settings={"default_personality": "default-persona"},
            selected_persona_id="event-persona",
        )

    assert persona_id == "session-persona"
    assert persona == {"name": "session-persona", "prompt": "session"}
    assert force_applied_persona_id == "session-persona"
    assert use_webchat_default is False
