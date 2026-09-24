import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text

from astrbot.core.db.sqlite import SQLiteDatabase
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


@pytest.mark.asyncio
async def test_persona_voice_prompt_migration_and_round_trip(tmp_path):
    """A legacy personas table gains voice_prompt, and the manager round-trips it."""
    db_path = tmp_path / "legacy-personas.db"
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            "CREATE TABLE personas ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "persona_id VARCHAR(255) NOT NULL UNIQUE, "
            "system_prompt TEXT NOT NULL, "
            "begin_dialogs JSON, "
            "tools JSON, "
            "created_at DATETIME, "
            "updated_at DATETIME)"
        )
        conn.execute(
            "INSERT INTO personas (persona_id, system_prompt, begin_dialogs) "
            "VALUES ('legacy', 'Legacy persona.', '[]')"
        )
    conn.close()

    db = SQLiteDatabase(str(db_path))
    try:
        await db.initialize()
        async with db.engine.connect() as conn:
            result = await conn.execute(text("PRAGMA table_info(personas)"))
            columns = {row[1] for row in result.fetchall()}
        assert {"custom_error_message", "voice_prompt"} <= columns
        legacy = await db.get_persona_by_id("legacy")
        assert legacy.voice_prompt is None

        acm = MagicMock()
        acm.default_conf = {}
        manager = PersonaManager(db, acm)
        await manager.initialize()

        created = await manager.create_persona(
            persona_id="voice",
            system_prompt="Full persona.",
            voice_prompt="Speak softly.",
        )
        assert created.voice_prompt == "Speak softly."
        assert (await db.get_persona_by_id("voice")).voice_prompt == "Speak softly."
        v3 = manager.get_persona_v3_by_id("voice")
        assert v3 is not None
        assert v3["voice_prompt"] == "Speak softly."

        clone = await manager.clone_persona("voice", "voice-clone")
        assert clone.voice_prompt == "Speak softly."

        # Updating other fields leaves voice_prompt unchanged.
        await manager.update_persona("voice", system_prompt="Updated persona.")
        assert (await db.get_persona_by_id("voice")).voice_prompt == "Speak softly."

        await manager.update_persona("voice", voice_prompt="Be brief.")
        assert (await db.get_persona_by_id("voice")).voice_prompt == "Be brief."

        await manager.update_persona("voice", voice_prompt=None)
        assert (await db.get_persona_by_id("voice")).voice_prompt is None
    finally:
        await db.engine.dispose()
