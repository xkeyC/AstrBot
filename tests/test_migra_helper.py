from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import astrbot.core.utils.migra_helper as migra_helper
from astrbot.core.utils.migra_helper import (
    _migrate_legacy_openai_responses_sources,
)


class _Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_count = 0

    def save_config(self) -> None:
        self.save_count += 1


def test_migrate_legacy_openai_responses_sources_uses_dedicated_adapter():
    config = _Config(
        provider_sources=[
            {
                "id": "responses-source",
                "type": "openai_chat_completion",
                "api_mode": "responses",
            },
            {
                "id": "chat-source",
                "type": "openai_chat_completion",
                "api_mode": "chat_completions",
            },
        ]
    )

    _migrate_legacy_openai_responses_sources(config)

    assert config["provider_sources"] == [
        {"id": "responses-source", "type": "openai_responses"},
        {"id": "chat-source", "type": "openai_chat_completion"},
    ]
    assert config.save_count == 1


def test_migrate_legacy_openai_responses_sources_is_idempotent():
    config = _Config(
        provider_sources=[
            {"id": "responses-source", "type": "openai_responses"},
        ]
    )

    _migrate_legacy_openai_responses_sources(config)

    assert config.save_count == 0


@pytest.mark.asyncio
async def test_migra_runs_legacy_responses_migration_on_default_config():
    """migra() logs and swallows migration errors, so a broken call is silent."""
    default_conf = _Config()
    acm = SimpleNamespace(confs={}, default_conf=default_conf)

    with (
        patch(
            "astrbot.core.db.migration.migra_45_to_46.migrate_45_to_46",
            AsyncMock(),
        ),
        patch(
            "astrbot.core.db.migration.migra_webchat_session.migrate_webchat_session",
            AsyncMock(),
        ),
        patch(
            "astrbot.core.db.migration.migra_token_usage.migrate_token_usage",
            AsyncMock(),
        ),
        patch.object(migra_helper, "finalize_config_migrations", return_value=False),
        patch.object(migra_helper, "_migra_provider_to_source_structure"),
        patch.object(
            migra_helper, "_migrate_legacy_openai_responses_sources"
        ) as migrate_responses,
        patch.object(migra_helper.logger, "error") as log_error,
    ):
        await migra_helper.migra(None, None, None, acm)

    migrate_responses.assert_called_once_with(default_conf)
    log_error.assert_not_called()
