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
