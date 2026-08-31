"""Tests for how the internal agent stage resolves context-window settings."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)


async def _build_stage(**provider_settings) -> InternalAgentSubStage:
    """Initialize the stage with the default config plus the given overrides.

    Args:
        **provider_settings: Values overriding ``provider_settings`` defaults.

    Returns:
        The initialized stage.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["provider_settings"].update(provider_settings)
    plugin_context = MagicMock()
    plugin_context.get_config.return_value = config
    ctx = SimpleNamespace(
        astrbot_config=config,
        plugin_manager=SimpleNamespace(context=plugin_context),
    )
    stage = InternalAgentSubStage()
    await stage.initialize(ctx)
    return stage


@pytest.mark.asyncio
async def test_default_drops_a_whole_chunk_of_turns():
    """A larger default keeps the prompt-cache prefix stable for many turns."""
    stage = await _build_stage()

    assert DEFAULT_CONFIG["provider_settings"]["dequeue_context_length"] == 10
    assert stage.dequeue_context_length == 10
    assert stage.main_agent_cfg.dequeue_context_length == 10


@pytest.mark.asyncio
async def test_unlimited_turns_keeps_the_configured_dequeue_length():
    """Turn-based limiting off must not force the value back down to one turn."""
    stage = await _build_stage(max_context_length=-1, dequeue_context_length=20)

    assert stage.dequeue_context_length == 20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_context_length", "configured", "expected"),
    [
        (50, 10, 10),
        (5, 10, 4),
        (-1, 0, 1),
        (-1, 1, 1),
    ],
)
async def test_dequeue_length_never_empties_a_limited_context(
    max_context_length, configured, expected
):
    stage = await _build_stage(
        max_context_length=max_context_length,
        dequeue_context_length=configured,
    )

    assert stage.dequeue_context_length == expected
