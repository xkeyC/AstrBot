"""Tests for how the internal agent stage resolves agent runner settings."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from astrbot.core.config.agent_runner import AGENT_RUNNER_CONFIG_DEFAULTS
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)


async def _build_stage(
    compression: dict | None = None,
    misc: dict | None = None,
) -> InternalAgentSubStage:
    """Initialize the stage with the default config plus the given overrides.

    Args:
        compression: Values overriding ``agent_runner.config.compression``.
        misc: Values overriding ``agent_runner.config.misc``.

    Returns:
        The initialized stage.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    runner_config = config["agent_runner"]["config"]
    runner_config["compression"].update(compression or {})
    runner_config["misc"].update(misc or {})
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

    assert AGENT_RUNNER_CONFIG_DEFAULTS["local"]["compression"]["trim_turns"] == 10
    assert stage.dequeue_context_length == 10
    assert stage.main_agent_cfg.dequeue_context_length == 10


@pytest.mark.asyncio
async def test_unlimited_turns_keeps_the_configured_trim_length():
    """Turn-based limiting off must not force the value back down to one turn."""
    stage = await _build_stage({"max_turns": -1, "trim_turns": 20})

    assert stage.dequeue_context_length == 20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_turns", "trim_turns", "expected"),
    [
        (50, 10, 10),
        (5, 10, 4),
        (-1, 0, 1),
        (-1, 1, 1),
    ],
)
async def test_trim_length_never_empties_a_limited_context(
    max_turns, trim_turns, expected
):
    stage = await _build_stage({"max_turns": max_turns, "trim_turns": trim_turns})

    assert stage.dequeue_context_length == expected


@pytest.mark.asyncio
async def test_search_registry_tool_schema_mode_is_accepted():
    stage = await _build_stage(misc={"tool_schema_mode": "search_registry"})

    assert stage.tool_schema_mode == "search_registry"
    assert stage.main_agent_cfg.tool_schema_mode == "search_registry"
