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
    # The internal stage still consumes the legacy local runner settings.
    config["agent_runner"] = {
        "runner_type": "local",
        "config": copy.deepcopy(AGENT_RUNNER_CONFIG_DEFAULTS["local"]),
    }
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
    assert stage.main_agent_cfg.dequeue_context_length == 10


@pytest.mark.asyncio
async def test_unlimited_turns_keeps_the_configured_trim_length():
    """Turn-based limiting off must not force the value back down to one turn."""
    stage = await _build_stage({"max_turns": -1, "trim_turns": 20})

    assert stage.main_agent_cfg.dequeue_context_length == 20


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

    assert stage.main_agent_cfg.max_context_length == max_turns
    assert stage.main_agent_cfg.dequeue_context_length == expected


@pytest.mark.asyncio
async def test_search_registry_tool_schema_mode_is_accepted():
    stage = await _build_stage(misc={"tool_schema_mode": "search_registry"})

    assert stage.tool_schema_mode == "search_registry"
    assert stage.main_agent_cfg.tool_schema_mode == "search_registry"


@pytest.mark.asyncio
async def test_compression_and_misc_settings_reach_the_build_config():
    """Every configured session setting must reach the main agent build config."""
    stage = await _build_stage(
        {
            "max_turns": 12,
            "trim_turns": 3,
            "overflow_strategy": "llm_compress",
            "instruction": "Keep decisions and unfinished tasks.",
            "keep_recent_ratio": 0.3,
            "provider_id": "summary-model",
            "fallback_max_tokens": 16384,
        },
        {"max_steps": 7, "tool_call_timeout": 45},
    )

    cfg = stage.main_agent_cfg
    assert cfg.max_context_length == 12
    assert cfg.dequeue_context_length == 3
    assert cfg.context_limit_reached_strategy == "llm_compress"
    assert cfg.llm_compress_instruction == "Keep decisions and unfinished tasks."
    assert cfg.llm_compress_keep_recent_ratio == 0.3
    assert cfg.llm_compress_provider_id == "summary-model"
    assert cfg.fallback_max_context_tokens == 16384
    assert cfg.tool_call_timeout == 45
    assert stage.max_step == 7
    assert stage.tool_call_timeout == 45
