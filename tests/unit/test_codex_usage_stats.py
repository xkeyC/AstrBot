"""Codex runs feed AstrBot's stats: per-turn usage, TTFT and the stats rows."""

import asyncio
import json

import pytest
from sqlmodel import select

from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.runners.codex import codex_agent_runner as runner_mod
from astrbot.core.agent.runners.codex.codex_agent_runner import CodexAgentRunner
from astrbot.core.agent.runners.codex.usage import to_token_usage, usage_between
from astrbot.core.db.po import ProviderStat
from astrbot.core.provider.entities import ProviderRequest, TokenUsage

UMO = "qq:GroupMessage:g1"


def _total(input_tokens, cached, output):
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output,
        "reasoning_output_tokens": 0,
        "total_tokens": input_tokens + output,
    }


# ------------------------------------------------------------ usage maths


def test_codex_input_includes_cached_tokens():
    usage = to_token_usage(_total(1000, 800, 50))

    assert usage == TokenUsage(input_other=200, input_cached=800, output=50)


def test_turn_usage_is_the_growth_of_the_thread_total():
    before = {"total_token_usage": _total(1000, 800, 50)}
    after = {"total_token_usage": _total(3500, 3000, 170)}

    assert usage_between(before, after) == TokenUsage(
        input_other=300, input_cached=2200, output=120
    )


def test_a_new_thread_counts_from_zero():
    after = {"total_token_usage": _total(900, 0, 40)}

    assert usage_between(None, after) == TokenUsage(input_other=900, output=40)


def test_unknown_usage_is_none():
    assert usage_between(None, None) is None
    assert usage_between(None, {"total_token_usage": None}) is None


# ------------------------------------------------------------ a whole turn


class _Rt:
    def __init__(self, totals):
        self.totals = list(totals)

    async def thread_usage(self, thread_id):
        return json.dumps(
            {
                "model": "gpt-5.5",
                "model_provider": "openai",
                "total_token_usage": self.totals.pop(0),
            }
        )


class _Pump:
    def __init__(self, events):
        self.events = events

    def open_turn(self, on_tool, on_approval):
        queue = asyncio.Queue()
        for event in self.events:
            queue.put_nowait(event)
        return queue

    def close_turn(self):
        pass


class _Engine:
    def __init__(self, events, totals):
        self.rt = _Rt(totals)
        self._pump = _Pump(events)
        self.saved_image_handlers = {}
        self.lock = asyncio.Lock()

    def session_lock(self, umo):
        return self.lock

    def pump(self, thread_id):
        return self._pump

    async def submit_turn(self, thread_id, request):
        return {"status": "started", "turn_id": "turn-1"}


async def _run(monkeypatch, temp_db, events, totals):
    engine = _Engine(events, totals)

    async def get(options):
        return engine

    async def open_thread(self, engine):
        return "thread-1", None

    async def no_history(self, text):
        pass

    monkeypatch.setattr(runner_mod.CodexEngine, "get", staticmethod(get))
    monkeypatch.setattr(CodexAgentRunner, "_open_thread", open_thread)
    monkeypatch.setattr(CodexAgentRunner, "_sync_history", no_history)
    monkeypatch.setattr(runner_mod, "db_helper", temp_db)

    class _Ctx:
        event = None

    class _Wrapper:
        context = _Ctx()

    req = ProviderRequest(prompt="hi", session_id=UMO)
    runner = CodexAgentRunner()
    await runner.reset(
        request=req,
        run_context=_Wrapper(),
        agent_hooks=BaseAgentRunHooks(),
        provider_config={},
    )
    responses = [r async for r in runner.step_until_done()]
    async with temp_db.get_db() as session:
        rows = (await session.execute(select(ProviderStat))).scalars().all()
    return runner, responses, rows


@pytest.mark.asyncio
async def test_a_turn_is_recorded_for_the_stats_page(monkeypatch, temp_db):
    # token_count repeats itself (rate-limit refreshes); only the thread total
    # tells what the turn spent.
    count = {
        "type": "token_count",
        "info": {
            "total_token_usage": _total(3500, 3000, 170),
            "last_token_usage": _total(2000, 1900, 70),
        },
    }
    events = [
        {"type": "agent_message_content_delta", "item_id": "m", "delta": "hi"},
        count,
        count,
        {"type": "agent_message", "message": "hello", "phase": "final_answer"},
        {"type": "task_complete", "time_to_first_token_ms": 850},
    ]
    runner, responses, rows = await _run(
        monkeypatch,
        temp_db,
        events,
        [_total(1000, 800, 50), _total(3500, 3000, 170)],
    )

    assert [r.type for r in responses] == ["agent_stats", "llm_result"]
    [row] = rows
    assert row.agent_type == "codex"
    assert row.status == "completed"
    assert row.umo == UMO
    assert row.provider_id == "openai"
    assert row.provider_model == "gpt-5.5"
    assert (row.token_input_other, row.token_input_cached, row.token_output) == (
        300,
        2200,
        120,
    )
    # Codex's own measurement wins over the host-side one.
    assert row.time_to_first_token == pytest.approx(0.85)
    assert row.end_time >= row.start_time > 0

    stats = responses[0].data["chain"].chain[0].data
    assert stats["token_usage"] == {
        "input_other": 300,
        "input_cached": 2200,
        "output": 120,
    }
    assert stats["current_context_tokens"] == 2000


@pytest.mark.asyncio
async def test_without_codex_ttft_the_first_output_is_timed(monkeypatch, temp_db):
    events = [
        {"type": "agent_message", "message": "hello", "phase": "final_answer"},
        {"type": "task_complete"},
    ]
    _, _, rows = await _run(monkeypatch, temp_db, events, [None, _total(100, 0, 10)])

    [row] = rows
    assert 0 < row.time_to_first_token < 5
    assert (row.token_input_other, row.token_output) == (100, 10)


@pytest.mark.asyncio
async def test_an_aborted_turn_is_recorded_as_aborted(monkeypatch, temp_db):
    events = [{"type": "turn_aborted"}]
    _, _, rows = await _run(
        monkeypatch, temp_db, events, [_total(10, 0, 1), _total(10, 0, 1)]
    )

    [row] = rows
    assert row.status == "aborted"
