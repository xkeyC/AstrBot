from astrbot.core.agent.message import AssistantMessageSegment, ToolCallMessageSegment
from astrbot.core.agent.runners.codex.provider_adapter import (
    _tool_results,
    render_transcript,
)
from astrbot.core.provider.entities import ToolCallsResult


def test_render_transcript_roles():
    system, transcript = render_transcript(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "weather"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ]
    )
    assert system == "be brief"
    assert "User: hi" in transcript
    assert "called tools: weather" in transcript
    assert "Tool result: sunny" in transcript
    assert transcript.endswith("Assistant: It is sunny.")


def test_tool_results_from_both_sources():
    result = ToolCallsResult(
        tool_calls_info=AssistantMessageSegment(content=None, tool_calls=[]),
        tool_calls_result=[ToolCallMessageSegment(tool_call_id="a", content="1")],
    )
    out = _tool_results(result, [{"role": "tool", "tool_call_id": "b", "content": "2"}])
    assert out == {"a": "1", "b": "2"}


def test_cron_and_background_prompts_quote_origin():
    from astrbot.core.agent.runners.codex.wake import (
        build_background_prompt,
        build_cron_prompt,
    )

    cron = build_cron_prompt(
        {"name": "daily", "run_started_at": "t"},
        {"note": "send the weather", "origin_message": "every day at 8 send weather"},
    )
    assert "> every day at 8 send weather" in cron
    assert "Do not create, change or cancel scheduled tasks" in cron
    assert cron.strip().endswith("</scheduled_task>")
    bg = build_background_prompt(
        {"tool_name": "render", "task_id": "1", "result": "done"}, "render my video"
    )
    assert "> render my video" in bg and "Result:\ndone" in bg
