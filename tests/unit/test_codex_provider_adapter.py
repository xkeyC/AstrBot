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
