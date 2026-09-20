"""分段回复中途发送失败时，要让用户知道回复可能不完整。

背景：`RespondStage` 逐段发送，每段各自 try/except，失败只进日志。于是
群里可能收到一条中间缺了一块的回复，而没有任何提示——那时 turn 已经结束，
agent 也不会知道。
"""

from unittest.mock import AsyncMock

import pytest

from astrbot.core.pipeline.respond.stage import RespondStage


def _sent_texts(send):
    return [call.args[0].get_plain_text() for call in send.call_args_list]


@pytest.mark.asyncio
async def test_no_notice_when_everything_was_sent():
    event = AsyncMock()

    await RespondStage._report_send_failures(RespondStage(), event, 0)

    event.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_notice_naming_how_many_segments_failed():
    event = AsyncMock()

    await RespondStage._report_send_failures(RespondStage(), event, 2)

    event.send.assert_awaited_once()
    text = _sent_texts(event.send)[0]
    assert "2" in text
    # 措辞必须留有余地：最常见的失败是协议端发送超时，消息往往其实送到了。
    assert "可能" in text


@pytest.mark.asyncio
async def test_a_failing_notice_does_not_raise():
    """通道本来就在出问题，提示发不出去也不该把 pipeline 带崩。"""
    event = AsyncMock()
    event.send.side_effect = RuntimeError("channel down")

    await RespondStage._report_send_failures(RespondStage(), event, 1)
