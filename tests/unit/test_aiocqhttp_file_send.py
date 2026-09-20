"""测试 aiocqhttp 平台发送文件段时对"协议端读不到本机路径"的处理。

Bug 背景：NapCat 等协议端常与 AstrBot 分处不同容器/主机，AstrBot 把
沙箱里下载到 data/temp 的文件按本机路径发出去，协议端会以
ActionFailed retcode=1200 ENOENT 失败：

    message="ENOENT: no such file or directory,
             open '/home/astrbot/data/temp/sandbox_xxxx_report.html'"

处理：不重发（消息可能已送达，重发会变成重复消息），而是把失败翻译成
一条可执行的指引，顺着工具返回值回到 agent——改用平台自带的上传工具。
"""

from unittest.mock import AsyncMock

import pytest

import astrbot.core.message.components as Comp
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)


class _FakeBot:
    """只用来当 _dispatch_send 的 bot 参数。"""


@pytest.fixture
def local_file(tmp_path):
    path = tmp_path / "sandbox_f292_report.html"
    path.write_bytes(b"<html>sandbox report</html>")
    return str(path)


@pytest.fixture(autouse=True)
def _no_file_service(monkeypatch):
    """不配置 callback_api_base，走本机路径分支。"""
    monkeypatch.setitem(Comp.astrbot_config, "callback_api_base", "")


async def _send(bot, local_file):
    chain = MessageChain([Comp.File(name="report.html", file=local_file)])
    await AiocqhttpMessageEvent.send_message(
        bot=bot,
        message_chain=chain,
        event=None,
        is_group=True,
        session_id="12345",
    )


@pytest.mark.asyncio
async def test_path_send_is_used_as_is(monkeypatch, local_file):
    """协议端能读到本机文件时，按路径发送。"""
    dispatch = AsyncMock()
    monkeypatch.setattr(AiocqhttpMessageEvent, "_dispatch_send", dispatch)

    await _send(_FakeBot(), local_file)

    assert dispatch.await_count == 1
    value = dispatch.call_args_list[0].args[4][0]["data"]["file"]
    assert value.startswith("file://")
    assert "base64://" not in value


@pytest.mark.asyncio
async def test_enoent_points_agent_at_platform_tool(monkeypatch, local_file):
    """协议端报 ENOENT 时不重发，改成告诉 agent 去用平台的上传工具。"""
    dispatch = AsyncMock(
        side_effect=RuntimeError(
            "ActionFailed retcode=1200 "
            f"message=\"ENOENT: no such file or directory, open '{local_file}'\""
        )
    )
    monkeypatch.setattr(AiocqhttpMessageEvent, "_dispatch_send", dispatch)

    with pytest.raises(RuntimeError) as excinfo:
        await _send(_FakeBot(), local_file)

    # 只发一次，不重试
    assert dispatch.await_count == 1
    message = str(excinfo.value)
    # 指引必须可执行：说清楚别重试、去哪个命名空间找、找什么能力。
    assert "Do not retry" in message
    assert "`qq_`" in message
    assert "upload" in message
    assert "callback_api_base" in message
    # 原始失败仍然挂在 __cause__ 上，排查时不丢信息。
    assert "ENOENT" in str(excinfo.value.__cause__)


@pytest.mark.asyncio
async def test_timeout_is_passed_through_untouched(monkeypatch, local_file):
    """发送超时原样抛出：消息可能已经送达，不该被当成路径问题。"""
    original = RuntimeError(
        "ActionFailed retcode=1200 message='Timeout: NTEvent "
        "serviceAndMethod:NodeIKernelMsgService/sendMsg'"
    )
    dispatch = AsyncMock(side_effect=original)
    monkeypatch.setattr(AiocqhttpMessageEvent, "_dispatch_send", dispatch)

    with pytest.raises(RuntimeError) as excinfo:
        await _send(_FakeBot(), local_file)

    assert excinfo.value is original
    assert dispatch.await_count == 1


@pytest.mark.asyncio
async def test_enoent_for_a_file_we_do_not_have_is_passed_through(monkeypatch):
    """本机也没有这个文件时，不该冒充成"文件系统不共享"。"""
    original = RuntimeError("ENOENT: no such file or directory")
    dispatch = AsyncMock(side_effect=original)
    monkeypatch.setattr(AiocqhttpMessageEvent, "_dispatch_send", dispatch)

    with pytest.raises(RuntimeError) as excinfo:
        await _send(_FakeBot(), "/nowhere/missing.html")

    assert excinfo.value is original
