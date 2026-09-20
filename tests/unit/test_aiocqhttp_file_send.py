"""测试 aiocqhttp 平台发送文件段时对"协议端读不到本机路径"的处理。

Bug 背景：NapCat 等协议端常与 AstrBot 分处不同容器/主机，AstrBot 把
沙箱里下载到 data/temp 的文件按本机路径发出去，协议端会以
ActionFailed retcode=1200 ENOENT 失败：

    message="ENOENT: no such file or directory,
             open '/home/astrbot/data/temp/sandbox_xxxx_report.html'"

修复：仅在协议端明确报告文件不存在时，把文件内联成 base64 重发一次；
发送超时之类的错误不重试，以免重复发送。
"""

import base64
from unittest.mock import AsyncMock

import pytest

import astrbot.core.message.components as Comp
import astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event as mod
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)

FILE_BODY = b"<html>sandbox report</html>"


class _FakeBot:
    """只用来当 _dispatch_send 的 bot 参数，以及做 id() 区分。"""


@pytest.fixture
def local_file(tmp_path):
    path = tmp_path / "sandbox_f292_report.html"
    path.write_bytes(FILE_BODY)
    return str(path)


@pytest.fixture(autouse=True)
def _no_file_service(monkeypatch):
    """不配置 callback_api_base，走本机路径分支。"""
    monkeypatch.setitem(Comp.astrbot_config, "callback_api_base", "")
    mod._PATH_SEND_UNSUPPORTED.clear()
    yield
    mod._PATH_SEND_UNSUPPORTED.clear()


async def _send(bot, local_file, dispatch):
    chain = MessageChain([Comp.File(name="report.html", file=local_file)])
    await AiocqhttpMessageEvent.send_message(
        bot=bot,
        message_chain=chain,
        event=None,
        is_group=True,
        session_id="12345",
    )
    return dispatch


def _sent_file_value(dispatch, call_index=0):
    messages = dispatch.call_args_list[call_index].args[4]
    return messages[0]["data"]["file"]


@pytest.mark.asyncio
async def test_path_send_succeeds_without_base64(monkeypatch, local_file):
    """协议端能读到本机文件时，按路径发送，不做 base64 内联。"""
    dispatch = AsyncMock()
    monkeypatch.setattr(AiocqhttpMessageEvent, "_dispatch_send", dispatch)

    await _send(_FakeBot(), local_file, dispatch)

    assert dispatch.await_count == 1
    value = _sent_file_value(dispatch)
    assert value.startswith("file://")
    assert "base64://" not in value


@pytest.mark.asyncio
async def test_enoent_falls_back_to_base64(monkeypatch, local_file):
    """协议端报 ENOENT 时，把文件内联成 base64 重发一次。"""
    dispatch = AsyncMock(
        side_effect=[
            RuntimeError(
                "ActionFailed retcode=1200 "
                f"message=\"ENOENT: no such file or directory, open '{local_file}'\""
            ),
            None,
        ]
    )
    monkeypatch.setattr(AiocqhttpMessageEvent, "_dispatch_send", dispatch)

    await _send(_FakeBot(), local_file, dispatch)

    assert dispatch.await_count == 2
    value = _sent_file_value(dispatch, 1)
    assert value.startswith("base64://")
    assert base64.b64decode(value.removeprefix("base64://")) == FILE_BODY


@pytest.mark.asyncio
async def test_timeout_is_not_retried(monkeypatch, local_file):
    """发送超时不重试：消息可能已经送达，重发会变成重复消息。"""
    dispatch = AsyncMock(
        side_effect=RuntimeError(
            "ActionFailed retcode=1200 message='Timeout: NTEvent "
            "serviceAndMethod:NodeIKernelMsgService/sendMsg'"
        )
    )
    monkeypatch.setattr(AiocqhttpMessageEvent, "_dispatch_send", dispatch)

    with pytest.raises(RuntimeError):
        await _send(_FakeBot(), local_file, dispatch)

    assert dispatch.await_count == 1


@pytest.mark.asyncio
async def test_connection_remembers_base64_mode(monkeypatch, local_file):
    """同一个连接失败过一次后，后续文件直接内联，不再白发一次。"""
    bot = _FakeBot()
    dispatch = AsyncMock(
        side_effect=[
            RuntimeError("ENOENT: no such file or directory"),
            None,
        ]
    )
    monkeypatch.setattr(AiocqhttpMessageEvent, "_dispatch_send", dispatch)
    await _send(bot, local_file, dispatch)
    assert dispatch.await_count == 2

    dispatch.reset_mock()
    dispatch.side_effect = None
    await _send(bot, local_file, dispatch)

    assert dispatch.await_count == 1
    assert _sent_file_value(dispatch).startswith("base64://")


@pytest.mark.asyncio
async def test_oversized_file_keeps_original_failure(monkeypatch, local_file):
    """超过内联上限时不硬塞 base64，保留可读的报错。"""
    monkeypatch.setattr(mod, "FILE_BASE64_LIMIT_BYTES", 1)
    dispatch = AsyncMock(side_effect=RuntimeError("ENOENT: no such file or directory"))
    monkeypatch.setattr(AiocqhttpMessageEvent, "_dispatch_send", dispatch)

    with pytest.raises(ValueError, match="base64"):
        await _send(_FakeBot(), local_file, dispatch)
