import asyncio
from pathlib import Path
from types import SimpleNamespace

from astrbot.core.tools.computer_tools import fs as fs_tools
from astrbot.core.tools.computer_tools.util import reject_secret_path
from astrbot.core.utils.astrbot_path import (
    get_astrbot_config_path,
    get_astrbot_data_path,
)


def test_credential_paths_are_refused():
    data = Path(get_astrbot_data_path())
    refused = [
        Path(get_astrbot_config_path()) / "astrbot_plugin_x_config.json",
        data / "cmd_config.json",
        data / "codex_home" / "auth.json",
        data / "data_v4.db",
        data / "backups" / "dump.zip",
        Path.home() / ".ssh" / "id_rsa",
        Path.home() / ".aws" / "credentials",
        Path("/srv/app/.env"),
        Path("/srv/app/server.pem"),
    ]
    for path in refused:
        assert reject_secret_path(path), path

    allowed = [
        data / "temp" / "sandbox_ab12_report.csv",
        data / "attachments" / "photo.png",
        Path.home() / "notes.md",
        Path("/srv/app/main.py"),
    ]
    for path in allowed:
        assert reject_secret_path(path) is None, path


def test_upload_tool_refuses_credentials(monkeypatch):
    event = SimpleNamespace(unified_msg_origin="qq:FriendMessage:1", role="admin")
    ctx = SimpleNamespace(context=SimpleNamespace(event=event, context=None))
    monkeypatch.setattr(fs_tools, "check_admin_permission", lambda *_: None)

    async def unreachable(*_args, **_kwargs):
        raise AssertionError("the sandbox must not be reached for a refused path")

    monkeypatch.setattr(fs_tools, "get_booter", unreachable)
    secret = str(Path(get_astrbot_data_path()) / "cmd_config.json")
    result = asyncio.run(fs_tools.FileUploadTool().call(ctx, secret))
    assert "credentials" in result or "credential file" in result


def test_send_tool_refuses_credential_paths(monkeypatch):
    from astrbot.core.tools import message_tools

    event = SimpleNamespace(unified_msg_origin="qq:FriendMessage:1", role="admin")
    ctx = SimpleNamespace(context=SimpleNamespace(event=event, context=None))
    # Local runtime + admin: anything else on this host is sendable.
    monkeypatch.setattr(message_tools, "is_local_runtime", lambda _: True)
    monkeypatch.setattr(message_tools, "_is_restricted_local_env", lambda _: False)

    secret = Path(get_astrbot_data_path()) / "cmd_config.json"
    assert message_tools._can_send_local_file(ctx, secret) is False
    assert (
        message_tools._can_send_local_file(ctx, Path.home() / ".ssh" / "id_rsa")
        is False
    )
    assert message_tools._can_send_local_file(ctx, Path.home() / "poster.png") is True
