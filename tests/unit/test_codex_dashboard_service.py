import asyncio
import json
from types import SimpleNamespace

import pytest

from astrbot.core.agent.runners.codex import native
from astrbot.dashboard.services.codex_service import CodexService, CodexServiceError


class FakeRuntime:
    def __init__(self):
        self.keys = []

    async def account_status(self):
        return json.dumps({"logged_in": bool(self.keys), "mode": "apikey"})

    async def login_api_key(self, key):
        self.keys.append(key)

    async def list_models(self, include_hidden):
        return json.dumps([{"model": "gpt-x", "hidden": include_hidden}])


def _service(monkeypatch, rt):
    async def fake_get(options):
        return SimpleNamespace(rt=rt)

    monkeypatch.setattr(native.CodexEngine, "get", staticmethod(fake_get))
    lifecycle = SimpleNamespace(
        astrbot_config={
            "agent_runner": {"runner_type": "codex", "config": {"model": "gpt-x"}}
        }
    )
    return CodexService(lifecycle)


def test_account_login_and_models(monkeypatch):
    rt = FakeRuntime()
    service = _service(monkeypatch, rt)
    assert asyncio.run(service.account())["logged_in"] is False
    asyncio.run(service.login_api_key("  sk-test "))
    assert rt.keys == ["sk-test"]
    account = asyncio.run(service.account())
    assert account["logged_in"] is True and account["model"] == "gpt-x"
    assert asyncio.run(service.models(True)) == [{"model": "gpt-x", "hidden": True}]


def test_errors_are_service_errors(monkeypatch):
    service = _service(monkeypatch, FakeRuntime())
    with pytest.raises(CodexServiceError):
        asyncio.run(service.login_api_key("   "))
    # Binding without the account API (outdated build).
    with pytest.raises(CodexServiceError, match="outdated"):
        asyncio.run(service.logout())
