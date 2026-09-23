"""Changing Codex settings restarts the engine instead of stacking a new one."""

from types import SimpleNamespace

import pytest

from astrbot.core.agent.runners.codex import native
from astrbot.core.agent.runners.codex.native import CodexEngine

pytestmark = pytest.mark.real_codex_engine


class _Rt:
    def __init__(self, options):
        self.options = options
        self.shut_down = False

    async def shutdown(self):
        self.shut_down = True

    def set_saved_image_hook(self, hook):
        pass


@pytest.fixture
def binding(monkeypatch, tmp_path):
    created = []

    class Runtime:
        @staticmethod
        async def create(options_json):
            rt = _Rt(options_json)
            created.append(rt)
            return rt

    monkeypatch.setattr(
        native, "_import_binding", lambda: SimpleNamespace(Runtime=Runtime)
    )
    monkeypatch.setattr(CodexEngine, "_instances", {})
    monkeypatch.setattr(CodexEngine, "_lock", None)
    return created


def _options(tmp_path, model):
    return {"codex_home": str(tmp_path), "config": {"model": model}}


@pytest.mark.asyncio
async def test_the_same_settings_reuse_one_engine(binding, tmp_path):
    first = await CodexEngine.get(_options(tmp_path, "gpt-5.5"))
    again = await CodexEngine.get(_options(tmp_path, "gpt-5.5"))

    assert again is first
    assert len(binding) == 1


@pytest.mark.asyncio
async def test_changed_settings_shut_the_old_engine_down_first(binding, tmp_path):
    first = await CodexEngine.get(_options(tmp_path, "gpt-5.5"))
    second = await CodexEngine.get(_options(tmp_path, "gpt-5.4"))

    assert second is not first
    # Shut down before the new runtime exists, so the chats' rollout files are
    # free and their threads can be resumed rather than started over.
    assert first.rt.shut_down is True
    assert len(binding) == 2
    assert list(CodexEngine._instances.values()) == [second]


@pytest.mark.asyncio
async def test_another_codex_home_is_left_alone(binding, tmp_path):
    other = tmp_path / "other"
    first = await CodexEngine.get(_options(tmp_path, "gpt-5.5"))
    second = await CodexEngine.get(_options(other, "gpt-5.5"))

    assert first.rt.shut_down is False
    assert len(CodexEngine._instances) == 2
    assert {first.codex_home, second.codex_home} == {str(tmp_path), str(other)}
