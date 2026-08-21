from pathlib import Path

import pytest

from astrbot.cli.utils import basic
from astrbot.core.config.default import VERSION


def _write_dashboard(dist: Path, version: str) -> None:
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("dashboard", encoding="utf-8")
    (dist / "assets" / "version").write_text(version, encoding="utf-8")


@pytest.mark.asyncio
async def test_check_dashboard_skips_download_for_compatible_bundle(
    monkeypatch,
    tmp_path,
):
    bundled_dist = tmp_path / "bundled"
    _write_dashboard(bundled_dist, f"v{VERSION}")

    class FailingUpdater:
        def __init__(self):
            pytest.fail("compatible bundled assets should not trigger a download")

    monkeypatch.setattr(basic, "_BUNDLED_DIST", bundled_dist)
    monkeypatch.setattr("astrbot.core.updater.AstrBotUpdater", FailingUpdater)

    await basic.check_dashboard(tmp_path)


@pytest.mark.asyncio
async def test_check_dashboard_downloads_when_bundle_is_incomplete(
    monkeypatch,
    tmp_path,
):
    bundled_dist = tmp_path / "bundled"
    bundled_dist.mkdir()
    called = False

    class FakeUpdater:
        async def ensure_dashboard(self):
            nonlocal called
            called = True
            return tmp_path / "data" / "dist"

    monkeypatch.setattr(basic, "_BUNDLED_DIST", bundled_dist)
    monkeypatch.setattr("astrbot.core.updater.AstrBotUpdater", FakeUpdater)

    await basic.check_dashboard(tmp_path)

    assert called
