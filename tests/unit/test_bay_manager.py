"""Tests for Shipyard Neo Bay container manager."""

from astrbot.core.computer.booters.bay_manager import BayContainerManager

PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


def clear_proxy_env(monkeypatch):
    for key in PROXY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_build_proxy_env_from_astrbot_process_env(monkeypatch):
    clear_proxy_env(monkeypatch)
    monkeypatch.setenv("http_proxy", "http://proxy:7890")
    monkeypatch.setenv("https_proxy", "http://proxy:7890")
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")

    env = BayContainerManager()._build_proxy_env()

    assert "http_proxy=http://proxy:7890" in env
    assert "https_proxy=http://proxy:7890" in env
    assert "no_proxy=localhost,127.0.0.1" in env
    assert "BAY_PROXY__ENABLED=true" in env
    assert "BAY_PROXY__HTTP_PROXY=http://proxy:7890" in env
    assert "BAY_PROXY__HTTPS_PROXY=http://proxy:7890" in env
    assert "BAY_PROXY__NO_PROXY=localhost,127.0.0.1" in env


def test_build_proxy_env_empty_without_proxy(monkeypatch):
    clear_proxy_env(monkeypatch)

    assert BayContainerManager()._build_proxy_env() == []


def test_proxy_env_mismatch_when_existing_container_has_old_proxy(monkeypatch):
    clear_proxy_env(monkeypatch)
    monkeypatch.setenv("http_proxy", "http://new-proxy:7890")

    manager = BayContainerManager()
    existing = {
        "Config": {
            "Env": [
                "http_proxy=http://old-proxy:7890",
                "BAY_PROXY__ENABLED=true",
                "BAY_PROXY__HTTP_PROXY=http://old-proxy:7890",
            ]
        }
    }

    assert manager._proxy_env_matches(existing, manager._build_bay_env()) is False
