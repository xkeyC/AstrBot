"""Codex account and model management for the WebUI (codex-native branch)."""

from __future__ import annotations

import json

from astrbot.core.core_lifecycle import AstrBotCoreLifecycle


class CodexServiceError(Exception):
    pass


class CodexService:
    def __init__(self, core_lifecycle: AstrBotCoreLifecycle) -> None:
        self.core_lifecycle = core_lifecycle

    def runner_config(self) -> dict:
        from astrbot.core.config.agent_runner import normalize_agent_runner

        return normalize_agent_runner(
            self.core_lifecycle.astrbot_config.get("agent_runner")
        )["config"]

    async def _engine(self):
        from astrbot.core.agent.runners.codex.codex_agent_runner import engine_options
        from astrbot.core.agent.runners.codex.native import (
            CodexEngine,
            CodexEngineError,
        )

        try:
            return await CodexEngine.get(engine_options(self.runner_config()))
        except CodexEngineError as exc:
            raise CodexServiceError(str(exc)) from exc

    async def _call(self, name: str, *args):
        engine = await self._engine()
        try:
            return await getattr(engine.rt, name)(*args)
        except AttributeError as exc:
            raise CodexServiceError(
                "codex_astrbot binding is outdated; rebuild it with maturin"
            ) from exc
        except RuntimeError as exc:
            raise CodexServiceError(str(exc)) from exc

    async def account(self) -> dict:
        status = json.loads(await self._call("account_status"))
        cfg = self.runner_config()
        status["model"] = cfg.get("model") or ""
        status["model_provider"] = cfg.get("model_provider") or ""
        return status

    async def login_api_key(self, api_key: str) -> None:
        if not api_key.strip():
            raise CodexServiceError("API Key 不能为空")
        await self._call("login_api_key", api_key.strip())

    async def start_device_login(self) -> dict:
        return json.loads(await self._call("start_device_login"))

    async def device_login_status(self, login_id: str) -> dict:
        return json.loads(await self._call("device_login_status", login_id))

    async def cancel_device_login(self, login_id: str) -> bool:
        return bool(await self._call("cancel_device_login", login_id))

    async def logout(self) -> bool:
        return bool(await self._call("logout"))

    async def models(self, include_hidden: bool = False) -> list[dict]:
        return json.loads(await self._call("list_models", include_hidden))
