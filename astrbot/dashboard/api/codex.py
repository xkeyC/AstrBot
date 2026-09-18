"""Codex account/model routes for the WebUI (codex-native branch).

Dashboard-only (not part of the public OpenAPI surface).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from astrbot.dashboard.responses import ApiError, ok
from astrbot.dashboard.services.codex_service import CodexService, CodexServiceError

from .auth import require_dashboard_user

legacy_router = APIRouter(
    prefix="/api/codex",
    tags=["Dashboard Codex"],
    include_in_schema=False,
)


class ApiKeyLogin(BaseModel):
    api_key: str


def get_service(request: Request) -> CodexService:
    return request.app.state.services.codex


async def _run(coro):
    try:
        return await coro
    except CodexServiceError as exc:
        raise ApiError(str(exc)) from exc


@legacy_router.get("/account")
async def get_account(
    _username: str = Depends(require_dashboard_user),
    service: CodexService = Depends(get_service),
):
    return ok(await _run(service.account()))


@legacy_router.post("/login/api-key")
async def login_api_key(
    payload: ApiKeyLogin,
    _username: str = Depends(require_dashboard_user),
    service: CodexService = Depends(get_service),
):
    await _run(service.login_api_key(payload.api_key))
    return ok(await _run(service.account()), message="登录成功")


@legacy_router.post("/login/device")
async def start_device_login(
    _username: str = Depends(require_dashboard_user),
    service: CodexService = Depends(get_service),
):
    return ok(await _run(service.start_device_login()))


@legacy_router.get("/login/device/{login_id}")
async def device_login_status(
    login_id: str,
    _username: str = Depends(require_dashboard_user),
    service: CodexService = Depends(get_service),
):
    return ok(await _run(service.device_login_status(login_id)))


@legacy_router.delete("/login/device/{login_id}")
async def cancel_device_login(
    login_id: str,
    _username: str = Depends(require_dashboard_user),
    service: CodexService = Depends(get_service),
):
    return ok({"cancelled": await _run(service.cancel_device_login(login_id))})


@legacy_router.post("/logout")
async def logout(
    _username: str = Depends(require_dashboard_user),
    service: CodexService = Depends(get_service),
):
    return ok({"logged_out": await _run(service.logout())})


@legacy_router.get("/models")
async def list_models(
    include_hidden: bool = False,
    _username: str = Depends(require_dashboard_user),
    service: CodexService = Depends(get_service),
):
    return ok(await _run(service.models(include_hidden)))
