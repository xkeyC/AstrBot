from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.core.updater import AstrBotUpdater


class AdminCommands:
    def __init__(self, context: star.Context) -> None:
        self.context = context

    async def update_dashboard(self, event: AstrMessageEvent) -> None:
        """更新管理面板"""
        await event.send(MessageChain().message("⏳ Updating dashboard..."))
        await AstrBotUpdater().ensure_dashboard()
        await event.send(MessageChain().message("✅ Dashboard updated successfully."))

    async def native_exec(self, event: AstrMessageEvent, mode: str = "") -> None:
        """Codex 原生命令执行的会话开关（K3）。切换不改变工具集，上下文与缓存不受影响。"""
        from astrbot.core import sp
        from astrbot.core.agent.runners.codex.constants import NATIVE_EXEC_SESSION_KEY

        umo = event.unified_msg_origin
        mode = mode.strip().lower()
        if mode in ("on", "off"):
            await sp.put_async(
                scope="umo",
                scope_id=umo,
                key=NATIVE_EXEC_SESSION_KEY,
                value=mode == "on",
            )
        elif mode in ("default", "reset"):
            await sp.remove_async(
                scope="umo", scope_id=umo, key=NATIVE_EXEC_SESSION_KEY
            )
        elif mode:
            await event.send(
                MessageChain().message("用法：/native_exec [on|off|default]")
            )
            return
        value = await sp.get_async(
            scope="umo", scope_id=umo, key=NATIVE_EXEC_SESSION_KEY, default=None
        )
        runner_cfg = (
            self.context.get_config(umo=umo).get("agent_runner", {}).get("config", {})
        )
        enabled = bool(runner_cfg.get("native_exec_tools"))
        state = {True: "开启", False: "关闭", None: "跟随全局"}[
            value if isinstance(value, bool) else None
        ]
        note = "" if enabled else "（全局未启用原生执行工具，本开关暂不生效）"
        await event.send(MessageChain().message(f"本会话原生命令执行：{state}{note}"))
