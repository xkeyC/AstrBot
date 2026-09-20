import asyncio
import base64
import os
import pathlib
import re
from collections.abc import AsyncGenerator

from aiocqhttp import CQHttp, Event

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import (
    At,
    BaseMessageComponent,
    File,
    Image,
    Node,
    Nodes,
    Plain,
    Record,
    Video,
)
from astrbot.api.platform import Group, MessageMember
from astrbot.core.utils.media_utils import file_uri_to_path, is_file_uri

# 文件段改用 base64 重发时的大小上限。base64 会膨胀约 1/3，且要整条塞进
# OneBot 的 WebSocket 帧里，过大的文件应当走 callback_api_base 文件服务。
FILE_BASE64_LIMIT_BYTES = 64 * 1024 * 1024

# 协议端报告"文件不存在"的几种说法。只有命中这些才会重发，避免把发送超时
# 之类的错误也重试一遍导致重复发送。
_MISSING_FILE_HINTS = (
    "enoent",
    "no such file",
    "文件不存在",
    "file not found",
)


def _local_file_path(value: object) -> str | None:
    """把消息段里的 file 字段还原成本机真实存在的路径。

    Args:
        value: file 字段的值，可能是路径、file: URI 或 http(s) 链接。

    Returns:
        本机存在该文件时返回绝对路径，否则返回 None。
    """
    if not isinstance(value, str) or not value:
        return None
    path = file_uri_to_path(value) if is_file_uri(value) else value
    if "://" in path:
        return None
    try:
        return os.path.abspath(path) if os.path.isfile(path) else None
    except OSError:
        return None


# 已经确认读不到本机路径的协议端连接（按 CQHttp 实例区分）。记住之后就直接
# 走 base64，不必每个文件都先失败一次。
_PATH_SEND_UNSUPPORTED: set[int] = set()


def _is_missing_file_error(exc: BaseException) -> bool:
    """判断协议端的报错是不是"它那边看不到这个文件"。"""
    text = f"{exc}".lower()
    return any(hint in text for hint in _MISSING_FILE_HINTS)


class AiocqhttpMessageEvent(AstrMessageEvent):
    def __init__(
        self,
        message_str,
        message_obj,
        platform_meta,
        session_id,
        bot: CQHttp,
    ) -> None:
        super().__init__(message_str, message_obj, platform_meta, session_id)
        self.bot = bot

    @staticmethod
    async def _from_segment_to_dict(
        segment: BaseMessageComponent,
        *,
        file_as_base64: bool = False,
    ) -> dict:
        """修复部分字段

        Args:
            segment: 要转换的消息段。
            file_as_base64: 文件段是否内联为 base64，用于协议端读不到本机
                路径时的重发。
        """
        if isinstance(segment, Image | Record):
            # For Image and Record segments, we convert them to base64
            bs64 = await segment.convert_to_base64()
            return {
                "type": segment.type.lower(),
                "data": {
                    "file": f"base64://{bs64}",
                },
            }
        if isinstance(segment, File):
            # For File segments, we need to handle the file differently
            d = await segment.to_dict()
            file_val = d.get("data", {}).get("file", "")
            if file_val:
                local_path = _local_file_path(file_val)
                if local_path and file_as_base64:
                    payload = await asyncio.to_thread(
                        AiocqhttpMessageEvent._encode_file_base64, local_path
                    )
                    d["data"]["file"] = f"base64://{payload}"
                    return d
                try:
                    # 使用 pathlib 处理路径，能更好地处理 Windows/Linux 差异
                    path_obj = pathlib.Path(file_val)
                    # 如果是绝对路径且不包含协议头 (://)，则转换为标准的 file: URI
                    if path_obj.is_absolute() and "://" not in file_val:
                        d["data"]["file"] = path_obj.as_uri()
                except Exception:
                    # 如果不是合法路径（例如已经是特定的特殊字符串），则跳过转换
                    pass
            return d
        if isinstance(segment, Video):
            d = await segment.to_dict()
            return d
        # For other segments, we simply convert them to a dict by calling toDict
        return segment.toDict()

    @staticmethod
    async def _parse_onebot_json(message_chain: MessageChain):
        """解析成 OneBot json 格式"""
        ret = []
        for segment in message_chain.chain:
            if isinstance(segment, At):
                # At 组件后插入一个空格，避免与后续文本粘连
                d = await AiocqhttpMessageEvent._from_segment_to_dict(segment)
                ret.append(d)
                ret.append({"type": "text", "data": {"text": " "}})
            elif isinstance(segment, Plain):
                if not segment.text.strip():
                    continue
                d = await AiocqhttpMessageEvent._from_segment_to_dict(segment)
                ret.append(d)
            else:
                d = await AiocqhttpMessageEvent._from_segment_to_dict(segment)
                ret.append(d)
        return ret

    @classmethod
    async def _dispatch_send(
        cls,
        bot: CQHttp,
        event: Event | None,
        is_group: bool,
        session_id: str | None,
        messages: list[dict],
    ) -> None:
        # session_id 必须是纯数字字符串
        session_id_int = (
            int(session_id) if session_id and session_id.isdigit() else None
        )
        routing_params = {}
        if isinstance(event, Event) and event.get("self_id"):
            routing_params["self_id"] = event["self_id"]

        if is_group and isinstance(session_id_int, int):
            await bot.send_group_msg(
                group_id=session_id_int,
                message=messages,
                **routing_params,
            )
        elif not is_group and isinstance(session_id_int, int):
            await bot.send_private_msg(
                user_id=session_id_int,
                message=messages,
                **routing_params,
            )
        elif isinstance(event, Event):  # 最后兜底
            await bot.send(event=event, message=messages)
        else:
            raise ValueError(
                f"无法发送消息：缺少有效的数字 session_id({session_id}) 或 event({event})",
            )

    @classmethod
    async def send_message(
        cls,
        bot: CQHttp,
        message_chain: MessageChain,
        event: Event | None = None,
        is_group: bool = False,
        session_id: str | None = None,
    ) -> None:
        """发送消息至 QQ 协议端（aiocqhttp）。

        Args:
            bot (CQHttp): aiocqhttp 机器人实例
            message_chain (MessageChain): 要发送的消息链
            event (Event | None, optional): aiocqhttp 事件对象.
            is_group (bool, optional): 是否为群消息.
            session_id (str | None, optional): 会话 ID（群号或 QQ 号

        """
        # 转发消息、文件消息不能和普通消息混在一起发送
        send_one_by_one = any(
            isinstance(seg, Node | Nodes | File) for seg in message_chain.chain
        )
        if not send_one_by_one:
            ret = await cls._parse_onebot_json(message_chain)
            if not ret:
                return
            await cls._dispatch_send(bot, event, is_group, session_id, ret)
            return
        for seg in message_chain.chain:
            if isinstance(seg, Node | Nodes):
                # 合并转发消息
                if isinstance(seg, Node):
                    nodes = Nodes([seg])
                    seg = nodes

                payload = await seg.to_dict()

                if is_group:
                    payload["group_id"] = session_id
                    if isinstance(event, Event) and event.get("self_id"):
                        payload["self_id"] = event["self_id"]
                    await bot.call_action("send_group_forward_msg", **payload)
                else:
                    payload["user_id"] = session_id
                    if isinstance(event, Event) and event.get("self_id"):
                        payload["self_id"] = event["self_id"]
                    await bot.call_action("send_private_forward_msg", **payload)
            elif isinstance(seg, File):
                await cls._send_file_segment(bot, event, is_group, session_id, seg)
            else:
                messages = await cls._parse_onebot_json(MessageChain([seg]))
                if not messages:
                    continue
                await cls._dispatch_send(bot, event, is_group, session_id, messages)
                await asyncio.sleep(0.5)

    @staticmethod
    def _encode_file_base64(path: str) -> str:
        """把本机文件读成 base64。超过上限时直接报错，让调用方保留原始失败。"""
        size = os.path.getsize(path)
        if size > FILE_BASE64_LIMIT_BYTES:
            raise ValueError(
                f"文件 {os.path.basename(path)} 有 {size / 1024 / 1024:.1f} MB，"
                f"超过 base64 内联上限 {FILE_BASE64_LIMIT_BYTES // 1024 // 1024} MB"
            )
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    @classmethod
    async def _send_file_segment(
        cls,
        bot: CQHttp,
        event: Event | None,
        is_group: bool,
        session_id: str | None,
        seg: File,
    ) -> None:
        """发送文件段，必要时把文件内联成 base64 重发一次。

        协议端（NapCat、Lagrange 等）常与 AstrBot 跑在不同的容器或主机上，
        此时本机路径在它那边并不存在，发送会以 ENOENT 失败。先按路径发
        （同机部署最省带宽），只有在协议端明确说文件不存在时才改用
        base64 重发，以免把发送超时之类的错误也重试成重复消息。
        """
        known_split = id(bot) in _PATH_SEND_UNSUPPORTED
        if not known_split:
            d = await cls._from_segment_to_dict(seg)
            try:
                await cls._dispatch_send(bot, event, is_group, session_id, [d])
                return
            except Exception as e:
                local_path = _local_file_path(d.get("data", {}).get("file", ""))
                if not local_path or not _is_missing_file_error(e):
                    raise
                _PATH_SEND_UNSUPPORTED.add(id(bot))
                logger.warning(
                    f"协议端读不到本机文件 {local_path}（{e}），改用 base64 重发；"
                    "该连接后续的文件将直接内联。若文件较大，"
                    "建议配置 callback_api_base 走文件服务。"
                )
        d = await cls._from_segment_to_dict(seg, file_as_base64=True)
        await cls._dispatch_send(bot, event, is_group, session_id, [d])

    async def send(self, message: MessageChain) -> None:
        """发送消息"""
        event = getattr(self.message_obj, "raw_message", None)

        is_group = bool(self.get_group_id())
        session_id = self.get_group_id() if is_group else self.get_sender_id()

        await self.send_message(
            bot=self.bot,
            message_chain=message,
            event=event,  # 不强制要求一定是 Event
            is_group=is_group,
            session_id=session_id,
        )
        await super().send(message)

    async def send_streaming(
        self,
        generator: AsyncGenerator,
        use_fallback: bool = False,
    ):
        if not use_fallback:
            buffer = None
            async for chain in generator:
                if not buffer:
                    buffer = chain
                else:
                    buffer.chain.extend(chain.chain)
            if not buffer:
                return None
            buffer.squash_plain()
            await self.send(buffer)
            return await super().send_streaming(generator, use_fallback)

        buffer = ""
        pattern = re.compile(r"[^。？！~…]+[。？！~…]+")

        async for chain in generator:
            if isinstance(chain, MessageChain):
                for comp in chain.chain:
                    if isinstance(comp, Plain):
                        buffer += comp.text
                        if any(p in buffer for p in "。？！~…"):
                            buffer = await self.process_buffer(buffer, pattern)
                    else:
                        await self.send(MessageChain(chain=[comp]))
                        await asyncio.sleep(1.5)  # 限速

        buffer = buffer.strip()
        if buffer:
            await self.send(MessageChain([Plain(buffer)]))
        return await super().send_streaming(generator, use_fallback)

    async def get_group(self, group_id=None, **kwargs):
        """Get OneBot group details while preserving inbound data on failures.

        Args:
            group_id: Optional OneBot group identifier.
            **kwargs: Reserved compatibility arguments.

        Returns:
            Enriched group information, or a basic group when an API is unavailable.
        """
        resolved_group_id = group_id or self.get_group_id()
        if not resolved_group_id:
            return None
        resolved_group_id = str(resolved_group_id)
        api_group_id = (
            int(resolved_group_id) if resolved_group_id.isdigit() else resolved_group_id
        )

        current_group = self.message_obj.group
        group = (
            current_group
            if current_group and current_group.group_id == resolved_group_id
            else Group(group_id=resolved_group_id)
        )

        routing_params = {}
        if getattr(self.message_obj, "self_id", None):
            routing_params["self_id"] = self.message_obj.self_id

        try:
            info = await self.bot.call_action(
                "get_group_info",
                group_id=api_group_id,
                **routing_params,
            )
            if isinstance(info, dict):
                group.group_name = info.get("group_name") or group.group_name
                member_count = info.get("member_count")
                if member_count is not None:
                    try:
                        group.member_count = int(member_count)
                    except (TypeError, ValueError):
                        logger.warning(
                            "[aiocqhttp] Invalid member_count for group %s",
                            resolved_group_id,
                        )
        except Exception as exc:
            logger.warning(
                "[aiocqhttp] Failed to get group information for %s: %s",
                resolved_group_id,
                exc,
            )

        try:
            members = await self.bot.call_action(
                "get_group_member_list",
                group_id=api_group_id,
                **routing_params,
            )
        except Exception as exc:
            logger.warning(
                "[aiocqhttp] Failed to get members for group %s: %s",
                resolved_group_id,
                exc,
            )
            return group
        if not isinstance(members, list):
            return group

        owner_id = None
        admin_ids: list[str] = []
        for member in members:
            if not isinstance(member, dict) or member.get("user_id") is None:
                continue
            if member.get("role") == "owner":
                owner_id = str(member["user_id"])
            if member.get("role") == "admin":
                admin_ids.append(str(member["user_id"]))

        group.group_admins = admin_ids
        group.group_owner = owner_id
        group.members = [
            MessageMember(
                user_id=str(member["user_id"]),
                nickname=member.get("nickname") or member.get("card"),
            )
            for member in members
            if isinstance(member, dict) and member.get("user_id") is not None
        ]
        if group.member_count is None:
            group.member_count = len(group.members)
        return group
