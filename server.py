"""
Mon3ter WebSocket 服务端

职责: 前端 ↔ 后端的通信管道。
- 接收前端 JSON 消息，路由到对话管理器（或回退到引擎裸聊）
- 将推理结果逐 token 推回前端
- 主动推送: 开机问候、情绪变化、空闲搭话

协议: JSON over WebSocket, 格式见 core/types.py → WSMessage
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from mon3ter.core.types import (
    InferenceRequest,
    Message,
    SamplingParams,
    ServerSection,
    WSMessage,
)

logger = logging.getLogger("mon3ter.server")

# 存储活跃连接，用于主动推送（开机问候、空闲搭话等）
_active_clients: set[Any] = set()


# ═══════════════════════════════════════════════════════════════════════
# 消息处理器
# ═══════════════════════════════════════════════════════════════════════

async def _handle_chat(websocket, payload: WSMessage, orchestrator, engine, default_sampling: SamplingParams) -> None:
    """
    处理聊天消息。

    优先使用 orchestrator（有记忆/人格/联网），
    orchestrator 不可用时回退到引擎裸聊。
    """
    user_text = payload.content.strip()
    if not user_text:
        await websocket.send(WSMessage(type="error", content="消息为空").to_json())
        return

    try:
        if orchestrator is not None:
            # ── 完整模式: 对话管理器编排 ─────────────────────
            async for token in orchestrator.handle_message(user_text):
                await websocket.send(WSMessage(type="token", content=token).to_json())
        else:
            # ── 回退模式: 引擎裸聊 ──────────────────────────
            request = InferenceRequest(
                messages=[Message.user(user_text)],
                sampling=default_sampling,
            )
            async for token in engine.chat_stream_async(request):
                await websocket.send(WSMessage(type="token", content=token).to_json())

        # 消息结束
        await websocket.send(WSMessage(type="done", content="").to_json())

    except Exception as exc:
        logger.exception("处理消息时出错")
        await websocket.send(WSMessage(type="error", content=str(exc)).to_json())


async def _handle_command(websocket, payload: WSMessage, orchestrator, engine) -> None:
    """处理命令（如抓取网页）。预留接口。"""
    action = payload.extra.get("action", "")
    logger.info("收到命令: %s (暂不支持)", action)
    await websocket.send(WSMessage(
        type="error",
        content=f"命令 '{action}' 尚未实现",
    ).to_json())


# ═══════════════════════════════════════════════════════════════════════
# 连接处理器
# ═══════════════════════════════════════════════════════════════════════

async def _handler(websocket, orchestrator, engine, default_sampling: SamplingParams):
    """
    每个 WebSocket 连接的生命周期。

    职责: 收消息 → 派发 → 推回复。不关心对话逻辑本身。
    """
    client_id = f"{websocket.remote_address}"
    logger.info("客户端连接: %s", client_id)
    _active_clients.add(websocket)

    mode_label = "full" if orchestrator else "fallback"
    await websocket.send(WSMessage(
        type="greeting",
        content="Mon3ter 已连接",
        extra={"mode": mode_label},
    ).to_json())

    try:
        async for raw in websocket:
            try:
                payload = WSMessage.from_json(raw)
            except Exception:
                await websocket.send(WSMessage(
                    type="error", content="消息格式错误，需要 JSON"
                ).to_json())
                continue

            msg_type = payload.type

            if msg_type == "chat":
                await _handle_chat(websocket, payload, orchestrator, engine, default_sampling)

            elif msg_type == "command":
                await _handle_command(websocket, payload, orchestrator, engine)

            elif msg_type == "ping":
                await websocket.send(WSMessage(type="pong", content="").to_json())

            else:
                await websocket.send(WSMessage(
                    type="error",
                    content=f"未知消息类型: {msg_type}",
                ).to_json())

    except ConnectionClosed:
        logger.info("客户端断开: %s", client_id)
    finally:
        _active_clients.discard(websocket)


# ═══════════════════════════════════════════════════════════════════════
# 主动推送
# ═══════════════════════════════════════════════════════════════════════

async def broadcast(msg: WSMessage) -> None:
    """向所有已连接的客户端广播一条消息。"""
    if not _active_clients:
        return
    raw = msg.to_json()
    dead: list[Any] = []
    for ws in _active_clients.copy():
        try:
            await ws.send(raw)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _active_clients.discard(ws)


# ═══════════════════════════════════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════════════════════════════════

async def start(
    server_config: ServerSection,
    default_sampling: SamplingParams,
    orchestrator=None,
    engine=None,
) -> None:
    """
    启动 WebSocket 服务端。

    Args:
        server_config: 服务端配置 (host, port)
        default_sampling: 回退模式的默认采样参数
        orchestrator: 对话管理器实例, None 时使用回退模式
        engine: 推理引擎实例 (回退模式需要)
    """
    async def _h(websocket):
        await _handler(websocket, orchestrator, engine, default_sampling)

    host = server_config.host
    port = server_config.port

    logger.info("WebSocket 服务启动: ws://%s:%d", host, port)
    if orchestrator is None:
        logger.info("  模式: 回退 (引擎裸聊, 无记忆/人格/联网)")
    else:
        logger.info("  模式: 完整 (对话管理器编排)")

    async with serve(_h, host, port) as ws_server:
        await ws_server.serve_forever()
