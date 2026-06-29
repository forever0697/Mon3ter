"""
Mon3ter CLI 测试客户端

在 Electron 前端完成之前的命令行对话界面。
用法:
  python test_cli.py                    # 连接默认 ws://127.0.0.1:9876
  python test_cli.py --port 12345       # 指定端口
  python test_cli.py --host 192.168.1.x # 指定 IP

交互:
  直接打字 → 发送消息
  /quit     → 退出
  /clear    → 清屏
  /help     → 显示帮助
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any

import websockets

# ═══════════════════════════════════════════════════════════════════════
# 终端颜色 (跨平台)
# ═══════════════════════════════════════════════════════════════════════

class Color:
    """ANSI 转义序列，Windows Terminal / PowerShell 均支持。"""
    RESET   = "\033[0m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"


# ═══════════════════════════════════════════════════════════════════════
# 客户端核心
# ═══════════════════════════════════════════════════════════════════════

class ChatClient:
    """WebSocket 聊天客户端。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 9876) -> None:
        self.uri = f"ws://{host}:{port}"
        self.ws: Any = None
        self._running = False
        self._token_count = 0           # 当前这一句收到的 token 数
        self._start_time: float = 0.0   # 当前这句开始推理的时间

    # ── 连接 ────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """连接到 Mon3ter 后端。返回是否成功。"""
        print(f"{Color.DIM}正在连接 {self.uri} ...{Color.RESET}", end=" ", flush=True)
        try:
            self.ws = await websockets.connect(self.uri, ping_interval=30)
            print(f"{Color.GREEN}已连接{Color.RESET}")
            return True
        except Exception as exc:
            print(f"{Color.RED}连接失败: {exc}{Color.RESET}")
            print(f"   请确认后端已启动: python main.py")
            return False

    # ── 主循环 ──────────────────────────────────────────────────────

    async def run(self) -> None:
        """主循环: 连接 → 监听 + 输入 → 断开。"""
        if not await self.connect():
            return

        self._running = True

        # 两个协程并发: 收消息 + 读输入
        receiver = asyncio.create_task(self._receive_loop())
        sender = asyncio.create_task(self._input_loop())

        _, pending = await asyncio.wait(
            [receiver, sender],
            return_when=asyncio.FIRST_COMPLETED,
        )

        self._running = False

        # 取消未完成的那个
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # 清理
        if self.ws:
            await self.ws.close()
        print(f"\n{Color.DIM}已断开连接。{Color.RESET}")

    # ── 收消息 ──────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        """持续接收服务端推送的消息。"""
        try:
            async for raw in self.ws:
                await self._dispatch(raw)
        except websockets.ConnectionClosed:
            if self._running:
                print(f"\n{Color.RED}与服务端的连接断开{Color.RESET}")
        except Exception as exc:
            if self._running:
                print(f"\n{Color.RED}接收错误: {exc}{Color.RESET}")

    async def _dispatch(self, raw: str) -> None:
        """根据消息类型派发到不同处理逻辑。"""
        import json as _json
        try:
            data = _json.loads(raw)
        except _json.JSONDecodeError:
            return

        msg_type = data.get("type", "")
        content = data.get("content", "")

        if msg_type == "token":
            self._on_token(content)

        elif msg_type == "done":
            self._on_done()

        elif msg_type == "greeting":
            self._on_greeting(content, data.get("extra", {}))

        elif msg_type == "emotion":
            self._on_emotion(content)

        elif msg_type == "idle":
            self._on_idle(content)

        elif msg_type == "pong":
            pass  # 心跳响应，忽略

        elif msg_type == "error":
            self._on_error(content)

        else:
            # 未知类型，仍然打印
            if content:
                print(f"\n[{msg_type}] {content}")

    # ── 消息处理器 ──────────────────────────────────────────────────

    def _on_token(self, token: str) -> None:
        """收到一个 token → 直接打印，不换行。"""
        if self._token_count == 0:
            self._start_time = time.time()
        self._token_count += 1
        print(token, end="", flush=True)

    def _on_done(self) -> None:
        """一句话结束。"""
        elapsed = time.time() - self._start_time if self._start_time else 0
        tok_s = self._token_count / elapsed if elapsed > 0 else 0
        print(f"\n{Color.DIM}({self._token_count} tokens, {elapsed:.1f}s, {tok_s:.0f} tok/s){Color.RESET}")
        print()  # 空一行，准备下次输入
        self._token_count = 0
        self._start_time = 0.0

    def _on_greeting(self, content: str, extra: dict) -> None:
        """收到开机问候（连接时的欢迎消息）。"""
        mode = extra.get("mode", "unknown")
        mode_label = "完整模式" if mode == "full" else "回退模式（无记忆/人格）"
        print(f"\n{Color.CYAN}{Color.BOLD}Mon3ter:{Color.RESET} {Color.CYAN}{content}{Color.RESET}")
        print(f"{Color.DIM}模式: {mode_label}{Color.RESET}")
        print()
        print(f"  {Color.DIM}输入消息开始对话，/quit 退出，/help 帮助{Color.RESET}")
        print()

    def _on_emotion(self, state: str) -> None:
        """情绪变化提示。"""
        print(f"{Color.YELLOW}[情绪 → {state}]{Color.RESET}")

    def _on_idle(self, content: str) -> None:
        """收到主动搭话。"""
        print(f"\n{Color.CYAN}{Color.BOLD}Mon3ter 搭话:{Color.RESET} {Color.CYAN}{content}{Color.RESET}")

    def _on_error(self, content: str) -> None:
        print(f"\n{Color.RED}错误: {content}{Color.RESET}")

    # ── 发消息 ──────────────────────────────────────────────────────

    async def _input_loop(self) -> None:
        """持续读取用户输入并发送。"""
        # input() 是阻塞的，放到线程里跑
        loop = asyncio.get_running_loop()

        while self._running:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except (EOFError, KeyboardInterrupt):
                break

            if not line:
                break

            line = line.strip()

            if not line:
                continue

            # ── 本地命令 ────────────────────────────────────────────
            if line.startswith("/"):
                should_quit = self._handle_local_command(line)
                if should_quit:
                    break

            # ── 发送聊天 ────────────────────────────────────────────
            else:
                await self._send_chat(line)

    def _handle_local_command(self, line: str) -> bool:
        """处理本地命令。返回 True 表示需要退出。"""
        cmd, _, arg = line[1:].partition(" ")

        if cmd == "quit" or cmd == "q":
            print(f"{Color.DIM}退出...{Color.RESET}")
            return True

        elif cmd == "clear" or cmd == "c":
            # Windows: cls, Unix: clear
            import os as _os
            _os.system("cls" if _os.name == "nt" else "clear")
            return False

        elif cmd == "help" or cmd == "h":
            print()
            print(f"  {Color.BOLD}命令列表{Color.RESET}")
            print(f"  /quit, /q    退出")
            print(f"  /clear, /c   清屏")
            print(f"  /help, /h    帮助")
            print(f"  直接输入     发送对话")
            print()
            return False

        else:
            print(f"{Color.YELLOW}未知命令: /{cmd}，输入 /help 查看可用命令{Color.RESET}")
            return False

    async def _send_chat(self, text: str) -> None:
        """发送聊天消息。"""
        import json as _json
        payload = _json.dumps({"type": "chat", "content": text}, ensure_ascii=False)
        try:
            await self.ws.send(payload)
        except Exception as exc:
            print(f"{Color.RED}发送失败: {exc}{Color.RESET}")


# ═══════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Mon3ter CLI 测试客户端")
    parser.add_argument("--host", default="127.0.0.1", help="服务端 IP (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9876, help="服务端端口 (默认 9876)")
    args = parser.parse_args()

    print(f"{Color.BOLD}Mon3ter CLI 客户端{Color.RESET}")
    print(f"{Color.DIM}后端: ws://{args.host}:{args.port}{Color.RESET}")
    print(f"  {Color.DIM}请先在另一个终端启动服务端: python main.py{Color.RESET}")
    print()

    client = ChatClient(host=args.host, port=args.port)

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print(f"\n{Color.DIM}中断。{Color.RESET}")


if __name__ == "__main__":
    main()
