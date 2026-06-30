"""
Mon3ter 一键启动脚本

自动启动后端 + CLI 客户端，Ctrl+C 一键退出。

用法:
  cd C:/AI_Project/Mon3ter
  .venv/Scripts/python A_Test/launch.py

  可选参数:
  --verbose    显示后端详细日志
  --port N     指定端口 (默认 9876)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# 项目根目录 = launch.py 的上上级 (Mon3ter/)
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Mon3ter 一键启动")
    parser.add_argument("--verbose", action="store_true", help="显示后端详细日志")
    parser.add_argument("--port", type=int, default=9876, help="WebSocket 端口")
    args = parser.parse_args()

    python = str(ROOT / ".venv" / "Scripts" / "python.exe")
    main_py = str(ROOT / "main.py")
    cli_py = str(ROOT / "test_cli.py")

    # ── 构建后端命令 ────────────────────────────────────────────
    backend_cmd = [python, main_py]
    if args.verbose:
        backend_cmd.append("--verbose")

    print("=" * 50)
    print("  Mon3ter 一键启动")
    print(f"  后端: {' '.join(backend_cmd)}")
    print(f"  端口: {args.port}")
    print("=" * 50)
    print()

    # ── 1. 启动后端 ────────────────────────────────────────────
    print("[1/2] 启动后端 (加载模型, 约 5-10 秒)...")
    print()

    backend = subprocess.Popen(
        backend_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
    )

    # ── 2. 等待后端就绪 ────────────────────────────────────────
    ready = False
    fail = False
    deadline = time.time() + 120  # 最多等 2 分钟

    for line in backend.stdout:
        line = line.rstrip()
        print(f"  [后端] {line}")

        if "WebSocket 服务启动" in line or "server listening" in line:
            ready = True
            break

        # 忽略 Python logging 系统内部的编码报错（不是应用错误）
        if "Logging error" in line or "logging/__init__.py" in line:
            continue

        if "Traceback" in line:
            fail = True
        elif "[ERROR" in line and "mon3ter" in line:
            fail = True
        elif "失败" in line and "软依赖" not in line and "未就绪" not in line:
            fail = True

        if time.time() > deadline:
            print("\n❌ 后端启动超时 (2分钟)")
            fail = True
            break

    if fail or not ready:
        print("\n❌ 后端启动失败，请检查上方的错误信息")
        backend.terminate()
        backend.wait()
        sys.exit(1)

    print()
    print("[2/2] 启动 CLI 客户端...")
    print()

    # ── 3. 启动 CLI 客户端 (前台) ──────────────────────────────
    try:
        cli = subprocess.run(
            [python, cli_py, "--port", str(args.port)],
            cwd=str(ROOT),
        )
    except KeyboardInterrupt:
        pass
    finally:
        print()
        print("正在关闭后端...")
        backend.terminate()
        try:
            backend.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend.kill()
        print("Mon3ter 已退出。")

    sys.exit(cli.returncode if cli else 0)


if __name__ == "__main__":
    main()
