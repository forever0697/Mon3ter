"""
Mon3ter 推理 Worker 子进程

在独立子进程中运行 Llama 模型，通过 multiprocessing.Queue 与主进程通信。
实现崩溃隔离、显存隔离，为未来模型热切换打下基础。

协议:
    request_queue.get()  → dict (InferenceRequest 序列化) 或 None (shutdown)
    response_queue.put(token: str)  → 单个 token
    response_queue.put(None)        → 推理完成 (sentinel)
    response_queue.put(e: Exception) → 推理出错
    response_queue.put(("loaded",))   → 模型加载完成信号
    response_queue.put(("unloaded",)) → 模型卸载完成信号
"""

from __future__ import annotations

import logging
import os
import sys
from queue import Empty

from mon3ter.core.types import ModelLoadConfig

logger = logging.getLogger("mon3ter.worker")


# ═══════════════════════════════════════════════════════════════════════
# Worker 入口
# ═══════════════════════════════════════════════════════════════════════

def _run_worker(
    config: ModelLoadConfig,
    request_queue: "multiprocessing.Queue",    # type: ignore[name-defined]
    response_queue: "multiprocessing.Queue",   # type: ignore[name-defined]
) -> None:
    """
    推理 worker 主循环。

    在子进程中运行，与主进程通过两个 Queue 通信。
    永不主动退出——只有收到 None 请求时才退出。
    """
    # ── 配置子进程日志 ────────────────────────────────────────────────
    _setup_worker_logging()
    worker_logger = logging.getLogger("mon3ter.worker")

    worker_logger.info("Worker 子进程启动 (PID=%d)", os.getpid())

    # ── 加载模型 ──────────────────────────────────────────────────────
    try:
        from llama_cpp import Llama

        worker_logger.info("正在加载模型: %s", config.model_path)
        worker_logger.info(
            "  GPU 层数: %d, 上下文: %d, 格式: %s",
            config.n_gpu_layers,
            config.n_ctx,
            config.chat_format,
        )

        model = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            n_threads=config.n_threads,
            chat_format=config.chat_format,
            verbose=config.verbose,
            n_batch=512,
            use_mmap=True,
            use_mlock=False,
        )

        worker_logger.info("模型加载完成 ✓")
        response_queue.put(("loaded",))

    except Exception as exc:
        worker_logger.error("模型加载失败: %s", exc)
        response_queue.put(exc)
        return

    # ── 推理循环 ──────────────────────────────────────────────────────
    worker_logger.info("进入推理循环，等待请求...")

    try:
        while True:
            request_dict = request_queue.get()

            # shutdown 信号
            if request_dict is None:
                worker_logger.info("收到 shutdown 信号")
                break

            msg_count = len(request_dict.get("messages", []))
            worker_logger.debug("收到推理请求: %d 条消息", msg_count)

            try:
                stream = model.create_chat_completion(
                    **request_dict,
                    stream=True,
                )

                for chunk in stream:
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        response_queue.put(token)

                response_queue.put(None)  # 完成 sentinel
                worker_logger.debug("推理完成 ✓")

            except Exception as exc:
                worker_logger.error("推理失败: %s", exc)
                try:
                    response_queue.put(exc)
                except Exception:
                    pass

    except KeyboardInterrupt:
        worker_logger.info("收到中断信号")
    except Exception as exc:
        worker_logger.exception("Worker 主循环异常: %s", exc)
    finally:
        worker_logger.info("Worker 正在退出...")
        # 尝试通知主进程（可能已不在监听）
        try:
            response_queue.put(("unloaded",))
        except Exception:
            pass
        worker_logger.info("Worker 已退出 (PID=%d)", os.getpid())


# ═══════════════════════════════════════════════════════════════════════
# 内部辅助
# ═══════════════════════════════════════════════════════════════════════

def _setup_worker_logging() -> None:
    """为 worker 子进程配置独立的日志输出（到 stderr）。"""
    is_verbose = os.environ.get("MON3TER_VERBOSE", "").lower() in ("1", "true", "yes")
    level = logging.DEBUG if is_verbose else logging.INFO

    fmt = logging.Formatter(
        fmt="%(asctime)s [WORKER] %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(fmt)

    wl = logging.getLogger("mon3ter.worker")
    wl.handlers.clear()
    wl.addHandler(handler)
    wl.setLevel(level)
    wl.propagate = False  # 不重复输出到主进程的 handler
