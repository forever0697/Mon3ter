"""
Mon3ter 推理引擎

封装 llama-cpp-python，提供统一的本地 LLM 推理接口。
全局单例 —— 其他模块不直接碰模型，只通过这里调用。

用法:
    from mon3ter.core.types import ModelLoadConfig, InferenceRequest, SamplingParams, Message
    from mon3ter.core.engine import InferenceEngine

    config = ModelLoadConfig(model_path="models/llm/qwen2.5-7b-instruct-q4_k_m.gguf")
    engine = InferenceEngine(config)
    engine.load()

    request = InferenceRequest(messages=[Message.user("你好")])
    reply = engine.chat(request)
    async for token in engine.chat_stream_async(request): ...  # 异步流式
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import AsyncIterator, Generator

from llama_cpp import Llama

from .types import InferenceRequest, Message, ModelLoadConfig

logger = logging.getLogger("mon3ter.engine")


# ═══════════════════════════════════════════════════════════════════════
# 异常
# ═══════════════════════════════════════════════════════════════════════

class EngineError(Exception):
    """推理引擎所有异常的基类。"""

class ModelNotLoadedError(EngineError):
    """模型未加载时调用推理。"""

class ModelAlreadyLoadedError(EngineError):
    """重复加载模型。"""

class ModelFileNotFoundError(EngineError):
    """GGUF 模型文件不存在。"""

class InferenceFailedError(EngineError):
    """推理过程出错。"""


# ═══════════════════════════════════════════════════════════════════════
# 引擎主体
# ═══════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    本地 LLM 推理引擎。

    封装 llama-cpp-python，负责模型生命周期和推理调用。
    全局单例 —— 通过模块级 get_engine() / reset_engine() 管理。
    """

    def __init__(self, config: ModelLoadConfig) -> None:
        self.config = config
        self._model: Llama | None = None
        self._lock = threading.Lock()

    # ── 模型生命周期 ────────────────────────────────────────────────

    def load(self) -> None:
        """加载 GGUF 模型到显存/内存。耗时 5~10 秒。"""
        if self._model is not None:
            raise ModelAlreadyLoadedError(
                f"模型已加载 ({self.config.model_path})，请先调用 unload()"
            )

        if not os.path.isfile(self.config.model_path):
            raise ModelFileNotFoundError(
                f"GGUF 模型文件不存在: {self.config.model_path}"
            )

        logger.info("正在加载模型: %s", self.config.model_path)
        logger.info(
            "  GPU 层数: %d, 上下文: %d, 格式: %s",
            self.config.n_gpu_layers,
            self.config.n_ctx,
            self.config.chat_format,
        )

        try:
            self._model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.n_gpu_layers,
                n_threads=self.config.n_threads,
                chat_format=self.config.chat_format,
                verbose=self.config.verbose,
                n_batch=512,
                use_mmap=True,
                use_mlock=False,
            )
        except Exception as exc:
            raise EngineError(f"模型加载失败: {exc}") from exc

        logger.info("模型加载完成 ✓")

    def unload(self) -> None:
        """卸载模型，释放显存。"""
        if self._model is not None:
            logger.info("正在卸载模型...")
            self._model = None
            logger.info("模型已卸载 ✓")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── 同步推理 ────────────────────────────────────────────────────

    def chat(self, request: InferenceRequest) -> str:
        """阻塞式对话。返回完整回复文本，适合后台任务（摘要、事实提取）。"""
        self._ensure_loaded()

        messages_dicts = [m.to_dict() for m in request.messages]

        logger.debug("推理请求: %d 条消息", len(messages_dicts))

        with self._lock:
            try:
                result = self._model.create_chat_completion(  # type: ignore[union-attr]
                    messages=messages_dicts,
                    temperature=request.sampling.temperature,
                    max_tokens=request.sampling.max_tokens,
                    stop=request.sampling.stop if request.sampling.stop else None,
                    stream=False,
                )
            except Exception as exc:
                raise InferenceFailedError(f"推理失败: {exc}") from exc

        choice = result["choices"][0]
        content = choice["message"].get("content", "")
        finish = choice.get("finish_reason", "")

        logger.debug("推理完成: finish_reason=%s, 长度=%d chars", finish, len(content))
        return content

    def chat_stream(self, request: InferenceRequest) -> Generator[str, None, None]:
        """
        同步流式对话生成器。

        每次 yield 一个 token 字符串，供调用方逐 token 处理。
        """
        self._ensure_loaded()

        messages_dicts = [m.to_dict() for m in request.messages]
        logger.debug("流式推理: %d 条消息", len(messages_dicts))

        with self._lock:
            try:
                stream = self._model.create_chat_completion(  # type: ignore[union-attr]
                    messages=messages_dicts,
                    temperature=request.sampling.temperature,
                    max_tokens=request.sampling.max_tokens,
                    stop=request.sampling.stop if request.sampling.stop else None,
                    stream=True,
                )
            except Exception as exc:
                raise InferenceFailedError(f"流式推理启动失败: {exc}") from exc

            try:
                for chunk in stream:
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
            except Exception as exc:
                raise InferenceFailedError(f"流式推理中断: {exc}") from exc

    # ── 异步流式推理 ────────────────────────────────────────────────

    async def chat_stream_async(self, request: InferenceRequest) -> AsyncIterator[str]:
        """
        异步流式对话 — 用 asyncio.Queue 桥接同步生成器。

        这是 server / orchestrator 的统一入口。
        不复制桥接逻辑——sync→async 桥接只在这一处。
        """
        self._ensure_loaded()

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _produce() -> None:
            try:
                for token in self.chat_stream(request):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        loop.run_in_executor(None, _produce)
        while True:
            item = await queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    # ── 内部 ────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is None:
            raise ModelNotLoadedError("模型未加载，请先调用 load()")


# ═══════════════════════════════════════════════════════════════════════
# 全局单例管理
# ═══════════════════════════════════════════════════════════════════════

_engine: InferenceEngine | None = None
_engine_lock = threading.Lock()


def get_engine(config: ModelLoadConfig | None = None) -> InferenceEngine:
    """获取或创建全局推理引擎单例。"""
    global _engine

    if _engine is not None:
        if config is not None and config != _engine.config:
            logger.warning(
                "引擎已用不同配置初始化，传入的 config 被忽略。"
                "如需更换配置，请先调用 reset_engine()。"
            )
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine

        if config is None:
            raise EngineError("首次初始化引擎必须提供 ModelLoadConfig")

        _engine = InferenceEngine(config)
        return _engine


def reset_engine() -> None:
    """卸载并销毁全局引擎实例。"""
    global _engine

    if _engine is not None:
        _engine.unload()
        _engine = None
        logger.info("引擎已重置")
