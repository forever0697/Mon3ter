"""
Mon3ter 情绪状态机

职责:
  - 维护当前情绪状态
  - 接收多来源信号更新情绪: LLM 输出标签（优先） > 关键词规则 > 空闲时长
  - 提供当前情绪的 prompt 描述（供人格引擎注入 System Prompt）

策略（对齐 docs/v0.0.1-改进与重构计划.md §3.2）:
  LLM 在回复中主动输出情绪标签 → 优先采用
  规则匹配兜底 → 关键词触发
  空闲时长 → 仅在无对话时影响情绪（由 conversation 模块调用时传入）
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mon3ter.core.types import Emotion, EmotionState

logger = logging.getLogger("mon3ter.emotion")


class EmotionStateMachine:
    """
    情绪状态机 — 管理当前情绪，提供信号驱动的状态切换。
    """

    # 默认衰减: 情绪持续此秒数后回退到 HAPPY
    DECAY_SECONDS: float = 600.0  # 10 分钟

    def __init__(self, triggers: list[dict[str, Any]] | None = None) -> None:
        """
        Args:
            triggers: 触发规则列表，每项含 keywords/emotion/weight/description。
                      为空时使用内置默认规则。
        """
        self._triggers = triggers or self._default_triggers()
        self._current = EmotionState()
        logger.debug("情绪状态机初始化: %s", self._current.emotion.value)

    # ── 状态查询 ────────────────────────────────────────────────────

    @property
    def current(self) -> EmotionState:
        return self._current

    @property
    def emotion(self) -> Emotion:
        return self._current.emotion

    def prompt_description(self) -> str:
        """
        返回一段可供注入 System Prompt 的情绪描述。
        格式: "轻松愉快，语气明快"
        """
        return self._current.emotion.description()

    # ── 外部更新入口 ────────────────────────────────────────────────

    def update(
        self,
        user_message: str | None = None,
        time_since_last: float = -1.0,
        llm_tag: Emotion | None = None,
    ) -> EmotionState:
        """
        根据可用信号更新情绪状态。

        优先级: LLM 标签 > 关键词规则 > 空闲时长衰减

        Args:
            user_message: 用户最新消息（关键词匹配用）
            time_since_last: 距上次对话的秒数（空闲检测用）
            llm_tag: LLM 回复中携带的情绪标签（最高优先级）

        Returns:
            更新后的情绪状态
        """
        # ── 1. LLM 标签（最高优先级）─────────────────────────────────
        if llm_tag is not None and llm_tag in Emotion:
            self._set(
                emotion=llm_tag,
                confidence=0.95,
                trigger=f"LLM 情绪标签: {llm_tag.value}",
            )
            return self._current

        # ── 2. 关键词规则 ────────────────────────────────────────────
        if user_message:
            match = self._match_keywords(user_message)
            if match is not None:
                emotion, weight, desc = match
                self._set(
                    emotion=emotion,
                    confidence=weight,
                    trigger=f"关键词匹配: {desc}",
                )
                return self._current

        # ── 3. 空闲时长衰减 ──────────────────────────────────────────
        if time_since_last > 0:
            if time_since_last >= 1800:  # 30 分钟 → QUIET
                self._set(
                    emotion=Emotion.QUIET,
                    confidence=min(0.7 + time_since_last / 3600, 1.0),
                    trigger=f"空闲 {time_since_last:.0f}s",
                )
                return self._current

        # ── 4. 自然衰减 → HAPPY ──────────────────────────────────────
        # 使用 monotonic 不受系统时间调整影响（NTP/夏令时/用户手动改时间）
        elapsed_since_update = time.monotonic() - self._current.updated_at
        if (
            self._current.emotion != Emotion.HAPPY
            and elapsed_since_update > self.DECAY_SECONDS
        ):
            self._set(
                emotion=Emotion.HAPPY,
                confidence=0.5,
                trigger="自然衰减回默认",
            )

        return self._current

    # ── 关键词匹配 ──────────────────────────────────────────────────

    def _match_keywords(
        self, text: str
    ) -> tuple[Emotion, float, str] | None:
        """
        在文本中匹配触发规则的关键词。

        Returns:
            (emotion, weight, description) 或 None
        """
        text_lower = text.lower()
        best: tuple[Emotion, float, str] | None = None

        for rule in self._triggers:
            keywords: list[str] = rule.get("keywords", [])
            emotion_str: str = rule.get("emotion", "happy")
            weight: float = rule.get("weight", 0.5)
            desc: str = rule.get("description", "")

            # 检查是否有任一关键词命中
            if any(kw.lower() in text_lower for kw in keywords):
                try:
                    em = Emotion(emotion_str)
                except ValueError:
                    continue
                # 取最高权重
                if best is None or weight > best[1]:
                    best = (em, weight, desc)

        return best

    # ── 内部 ────────────────────────────────────────────────────────

    def _set(self, emotion: Emotion, confidence: float, trigger: str) -> None:
        if emotion == self._current.emotion:
            # 同情绪，只更新置信度和触发原因
            self._current.confidence = max(self._current.confidence, confidence)
            self._current.trigger = trigger
            return

        logger.debug(
            "情绪切换: %s → %s (%.0f%%, %s)",
            self._current.emotion.value,
            emotion.value,
            confidence * 100,
            trigger,
        )
        self._current = EmotionState(
            emotion=emotion,
            confidence=confidence,
            trigger=trigger,
            updated_at=time.monotonic(),
        )

    @staticmethod
    def _default_triggers() -> list[dict[str, Any]]:
        """内置默认触发规则（未加载 YAML 时使用）。"""
        return [
            {
                "keywords": ["不开心", "难过", "伤心", "哭", "抑郁", "焦虑", "崩溃", "绝望", "孤独"],
                "emotion": "caring",
                "weight": 0.9,
                "description": "用户表达负面情绪",
            },
            {
                "keywords": ["开心", "哈哈", "太好了", "爽", "兴奋", "激动"],
                "emotion": "happy",
                "weight": 0.8,
                "description": "用户表达积极情绪",
            },
            {
                "keywords": ["好奇", "为什么", "推荐", "建议"],
                "emotion": "curious",
                "weight": 0.7,
                "description": "用户探讨或提问",
            },
            {
                "keywords": ["无聊", "好烦", "没意思"],
                "emotion": "playful",
                "weight": 0.6,
                "description": "用户感到无聊",
            },
            {
                "keywords": ["再见", "晚安", "拜拜"],
                "emotion": "quiet",
                "weight": 0.8,
                "description": "对话接近尾声",
            },
        ]
