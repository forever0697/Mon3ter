"""
Mon3ter 人格引擎

职责:
  - 加载 character.yaml 角色卡
  - 根据当前上下文（情绪/记忆/环境/用户事实）动态拼装 System Prompt
  - 不运行推理 —— 只产文本

用法:
    from mon3ter.personality.character import PersonalityEngine

    engine = PersonalityEngine("personality/character.yaml")
    prompt = engine.build_system_prompt(
        user_name="阿杰",
        emotion=Emotion.HAPPY,
        memories=recent_memories,
        env_info="现在是2026年6月30日周六下午3点，北京晴28°C",
        user_facts={"昵称": "阿杰", "宠物": "猫叫团子"},
    )
    # prompt 可直接作为 Message.system(content=prompt) 使用
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from mon3ter.core.types import Emotion, EmotionState, MemoryItem

logger = logging.getLogger("mon3ter.personality")


class PersonalityEngine:
    """
    人格引擎。

    持有角色卡定义，对外提供 System Prompt 拼装。
    不做推理、不存状态 —— 情绪状态由 EmotionStateMachine 独立管理。
    """

    def __init__(self, character_file: str) -> None:
        """
        Args:
            character_file: character.yaml 的路径
        """
        self._path = Path(character_file)
        self._raw: dict[str, Any] = {}

        if self._path.is_file():
            self._load()
        else:
            logger.warning("角色卡文件不存在: %s，使用默认人格", self._path)
            self._raw = self._defaults()

    # ── 加载 ────────────────────────────────────────────────────────

    def _load(self) -> None:
        with open(self._path, "r", encoding="utf-8") as fh:
            self._raw = yaml.safe_load(fh) or {}
        name = self._raw.get("name", "未命名")
        rule_count = len(self._raw.get("speaking_rules", []))
        logger.info("角色卡已加载: %s (%d 条说话规则)", name, rule_count)

    @staticmethod
    def _defaults() -> dict[str, Any]:
        return {
            "name": "Mon3ter",
            "core_identity": "我是Mon3ter，一个住在用户电脑里的AI伴侣。",
            "core_values": [],
            "persona": "我是一个友好的AI伴侣。",
            "speaking_rules": ["口语化表达", "简短回复"],
            "emotion_triggers": [],
        }

    # ── 查询 ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._raw.get("name", "Mon3ter")

    @property
    def core_identity_text(self) -> str:
        return self._raw.get("core_identity", "").strip()

    @property
    def core_values(self) -> list[dict[str, str]]:
        return self._raw.get("core_values", [])

    @property
    def persona_text(self) -> str:
        return self._raw.get("persona", "").strip()

    @property
    def speaking_rules(self) -> list[str]:
        return self._raw.get("speaking_rules", [])

    @property
    def emotion_triggers(self) -> list[dict[str, Any]]:
        return self._raw.get("emotion_triggers", [])

    # ── Prompt 拼装 ─────────────────────────────────────────────────

    def build_system_prompt(
        self,
        user_name: str = "用户",
        emotion: Emotion = Emotion.HAPPY,
        memories: list[MemoryItem] | None = None,
        env_info: str = "",
        user_facts: dict[str, str] | None = None,
    ) -> str:
        """
        拼装完整的 System Prompt — 三层身份模型。

          ① Core Identity   我是谁，我从哪里来（固定）
          ② Core Values     我相信什么，我在乎什么（固定）
          ③ Persona         我给人什么感觉（固定）
          ④ 当前情绪         我现在的心情（可变）
          ⑤ 用户事实         关于对方的信息（可变, 来自记忆系统）
          ⑥ 相关记忆         过往对话要点（可变, 来自记忆系统检索）
          ⑦ 环境信息         时间/天气/新闻（可变, 来自联网模块）
          ⑧ 说话规则         可量化行为约束（固定）
        """
        segments: list[str] = []

        # ── ① 核心身份 ──────────────────────────────────────────────
        identity = self.core_identity_text
        if identity:
            segments.append(identity)

        # ── ② 核心价值观 ────────────────────────────────────────────
        values = self.core_values
        if values:
            value_lines = [
                f"关于'{v.get('key', '')}': {v.get('belief', '')}"
                for v in values
            ]
            segments.append(
                "我秉持这些信念来理解世界:\n" + "\n".join(value_lines)
            )

        # ── ③ 性格表现 ──────────────────────────────────────────────
        persona = self.persona_text
        if persona:
            segments.append(persona)

        # ── ④ 当前情绪 ──────────────────────────────────────────────
        segments.append(f"[当前状态] 现在我的心情是: {emotion.description()}。")

        # ── ⑤ 用户事实 ──────────────────────────────────────────────
        facts = user_facts or {}
        if facts:
            fact_lines = [f"  - {k}: {v}" for k, v in facts.items()]
            segments.append(
                f"[关于用户 {user_name}] 我已知的信息:\n" + "\n".join(fact_lines)
            )

        # ── ⑥ 相关记忆 ──────────────────────────────────────────────
        if memories:
            mem_lines = [f"  - {m.content}" for m in memories[:5]]
            segments.append(
                "[相关记忆] 与此话题相关的过往对话:\n" + "\n".join(mem_lines)
            )

        # ── ⑦ 环境信息 ──────────────────────────────────────────────
        if env_info:
            segments.append(f"[环境] {env_info}")

        # ── ⑧ 说话规则 ──────────────────────────────────────────────
        rules = self.speaking_rules
        if rules:
            rule_lines = [f"  - {r}" for r in rules]
            segments.append(
                "[说话规则]\n" + "\n".join(rule_lines)
            )

        return "\n\n".join(segments)

    # ── 重载 ────────────────────────────────────────────────────────

    def reload(self) -> None:
        """重新加载角色卡（用户修改 YAML 后可热更新）。"""
        self._load()
