"""
Mon3ter 人格系统

├── character.yaml    — 角色卡（用户可编辑）
├── character.py      — 人格引擎: 加载角色卡 + 拼装 System Prompt
└── emotion_state.py  — 情绪状态机: 多来源信号驱动的情绪管理
"""

from mon3ter.personality.character import PersonalityEngine
from mon3ter.personality.emotion_state import EmotionStateMachine

__all__ = ["PersonalityEngine", "EmotionStateMachine"]
