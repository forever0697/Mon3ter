"""
Mon3ter 全局数据类型定义

本模块是项目的"词汇表"——所有模块共享的数据结构都在这里定义。
核心数据结构（Message / MemoryItem / ...）只用标准库 dataclasses；
全局配置（AppConfig 及各 Section）使用 pydantic 做校验与反序列化。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from mon3ter.core.clock import now_iso, now as clock_now


# ═══════════════════════════════════════════════════════════════════════
# 推理引擎相关
# ═══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class Message:
    """
    兼容 OpenAI Chat Completion 格式的单条消息。

    这是系统内部传递对话内容的统一容器。
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)


class ModelLoadConfig(BaseModel):
    """
    模型加载参数。

    这些参数在加载后不可变——更改需卸载重载。
    """
    model_path: str                                       # GGUF 模型文件绝对路径
    n_ctx: int = Field(default=8192, gt=0)                # 上下文窗口
    n_gpu_layers: int = Field(default=-1, ge=-1)          # -1=全部, 0=纯CPU
    n_threads: int | None = None                          # CPU线程数, None=自动
    verbose: bool = False
    chat_format: str = "chatml"
    use_worker: bool = False                              # True=子进程隔离, False=同进程加载

    @field_validator("model_path")
    @classmethod
    def path_must_exist(cls, v: str) -> str:
        if not Path(v).is_file():
            raise ValueError(f"GGUF 模型文件不存在: {v}")
        return v


class SamplingParams(BaseModel):
    """
    每次推理可变的采样参数。

    与 InferenceRequest 配合使用，支持每请求不同参数。
    """
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    stop: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class InferenceRequest:
    """
    一次推理请求的参数集。

    由对话管理器构造，传给推理引擎。
    """
    messages: list[Message]
    sampling: SamplingParams = field(default_factory=SamplingParams)


# ═══════════════════════════════════════════════════════════════════════
# 人格系统相关
# ═══════════════════════════════════════════════════════════════════════

class Emotion(str, Enum):
    """情绪状态枚举。"""
    HAPPY = "happy"
    QUIET = "quiet"
    CARING = "caring"
    CURIOUS = "curious"
    SAD = "sad"
    PLAYFUL = "playful"

    def description(self) -> str:
        _map = {
            Emotion.HAPPY:   "轻松愉快，语气明快",
            Emotion.QUIET:   "安静平和，语气舒缓",
            Emotion.CARING:  "温柔关切，语气体贴",
            Emotion.CURIOUS: "好奇，喜欢追问细节",
            Emotion.SAD:     "略带伤感，语气柔软",
            Emotion.PLAYFUL: "调皮幽默，偶尔开玩笑",
        }
        return _map.get(self, "自然")

    def emoji(self) -> str:
        _map = {
            Emotion.HAPPY:   "😊",
            Emotion.QUIET:   "😴",
            Emotion.CARING:  "🫂",
            Emotion.CURIOUS: "🤔",
            Emotion.SAD:     "😢",
            Emotion.PLAYFUL: "😜",
        }
        return _map.get(self, "😶")


@dataclass(slots=True)
class EmotionState:
    """当前情绪状态快照。"""
    emotion: Emotion = Emotion.HAPPY
    confidence: float = 1.0
    trigger: str = ""
    updated_at: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# 记忆系统相关
# ═══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class MemoryItem:
    """从向量库检索出的一条记忆。"""
    id: str
    content: str
    similarity: float
    timestamp: str
    topics: list[str] = field(default_factory=list)
    importance: float = 0.5

    def recency_days(self) -> float:
        try:
            dt = datetime.fromisoformat(self.timestamp)
            # 确保两者都是 aware 或都是 naive 后再相减
            now = clock_now()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=now.tzinfo)
            return (now - dt).total_seconds() / 86400.0
        except (ValueError, TypeError):
            return 999.0


@dataclass(slots=True)
class UserFact:
    """关于用户的一条结构化事实。"""
    key: str
    value: str
    source: str = ""
    confidence: float = 1.0
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class ConversationSummary:
    """对话摘要。"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=now_iso)
    summary: str = ""
    topics: list[str] = field(default_factory=list)
    emotion_at_time: str = ""
    importance: float = 0.5


# ═══════════════════════════════════════════════════════════════════════
# 对话管理相关
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ConversationContext:
    """
    当前会话的上下文窗口。

    分两段管理:
      prefix — 固定前缀（System Prompt, 角色卡, 环境信息）
      _rounds — 对话轮次列表，每轮为 (user_msg, assistant_msg) 元组

    采用轮索引管理而非平铺配对删除——确保 system/tool 消息混入时不会错位。
    """
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    prefix: list[Message] = field(default_factory=list)
    _rounds: list[tuple[Message, Message]] = field(default_factory=list)
    max_turns: int = 20

    @property
    def history(self) -> list[Message]:
        """将轮次展平为 Message 列表，供推理使用。"""
        result: list[Message] = []
        for user_msg, assistant_msg in self._rounds:
            result.append(user_msg)
            result.append(assistant_msg)
        return result

    def all_messages(self) -> list[Message]:
        """返回完整消息列表（prefix + history），供推理使用。"""
        return self.prefix + self.history

    @property
    def turn_count(self) -> int:
        return len(self._rounds)

    def append(self, user_msg: Message, assistant_msg: Message) -> None:
        """追加一轮对话，超出上限时丢弃最早一轮。"""
        self._rounds.append((user_msg, assistant_msg))
        self._trim()

    def _trim(self) -> None:
        """丢弃最早一轮——O(1) pop(0)，不受消息类型干扰。"""
        while len(self._rounds) > self.max_turns:
            self._rounds.pop(0)

    def clear(self) -> None:
        self._rounds.clear()
        self.session_id = uuid.uuid4().hex[:8]


# ═══════════════════════════════════════════════════════════════════════
# 联网模块相关
# ═══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class WeatherInfo:
    """天气数据。"""
    city: str = ""
    temperature_c: float = 0.0
    description: str = ""
    humidity: int = 0
    wind_speed_kmh: float = 0.0
    fetched_at: str = field(default_factory=now_iso)

    def summary(self) -> str:
        if not self.city:
            return "天气数据暂不可用"
        return (
            f"{self.city}，{self.description}，"
            f"{self.temperature_c:.0f}°C，湿度{self.humidity}%"
        )


@dataclass(slots=True)
class NewsItem:
    """一条新闻或热搜。"""
    title: str
    url: str = ""
    source: str = ""
    rank: int = 0
    fetched_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class EnvSummary:
    """注入 System Prompt 的环境信息。"""
    datetime_str: str = field(default_factory=lambda: clock_now().strftime(
        "%Y年%m月%d日 %A %H:%M"
    ))
    weather: WeatherInfo | None = None
    news_headlines: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        lines = [f"现在是 {self.datetime_str}。"]
        if self.weather:
            lines.append(f"天气: {self.weather.summary()}。")
        if self.news_headlines:
            lines.append(f"今日热搜: {'; '.join(self.news_headlines[:5])}。")
        return " ".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# WebSocket 通信协议
# ═══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class WSMessage:
    """
    WebSocket 前后端通信的统一消息格式。

    extra 字段嵌套而非平铺，防止字段覆盖风险。
    """
    type: str                              # chat | token | done | emotion | greeting | idle | command | error | ping
    content: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"type": self.type, "content": self.content, "extra": self.extra},
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> "WSMessage":
        data = json.loads(raw)
        return cls(
            type=data.get("type", ""),
            content=data.get("content", ""),
            extra=data.get("extra", {}),
        )


# ═══════════════════════════════════════════════════════════════════════
# 全局配置 (pydantic — 启动期校验)
# ═══════════════════════════════════════════════════════════════════════

class ModelLoadSection(BaseModel):
    """models.load — 模型加载参数。"""
    llm_path: str = "models/llm/qwen2.5-7b-instruct-q4_k_m.gguf"
    n_ctx: int = Field(default=8192, gt=0)
    n_gpu_layers: int = Field(default=-1, ge=-1)
    use_worker: bool = False  # True=子进程隔离, False=同进程加载


class ModelSamplingSection(BaseModel):
    """models.sampling — 默认采样参数。"""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)


class ModelsSection(BaseModel):
    """models — 模型相关配置。"""
    load: ModelLoadSection = Field(default_factory=ModelLoadSection)
    sampling: ModelSamplingSection = Field(default_factory=ModelSamplingSection)
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"


class PersonalitySection(BaseModel):
    character_file: str = "mon3ter/personality/character.yaml"


class MemorySection(BaseModel):
    chroma_path: str = "data/chroma"
    sqlite_path: str = "data/summaries.db"
    retrieval_top_k: int = Field(default=5, gt=0)
    compress_interval_hours: int = Field(default=24, ge=0)


class WebSection(BaseModel):
    weather_city: str = "北京"
    weather_cache_ttl: int = Field(default=3600, gt=0)
    news_cache_ttl: int = Field(default=7200, gt=0)


class ServerSection(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=9876, gt=1024, lt=65536)


class UserSection(BaseModel):
    name: str = "用户"


class AppConfig(BaseModel):
    """从 config.yaml 加载的全局配置。"""
    models: ModelsSection = Field(default_factory=ModelsSection)
    personality: PersonalitySection = Field(default_factory=PersonalitySection)
    memory: MemorySection = Field(default_factory=MemorySection)
    web: WebSection = Field(default_factory=WebSection)
    server: ServerSection = Field(default_factory=ServerSection)
    user: UserSection = Field(default_factory=UserSection)

    @classmethod
    def from_yaml(cls, path: str | Path, root_dir: str | Path) -> "AppConfig":
        """
        从 YAML 文件加载并校验配置。

        root_dir: 项目根目录，用于解析配置中的相对路径。
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        # ── 路径解析: 相对路径 → 相对 root_dir ──────────────────
        root = Path(root_dir)

        def _resolve(p: str) -> str:
            path_obj = Path(p)
            if path_obj.is_absolute():
                return str(path_obj)
            return str(root / path_obj)

        # ── models.load.llm_path ────────────────────────────────
        models_raw = raw.get("models", {})
        if "load" in models_raw and "llm_path" in models_raw["load"]:
            models_raw["load"]["llm_path"] = _resolve(models_raw["load"]["llm_path"])
        else:
            # 兼容旧平铺格式: models.llm_path
            old_llm = models_raw.get("llm_path", "")
            if old_llm:
                models_raw.setdefault("load", {})
                models_raw["load"]["llm_path"] = _resolve(old_llm)

        # ── personality.character_file ──────────────────────────
        if "personality" in raw and "character_file" in raw["personality"]:
            raw["personality"]["character_file"] = _resolve(raw["personality"]["character_file"])

        # ── memory 路径 ─────────────────────────────────────────
        for key in ("chroma_path", "sqlite_path"):
            if "memory" in raw and key in raw["memory"]:
                raw["memory"][key] = _resolve(raw["memory"][key])

        return cls.model_validate(raw)
