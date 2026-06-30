"""
Mon3ter 统一时钟抽象

职责:
  - 提供单一时间源，所有模块通过此获取当前时间
  - 显式时区 (Asia/Shanghai)
  - 支持注入假时钟用于测试（recency 衰减、问候语时段分支等）

用法:
    from mon3ter.core.clock import clock

    now = clock.now()           # datetime with tz
    iso  = clock.now_iso()      # ISO format string
    stamp = clock.timestamp()   # Unix timestamp (float)
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timezone, timedelta
from typing import Protocol


# ── 时区 ────────────────────────────────────────────────────────────
_CST = timezone(timedelta(hours=8), name="Asia/Shanghai")


# ── Clock 协议 ──────────────────────────────────────────────────────

class Clock(Protocol):
    """可注入的时间源协议。默认实现使用 Asia/Shanghai 时区。"""

    def now(self) -> datetime:
        """返回当前时间（带时区）。"""
        ...

    def now_iso(self) -> str:
        """返回 ISO 8601 格式的时间字符串。"""
        ...

    def timestamp(self) -> float:
        """返回 Unix 时间戳。"""
        ...


# ── 默认实现 ────────────────────────────────────────────────────────

class RealClock:
    """生产环境时钟——Asia/Shanghai 时区。"""

    def now(self) -> datetime:
        return datetime.now(_CST)

    def now_iso(self) -> str:
        return self.now().isoformat()

    def timestamp(self) -> float:
        return _time.time()


class FrozenClock:
    """测试用冻结时钟——时间固定不变。"""

    def __init__(self, dt: datetime | None = None) -> None:
        self._dt = dt if dt and dt.tzinfo else (dt or datetime(2026, 1, 1, tzinfo=_CST))
        if self._dt.tzinfo is None:
            self._dt = self._dt.replace(tzinfo=_CST)

    def now(self) -> datetime:
        return self._dt

    def now_iso(self) -> str:
        return self._dt.isoformat()

    def timestamp(self) -> float:
        return self._dt.timestamp()


# ── 全局时钟实例（测试时可替换为 FrozenClock） ──────────────────────
_clock: Clock = RealClock()


def get_clock() -> Clock:
    """获取当前时钟实例。"""
    return _clock


def set_clock(c: Clock) -> None:
    """替换时钟实例（仅用于测试）。"""
    global _clock
    _clock = c


# ── 便捷函数 ────────────────────────────────────────────────────────

def now() -> datetime:
    """返回当前时间（带时区）。等价于 clock.now()。"""
    return _clock.now()


def now_iso() -> str:
    """返回 ISO 8601 格式的当前时间字符串。"""
    return _clock.now_iso()
