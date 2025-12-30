"""事件类型和事件基类定义"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class EventType(IntEnum):
    """事件类型枚举"""
    TICK_START = 1
    TICK_END = 2
    ORDER_PLACED = 3
    ORDER_CANCELLED = 4
    TRADE_EXECUTED = 5
    LIQUIDATION = 6


@dataclass
class Event:
    """事件基类"""
    event_type: EventType
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)
    target_ids: set[int] | None = None  # None 表示广播，否则只发送给指定 ID
