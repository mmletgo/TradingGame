"""事件类型和事件基类定义"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    """事件类型枚举"""
    TICK_START = "tick_start"
    TICK_END = "tick_end"
    ORDER_PLACED = "order_placed"
    ORDER_CANCELLED = "order_cancelled"
    TRADE_EXECUTED = "trade_executed"
    LIQUIDATION = "liquidation"


@dataclass
class Event:
    """事件基类"""
    event_type: EventType
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)
    target_ids: set[int] | None = None  # None 表示广播，否则只发送给指定 ID
