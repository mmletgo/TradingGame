"""事件类型和事件基类定义"""

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


class Event:
    """事件基类

    使用 __slots__ 优化内存占用和属性访问速度。

    Attributes:
        event_type: 事件类型
        timestamp: 时间戳
        data: 事件数据字典
        target_ids: 目标 ID 集合，None 表示广播
    """
    __slots__ = ('event_type', 'timestamp', 'data', 'target_ids')

    def __init__(
        self,
        event_type: EventType,
        timestamp: float,
        data: dict[str, Any] | None = None,
        target_ids: set[int] | None = None,
    ) -> None:
        self.event_type: EventType = event_type
        self.timestamp: float = timestamp
        self.data: dict[str, Any] = data if data is not None else {}
        self.target_ids: set[int] | None = target_ids
