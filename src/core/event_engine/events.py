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
    ORDER_FILLED = "order_filled"
    TRADE_EXECUTED = "trade_executed"
    AGENT_ELIMINATED = "agent_eliminated"
    AGENT_SPAWNED = "agent_spawned"
    MARKET_INITIALIZED = "market_initialized"
    LIQUIDATION = "liquidation"


@dataclass
class Event:
    """事件基类"""
    event_type: EventType
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)
