"""
fast_execution 模块的类型存根文件

提供 IDE 类型提示支持。
"""

from collections import deque
from typing import Any

from src.bio.agents.base import ActionType, Agent
from src.market.matching.matching_engine import MatchingEngine
from src.market.matching.trade import Trade
from src.market.orderbook.orderbook import OrderBook


def execute_non_mm_batch(
    decisions: list[tuple[Agent, ActionType | int, dict[str, Any]]],
    matching_engine: MatchingEngine,
    orderbook: OrderBook,
    recent_trades: deque[Trade],
) -> list[Trade]:
    """
    批量执行非做市商的订单决策

    Args:
        decisions: 决策列表，每个元素为 (agent, action, params)
        matching_engine: MatchingEngine 实例
        orderbook: OrderBook 实例
        recent_trades: deque，用于记录成交

    Returns:
        所有成交记录列表
    """
    ...


def execute_non_mm_batch_with_maker_update(
    decisions: list[tuple[Agent, ActionType | int, dict[str, Any]]],
    matching_engine: MatchingEngine,
    orderbook: OrderBook,
    recent_trades: deque[Trade],
    agent_map: dict[int, Agent],
    tick_trades: list[Trade],
) -> list[Trade]:
    """
    批量执行非做市商的订单决策（包含 maker 账户更新）

    Args:
        decisions: 决策列表，每个元素为 (agent, action, params)
        matching_engine: MatchingEngine 实例
        orderbook: OrderBook 实例
        recent_trades: deque，用于记录成交
        agent_map: Agent ID 到 Agent 对象的映射
        tick_trades: 用于记录本 tick 的成交

    Returns:
        所有成交记录列表
    """
    ...


def execute_non_mm_batch_raw(
    raw_decisions: list[tuple[Agent, int, int, float, float]],
    matching_engine: MatchingEngine,
    orderbook: OrderBook,
    recent_trades: deque[Trade],
    agent_map: dict[int, Agent],
    tick_trades: list[Trade],
    mid_price: float,
) -> list[Trade]:
    """
    批量执行非做市商的原始决策数据（内联数量计算逻辑）

    此函数直接接受原始决策数据，避免创建 Python dict 和调用 Python 方法的开销。
    数量计算逻辑内联到此函数中，避免调用 agent._calculate_order_quantity()。

    Args:
        raw_decisions: 原始决策列表，每个元素为 (agent, action_type_int, side_int, price, quantity_ratio)
            - agent: Agent 实例（非做市商）
            - action_type_int: 动作类型整数（0=HOLD, 1=PLACE_BID, 2=PLACE_ASK, 3=CANCEL, 4=MARKET_BUY, 5=MARKET_SELL）
            - side_int: 方向整数（1=买, 2=卖），用于限价单
            - price: 订单价格
            - quantity_ratio: 数量比例（0.0-1.0）
        matching_engine: MatchingEngine 实例
        orderbook: OrderBook 实例
        recent_trades: deque，用于记录成交
        agent_map: Agent ID 到 Agent 对象的映射
        tick_trades: 用于记录本 tick 的成交
        mid_price: 中间价（用于市价单的数量计算）

    Returns:
        所有成交记录列表
    """
    ...
