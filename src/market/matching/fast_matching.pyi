"""
快速撮合引擎类型存根文件
"""

from typing import Any

from src.config.config import MarketConfig


class FastTrade:
    """快速成交记录（Cython）"""
    trade_id: int
    price: float
    quantity: int
    buyer_id: int
    seller_id: int
    buyer_fee: float
    seller_fee: float
    is_buyer_taker: bool
    timestamp: float

    def __init__(
        self,
        trade_id: int,
        price: float,
        quantity: int,
        buyer_id: int,
        seller_id: int,
        buyer_fee: float,
        seller_fee: float,
        is_buyer_taker: bool,
    ) -> None: ...


class FastMatchingEngine:
    """
    快速撮合引擎（Cython）

    交易市场的核心组件，负责：
    - 管理订单簿
    - 处理订单提交和撤销
    - 执行撮合逻辑，生成成交记录

    与 MatchingEngine 保持相同的接口，但使用 Cython 优化以提高性能。
    """
    orderbook: Any
    _next_trade_id: int
    _fee_rates: dict[int, tuple[float, float]]
    _tick_size: float

    def __init__(self, config: MarketConfig) -> None:
        """
        初始化撮合引擎

        Args:
            config: MarketConfig 配置对象
        """
        ...

    def register_agent(
        self, agent_id: int, maker_rate: float, taker_rate: float
    ) -> None:
        """
        注册/更新 Agent 的费率配置

        Args:
            agent_id: Agent ID
            maker_rate: 挂单费率（如 0.0002 表示万2）
            taker_rate: 吃单费率（如 0.0005 表示万5）
        """
        ...

    def calculate_fee(self, agent_id: int, amount: float, is_maker: bool) -> float:
        """
        计算手续费

        Args:
            agent_id: Agent ID
            amount: 成交金额
            is_maker: True 表示挂单，False 表示吃单

        Returns:
            手续费金额
        """
        ...

    def match_limit_order(self, order: Any) -> list[FastTrade]:
        """
        限价单撮合

        Args:
            order: 限价订单对象

        Returns:
            本次撮合产生的所有 FastTrade 成交记录列表
        """
        ...

    def match_market_order(self, order: Any) -> list[FastTrade]:
        """
        市价单撮合

        Args:
            order: 市价订单对象

        Returns:
            本次撮合产生的所有 FastTrade 成交记录列表
        """
        ...

    def process_order(self, order: Any) -> list[FastTrade]:
        """
        处理订单，执行撮合

        Args:
            order: 订单对象（限价单或市价单）

        Returns:
            本次撮合产生的所有 FastTrade 成交记录列表（可能为空）
        """
        ...

    def cancel_order(self, order_id: int) -> bool:
        """
        撤单

        Args:
            order_id: 订单ID

        Returns:
            是否成功撤单
        """
        ...

    @property
    def tick_size(self) -> float:
        """获取最小变动单位"""
        ...

    @property
    def next_trade_id(self) -> int:
        """获取下一个成交ID"""
        ...

    @property
    def fee_rates(self) -> dict[int, tuple[float, float]]:
        """获取费率配置"""
        ...


def fast_match_orders(
    orderbook: Any,
    order: Any,
    fee_rates: dict[int, tuple[float, float]],
    next_trade_id: int,
    is_limit_order: bool,
) -> tuple[list[FastTrade], int, int]:
    """
    快速撮合核心逻辑

    Args:
        orderbook: OrderBook 实例
        order: 待撮合订单
        fee_rates: agent_id -> (maker_rate, taker_rate)
        next_trade_id: 下一个成交 ID
        is_limit_order: 是否为限价单

    Returns:
        (trades, remaining, next_trade_id): 成交列表、剩余数量、更新后的 trade_id
    """
    ...
