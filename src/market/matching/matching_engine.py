"""
撮合引擎模块

负责订单撮合的核心逻辑，管理订单簿，处理订单提交、撤销和撮合。
"""

from typing import Callable

from src.config.config import MarketConfig
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.market.matching.trade import Trade
from src.market.matching.fast_matching import fast_match_orders, FastTrade

from src.core.log_engine.logger import get_logger
from src.market.orderbook.orderbook import OrderBook


class MatchingEngine:
    """
    撮合引擎

    交易市场的核心组件，负责：
    - 管理订单簿
    - 处理订单提交和撤销
    - 执行撮合逻辑，生成成交记录

    Attributes:
        _config: 市场配置
        _orderbook: 订单簿实例
        _logger: 日志器
        _next_trade_id: 下一个成交ID
    """

    def __init__(self, config: MarketConfig) -> None:
        """
        初始化撮合引擎

        Args:
            config: 市场配置
        """
        self._config = config
        self._orderbook = OrderBook(tick_size=config.tick_size)
        self._logger = get_logger(__name__)
        self._next_trade_id: int = 1
        self._fee_rates: dict[int, tuple[float, float]] = {}

        self._logger.info("撮合引擎初始化完成")

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
        self._fee_rates[agent_id] = (maker_rate, taker_rate)

    def calculate_fee(self, agent_id: int, amount: float, is_maker: bool) -> float:
        """
        计算手续费

        根据 Agent 类型和订单类型（挂单/吃单）计算手续费。

        费率配置（默认）：
        - 散户: 挂单万2 (0.0002)，吃单万5 (0.0005)
        - 庄家: 挂单0，吃单万1 (0.0001)
        - 做市商: 挂单0，吃单万1 (0.0001)

        Args:
            agent_id: Agent ID
            amount: 成交金额
            is_maker: True 表示挂单，False 表示吃单

        Returns:
            手续费金额

        Raises:
            ValueError: 如果 agent_id 未注册且金额为正
        """
        rates = self._fee_rates.get(agent_id)
        if rates is None:
            # 未注册的 Agent 使用默认散户费率
            rates = (0.0002, 0.0005)

        maker_rate, taker_rate = rates
        rate = maker_rate if is_maker else taker_rate
        return amount * rate

    def _match_orders(
        self,
        order: Order,
        price_check: Callable[[float, float, OrderSide], bool] | None = None,
    ) -> tuple[list[FastTrade], int]:
        """
        通用撮合逻辑（委托给 Cython 实现）

        根据价格优先、时间优先的原则，将订单与对手盘进行撮合。

        Args:
            order: 订单对象
            price_check: 价格检查函数，接收 (order_price, best_price, order_side)，返回是否可以继续撮合
                        如果为 None，则不进行价格检查（市价单行为）

        Returns:
            (trades, remaining): 成交列表和剩余数量
        """
        is_limit_order = price_check is not None

        trades, remaining, self._next_trade_id = fast_match_orders(
            self._orderbook,
            order,
            self._fee_rates,
            self._next_trade_id,
            is_limit_order
        )

        return trades, remaining

    def match_limit_order(self, order: Order) -> list[Trade]:
        """
        限价单撮合

        根据价格优先、时间优先的原则，将限价单与对手盘进行撮合。
        无法完全成交的剩余部分挂在订单簿上。

        Args:
            order: 限价订单对象

        Returns:
            本次撮合产生的所有成交记录列表
        """

        def limit_price_check(
            order_price: float, best_price: float, side: OrderSide
        ) -> bool:
            """限价单价格检查：买单价格 >= 卖一价，卖单价格 <= 买一价"""
            if side == OrderSide.BUY:
                return order_price >= best_price
            else:
                return order_price <= best_price

        trades, remaining = self._match_orders(
            order=order, price_check=limit_price_check
        )

        # 如果有剩余数量，挂在订单簿上
        if remaining > 0:
            # 更新订单数量为剩余数量，重置已成交数量（新挂单状态）
            order.quantity = remaining
            order.filled_quantity = 0
            self._orderbook.add_order(order)

        return trades

    def match_market_order(self, order: Order) -> list[Trade]:
        """
        市价单撮合

        根据价格优先、时间优先的原则，将市价单与对手盘进行撮合。
        吃掉对手盘直到完全成交或对手盘为空。
        市价单不会挂在订单簿上，未成交部分直接丢弃。

        Args:
            order: 市价订单对象

        Returns:
            本次撮合产生的所有成交记录列表
        """
        # 市价单不检查价格，price_check 为 None
        trades, _ = self._match_orders(order, None)
        return trades

    def process_order(self, order: Order) -> list[Trade]:
        """
        处理订单，执行撮合

        根据订单类型调用对应的撮合函数。

        Args:
            order: 订单对象（限价单或市价单）

        Returns:
            本次撮合产生的所有成交记录列表（可能为空）
        """
        if order.order_type == OrderType.LIMIT:
            trades = self.match_limit_order(order)
        else:  # OrderType.MARKET
            trades = self.match_market_order(order)
        return trades

    def cancel_order(self, order_id: int) -> bool:
        """撤单

        从订单簿中撤销订单。

        Args:
            order_id: 订单ID

        Returns:
            是否成功撤单
        """
        return self._orderbook.cancel_order(order_id)

    @property
    def orderbook(self) -> OrderBook:
        """获取订单簿"""
        return self._orderbook
