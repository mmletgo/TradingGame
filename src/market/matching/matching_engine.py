"""
撮合引擎模块

负责订单撮合的核心逻辑，管理订单簿，处理订单提交、撤销和撮合。
"""

from src.config.config import MarketConfig
from src.core.event_engine.event_bus import EventBus
from src.core.event_engine.events import Event, EventType
from src.core.log_engine.logger import get_logger
from src.market.orderbook.orderbook import OrderBook


class MatchingEngine:
    """
    撮合引擎

    交易市场的核心组件，负责：
    - 管理订单簿
    - 处理订单提交和撤销
    - 执行撮合逻辑，生成成交记录
    - 发布成交事件

    Attributes:
        _event_bus: 事件总线
        _config: 市场配置
        _orderbook: 订单簿实例
        _logger: 日志器
        _next_trade_id: 下一个成交ID
    """

    def __init__(self, event_bus: EventBus, config: MarketConfig) -> None:
        """
        初始化撮合引擎

        Args:
            event_bus: 事件总线，用于发布成交事件
            config: 市场配置
        """
        self._event_bus = event_bus
        self._config = config
        self._orderbook = OrderBook(tick_size=config.tick_size)
        self._logger = get_logger(__name__)
        self._next_trade_id: int = 1
        self._fee_rates: dict[int, tuple[float, float]] = {}

        # 订阅订单事件
        event_bus.subscribe(EventType.ORDER_PLACED, self._handle_order_placed)
        event_bus.subscribe(
            EventType.ORDER_CANCELLED, self._handle_order_cancelled
        )

        self._logger.info("撮合引擎初始化完成")

    def _handle_order_placed(self, event: Event) -> None:
        """处理订单提交事件（占位）"""
        pass

    def _handle_order_cancelled(self, event: Event) -> None:
        """处理订单撤销事件（占位）"""
        pass

    def register_agent(self, agent_id: int, maker_rate: float, taker_rate: float) -> None:
        """
        注册/更新 Agent 的费率配置

        Args:
            agent_id: Agent ID
            maker_rate: 挂单费率（如 0.0002 表示万2）
            taker_rate: 吃单费率（如 0.0005 表示万5）
        """
        self._fee_rates[agent_id] = (maker_rate, taker_rate)

    def unregister_agent(self, agent_id: int) -> None:
        """
        移除 Agent 的费率配置（淘汰时调用）

        Args:
            agent_id: Agent ID
        """
        self._fee_rates.pop(agent_id, None)

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

    def match_limit_order(self, order: "Order") -> list["Trade"]:
        """
        限价单撮合

        根据价格优先、时间优先的原则，将限价单与对手盘进行撮合。
        无法完全成交的剩余部分挂在订单簿上。

        Args:
            order: 限价订单对象

        Returns:
            本次撮合产生的所有成交记录列表
        """
        from src.market.orderbook.order import OrderSide
        from src.market.matching.trade import Trade

        trades: list[Trade] = []
        remaining = order.quantity - order.filled_quantity

        while remaining > 0:
            if order.side == OrderSide.BUY:
                # 买单与卖盘撮合
                best_price = self._orderbook.get_best_ask()
                if best_price is None or order.price < best_price:
                    break  # 无法撮合，价格不满足
                side_book = self._orderbook.asks
            else:  # OrderSide.SELL
                # 卖单与买盘撮合
                best_price = self._orderbook.get_best_bid()
                if best_price is None or order.price > best_price:
                    break  # 无法撮合，价格不满足
                side_book = self._orderbook.bids

            # 获取该价格档位
            if best_price not in side_book:
                continue

            price_level = side_book[best_price]

            # 遍历该价格档位的订单（按时间优先顺序）
            for maker_order in list(price_level.orders.values()):
                if remaining <= 0:
                    break

                # 计算成交量
                maker_remaining = maker_order.quantity - maker_order.filled_quantity
                if maker_remaining <= 0:
                    continue

                trade_qty = min(remaining, maker_remaining)

                # 成交价格 = maker订单价格（对手盘价格）
                trade_price = maker_order.price

                # 确定买卖方和手续费类型
                if order.side == OrderSide.BUY:
                    buyer_id = order.agent_id
                    seller_id = maker_order.agent_id
                    # 新订单是 taker，对手盘是 maker
                    buyer_fee = self.calculate_fee(buyer_id, trade_price * trade_qty, is_maker=False)
                    seller_fee = self.calculate_fee(seller_id, trade_price * trade_qty, is_maker=True)
                else:  # OrderSide.SELL
                    buyer_id = maker_order.agent_id
                    seller_id = order.agent_id
                    # 对手盘是 maker，新订单是 taker
                    buyer_fee = self.calculate_fee(buyer_id, trade_price * trade_qty, is_maker=True)
                    seller_fee = self.calculate_fee(seller_id, trade_price * trade_qty, is_maker=False)

                # 创建成交记录
                trade = Trade(
                    trade_id=self._next_trade_id,
                    price=trade_price,
                    quantity=trade_qty,
                    buyer_id=buyer_id,
                    seller_id=seller_id,
                    buyer_fee=buyer_fee,
                    seller_fee=seller_fee,
                )
                self._next_trade_id += 1
                trades.append(trade)

                # 更新已成交数量
                order.filled_quantity += trade_qty
                maker_order.filled_quantity += trade_qty
                remaining -= trade_qty

                # 如果 maker 订单完全成交，从订单簿移除
                if maker_order.filled_quantity >= maker_order.quantity:
                    self._orderbook.cancel_order(maker_order.order_id)

            # 检查该价格档位是否已空
            if best_price not in side_book:
                continue  # 档位已空，继续下一个价格

        # 如果有剩余数量，挂在订单簿上
        if remaining > 0:
            # 更新订单数量为剩余数量，重置已成交数量（新挂单状态）
            order.quantity = remaining
            order.filled_quantity = 0.0
            self._orderbook.add_order(order)

        return trades
