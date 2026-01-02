"""
撮合引擎模块

负责订单撮合的核心逻辑，管理订单簿，处理订单提交、撤销和撮合。
"""

from typing import Callable

from src.config.config import MarketConfig
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.market.matching.trade import Trade

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
        """处理订单提交事件

        从事件中获取订单对象，调用 process_order 进行撮合。

        Args:
            event: 订单提交事件，data 中包含 "order" 字段
        """
        order = event.data.get("order")
        if order is not None:
            self.process_order(order)

    def _handle_order_cancelled(self, event: Event) -> None:
        """处理订单撤销事件

        从事件中获取订单ID，从订单簿中撤销该订单。

        Args:
            event: 订单撤销事件，data 中包含 "order_id" 字段
        """
        order_id = event.data.get("order_id")
        if order_id is not None:
            self._orderbook.cancel_order(order_id)

    def register_agent(self, agent_id: int, maker_rate: float, taker_rate: float) -> None:
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
    ) -> tuple[list[Trade], float]:
        """
        通用撮合逻辑

        根据价格优先、时间优先的原则，将订单与对手盘进行撮合。

        Args:
            order: 订单对象
            price_check: 价格检查函数，接收 (order_price, best_price, order_side)，返回是否可以继续撮合
                        如果为 None，则不进行价格检查（市价单行为）

        Returns:
            (trades, remaining): 成交列表和剩余数量
        """
        trades: list[Trade] = []
        remaining = order.quantity - order.filled_quantity

        # 预计算 taker（新订单）的费率信息，避免循环内重复查找
        taker_agent_id = order.agent_id
        taker_rates = self._fee_rates.get(taker_agent_id, (0.0002, 0.0005))
        taker_rate = taker_rates[1]  # taker费率

        while remaining > 0:
            if order.side == OrderSide.BUY:
                # 买单与卖盘撮合
                best_price = self._orderbook.get_best_ask()
                if best_price is None:
                    break  # 对手盘为空
                # 价格检查（限价单需要检查，市价单不检查）
                if price_check is not None and not price_check(order.price, best_price, order.side):
                    break
                side_book = self._orderbook.asks
            else:  # OrderSide.SELL
                # 卖单与买盘撮合
                best_price = self._orderbook.get_best_bid()
                if best_price is None:
                    break  # 对手盘为空
                # 价格检查（限价单需要检查，市价单不检查）
                if price_check is not None and not price_check(order.price, best_price, order.side):
                    break
                side_book = self._orderbook.bids

            # 获取该价格档位
            if best_price not in side_book:
                # 数据不一致：get_best_ask/bid 返回的价格不在 side_book 中
                # 这是一个异常情况，应该修复订单簿状态
                # 为避免无限循环，强制重新获取最佳价格或终止
                self.logger.warning(
                    f"价格档位不一致: best_price={best_price} 不在 side_book 中，终止撮合"
                )
                break

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
                trade_amount = trade_price * trade_qty

                # 计算 taker 手续费（使用预计算的费率）
                taker_fee_amount = trade_amount * taker_rate

                # 确定买卖方和手续费（内联计算以减少函数调用开销）
                # 新订单是 taker，对手盘是 maker
                maker_agent_id = maker_order.agent_id
                maker_rates = self._fee_rates.get(maker_agent_id, (0.0002, 0.0005))
                maker_fee = trade_amount * maker_rates[0]  # maker费率

                if order.side == OrderSide.BUY:
                    buyer_id = taker_agent_id
                    seller_id = maker_agent_id
                    buyer_fee = taker_fee_amount
                    seller_fee = maker_fee
                else:  # OrderSide.SELL
                    buyer_id = maker_agent_id
                    seller_id = taker_agent_id
                    buyer_fee = maker_fee
                    seller_fee = taker_fee_amount

                # 创建成交记录
                trade = Trade(
                    trade_id=self._next_trade_id,
                    price=trade_price,
                    quantity=trade_qty,
                    buyer_id=buyer_id,
                    seller_id=seller_id,
                    buyer_fee=buyer_fee,
                    seller_fee=seller_fee,
                    is_buyer_taker=(order.side == OrderSide.BUY),
                )
                self._next_trade_id += 1
                trades.append(trade)

                # 更新订单簿最新价
                self._orderbook.last_price = trade_price

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
        def limit_price_check(order_price: float, best_price: float, side: OrderSide) -> bool:
            """限价单价格检查：买单价格 >= 卖一价，卖单价格 <= 买一价"""
            if side == OrderSide.BUY:
                return order_price >= best_price
            else:
                return order_price <= best_price

        trades, remaining = self._match_orders(order, limit_price_check)

        # 如果有剩余数量，挂在订单簿上
        if remaining > 0:
            # 更新订单数量为剩余数量，重置已成交数量（新挂单状态）
            order.quantity = remaining
            order.filled_quantity = 0.0
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

        根据订单类型调用对应的撮合函数，并发布成交事件。

        Args:
            order: 订单对象（限价单或市价单）

        Returns:
            本次撮合产生的所有成交记录列表（可能为空）
        """
        # 根据订单类型调用对应撮合函数
        if order.order_type == OrderType.LIMIT:
            trades = self.match_limit_order(order)
        else:  # OrderType.MARKET
            trades = self.match_market_order(order)

        # 如果有成交，发布成交事件（定向发送给买卖双方）
        for trade in trades:
            event = Event(
                event_type=EventType.TRADE_EXECUTED,
                timestamp=trade.timestamp,
                data={
                    "trade_id": trade.trade_id,
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "buyer_id": trade.buyer_id,
                    "seller_id": trade.seller_id,
                    "buyer_fee": trade.buyer_fee,
                    "seller_fee": trade.seller_fee,
                    "is_buyer_taker": trade.is_buyer_taker,
                },
                target_ids={trade.buyer_id, trade.seller_id},
            )
            self._event_bus.publish(event)

        return trades

    def process_order_direct(self, order: Order) -> list[Trade]:
        """直接处理订单（训练模式，不发布事件）

        根据订单类型调用对应的撮合函数，不发布成交事件。
        用于训练模式下绕过事件系统，减少开销。

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

    def cancel_order_direct(self, order_id: int) -> bool:
        """直接撤单（训练模式）

        不发布任何事件，直接从订单簿中撤销订单。

        Args:
            order_id: 订单ID

        Returns:
            是否成功撤单
        """
        return self._orderbook.cancel_order(order_id)

    @property
    def orderbook(self) -> OrderBook:
        """获取订单簿（供直接调用模式使用）"""
        return self._orderbook
