# cython: language_level=3
"""
快速撮合引擎（Cython 加速）

本模块实现了撮合引擎的核心撮合逻辑，使用 Cython 优化以提高性能。
"""


cdef class FastTrade:
    """
    快速成交记录（Cython）

    与 Trade 类保持相同的属性接口，用于 Cython 内部高效处理。

    Attributes:
        trade_id: 成交ID
        price: 成交价格
        quantity: 成交数量
        buyer_id: 买方Agent ID
        seller_id: 卖方Agent ID
        buyer_fee: 买方手续费
        seller_fee: 卖方手续费
        is_buyer_taker: 买方是否为taker
        timestamp: 成交时间戳（固定为 0.0）
    """
    # 属性声明已移至 fast_matching.pxd

    def __init__(self, int trade_id, double price, int quantity,
                 int buyer_id, int seller_id,
                 double buyer_fee, double seller_fee, bint is_buyer_taker):
        self.trade_id = trade_id
        self.price = price
        self.quantity = quantity
        self.buyer_id = buyer_id
        self.seller_id = seller_id
        self.buyer_fee = buyer_fee
        self.seller_fee = seller_fee
        self.is_buyer_taker = is_buyer_taker
        self.timestamp = 0.0  # 训练模式使用固定时间戳


cpdef tuple fast_match_orders(
    object orderbook,
    object order,
    dict fee_rates,
    int next_trade_id,
    bint is_limit_order
):
    """
    快速撮合核心逻辑

    根据价格优先、时间优先的原则，将订单与对手盘进行撮合。

    Args:
        orderbook: OrderBook 实例
        order: 待撮合订单
        fee_rates: dict[int, tuple[float, float]] - agent_id -> (maker_rate, taker_rate)
        next_trade_id: 下一个成交 ID
        is_limit_order: 是否为限价单（True=限价单需要价格检查，False=市价单不检查）

    Returns:
        (trades, remaining, next_trade_id): 成交列表、剩余数量、更新后的 trade_id
    """
    cdef list trades = []
    cdef int remaining = order.quantity - order.filled_quantity

    # 预计算 taker 费率
    cdef int taker_agent_id = order.agent_id
    cdef tuple taker_rates = fee_rates.get(taker_agent_id, (0.0002, 0.0005))
    cdef double taker_rate = taker_rates[1]

    cdef double order_price = order.price
    cdef int order_side_value = order.side  # OrderSide.BUY=1, SELL=-1
    cdef bint is_buy = (order_side_value == 1)

    cdef double best_price
    cdef object best_price_obj
    cdef object side_book
    cdef object price_level
    cdef object maker_order
    cdef int maker_remaining, trade_qty, maker_remaining_after, zombie_remaining
    cdef double trade_price, trade_amount
    cdef int maker_agent_id, buyer_id, seller_id
    cdef double maker_fee, taker_fee_amount, buyer_fee, seller_fee
    cdef tuple maker_rates
    cdef object current_level

    while remaining > 0:
        # 获取最优价格
        if is_buy:
            best_price_obj = orderbook.get_best_ask()
            if best_price_obj is None:
                break
            best_price = best_price_obj
            # 限价单价格检查：买单价格 >= 卖一价
            if is_limit_order and order_price < best_price:
                break
            side_book = orderbook.asks
        else:
            best_price_obj = orderbook.get_best_bid()
            if best_price_obj is None:
                break
            best_price = best_price_obj
            # 限价单价格检查：卖单价格 <= 买一价
            if is_limit_order and order_price > best_price:
                break
            side_book = orderbook.bids

        # 获取价格档位
        if best_price not in side_book:
            break

        price_level = side_book[best_price]

        # 遍历该价格档位的订单（按时间优先顺序）
        for maker_order in list(price_level.orders.values()):
            if remaining <= 0:
                break

            # 僵尸订单检测与清理
            if maker_order.order_id not in orderbook.order_map:
                # 直接从 price_level.orders 中移除
                del price_level.orders[maker_order.order_id]
                # 更新 total_quantity（使用未成交数量）
                zombie_remaining = maker_order.quantity - maker_order.filled_quantity
                price_level.total_quantity = max(0, price_level.total_quantity - zombie_remaining)
                continue

            # 计算成交数量
            maker_remaining = maker_order.quantity - maker_order.filled_quantity
            trade_qty = min(remaining, maker_remaining)

            # 成交价格 = maker 订单价格
            trade_price = maker_order.price
            trade_amount = trade_price * trade_qty

            # 计算手续费
            taker_fee_amount = trade_amount * taker_rate
            maker_agent_id = maker_order.agent_id
            maker_rates = fee_rates.get(maker_agent_id, (0.0002, 0.0005))
            maker_fee = trade_amount * maker_rates[0]

            # 确定买卖方
            if is_buy:
                buyer_id = taker_agent_id
                seller_id = maker_agent_id
                buyer_fee = taker_fee_amount
                seller_fee = maker_fee
            else:
                buyer_id = maker_agent_id
                seller_id = taker_agent_id
                buyer_fee = maker_fee
                seller_fee = taker_fee_amount

            # 创建成交记录
            trade = FastTrade(
                trade_id=next_trade_id,
                price=trade_price,
                quantity=trade_qty,
                buyer_id=buyer_id,
                seller_id=seller_id,
                buyer_fee=buyer_fee,
                seller_fee=seller_fee,
                is_buyer_taker=is_buy
            )
            next_trade_id += 1
            trades.append(trade)

            # 更新订单簿最新价
            orderbook.last_price = trade_price

            # 更新已成交数量
            order.filled_quantity += trade_qty
            maker_order.filled_quantity += trade_qty
            remaining -= trade_qty

            # 更新档位 total_quantity
            price_level.total_quantity -= trade_qty

            # maker 订单完全成交则移除
            maker_remaining_after = maker_order.quantity - maker_order.filled_quantity
            if maker_remaining_after == 0:
                orderbook.cancel_order(maker_order.order_id)

        # 空档位清理
        if best_price in side_book:
            current_level = side_book[best_price]
            if len(current_level.orders) == 0:
                del side_book[best_price]
                orderbook._depth_dirty = True
                continue

        if best_price not in side_book:
            continue

    return trades, remaining, next_trade_id


cdef class FastMatchingEngine:
    """
    快速撮合引擎（Cython）

    交易市场的核心组件，负责：
    - 管理订单簿
    - 处理订单提交和撤销
    - 执行撮合逻辑，生成成交记录

    与 MatchingEngine 保持相同的接口，但使用 Cython 优化以提高性能。
    核心路径不包含日志以避免性能开销。

    Attributes:
        orderbook: OrderBook 实例
        _next_trade_id: 下一个成交ID
        _fee_rates: Agent 费率映射 (agent_id -> (maker_rate, taker_rate))
        _tick_size: 最小变动单位
    """
    # 属性声明已移至 fast_matching.pxd

    def __init__(self, object config):
        """
        初始化撮合引擎

        Args:
            config: MarketConfig 配置对象
        """
        # 延迟导入以避免循环依赖
        from src.market.orderbook.orderbook import OrderBook

        self._tick_size = config.tick_size
        self.orderbook = OrderBook(tick_size=self._tick_size)
        self._next_trade_id = 1
        self._fee_rates = {}

    cpdef void register_agent(self, int agent_id, double maker_rate, double taker_rate):
        """
        注册/更新 Agent 的费率配置

        Args:
            agent_id: Agent ID
            maker_rate: 挂单费率（如 0.0002 表示万2）
            taker_rate: 吃单费率（如 0.0005 表示万5）
        """
        self._fee_rates[agent_id] = (maker_rate, taker_rate)

    cpdef double calculate_fee(self, int agent_id, double amount, bint is_maker):
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
        """
        cdef tuple rates = self._fee_rates.get(agent_id, (0.0002, 0.0005))
        cdef double rate
        if is_maker:
            rate = rates[0]
        else:
            rate = rates[1]
        return amount * rate

    cpdef list match_limit_order(self, object order):
        """
        限价单撮合

        根据价格优先、时间优先的原则，将限价单与对手盘进行撮合。
        无法完全成交的剩余部分挂在订单簿上。

        Args:
            order: 限价订单对象

        Returns:
            本次撮合产生的所有 FastTrade 成交记录列表
        """
        cdef list trades
        cdef int remaining

        trades, remaining, self._next_trade_id = fast_match_orders(
            self.orderbook,
            order,
            self._fee_rates,
            self._next_trade_id,
            True  # is_limit_order = True
        )

        # 如果有剩余数量，挂在订单簿上
        if remaining > 0:
            # 更新订单数量为剩余数量，重置已成交数量（新挂单状态）
            order.quantity = remaining
            order.filled_quantity = 0
            self.orderbook.add_order(order)

        return trades

    cpdef list match_market_order(self, object order):
        """
        市价单撮合

        根据价格优先、时间优先的原则，将市价单与对手盘进行撮合。
        吃掉对手盘直到完全成交或对手盘为空。
        市价单不会挂在订单簿上，未成交部分直接丢弃。

        Args:
            order: 市价订单对象

        Returns:
            本次撮合产生的所有 FastTrade 成交记录列表
        """
        cdef list trades
        cdef int remaining

        # 市价单不检查价格，is_limit_order = False
        trades, remaining, self._next_trade_id = fast_match_orders(
            self.orderbook,
            order,
            self._fee_rates,
            self._next_trade_id,
            False  # is_limit_order = False
        )

        return trades

    cpdef list process_order(self, object order):
        """
        处理订单，执行撮合

        根据订单类型调用对应的撮合函数。
        OrderType.LIMIT = 1, OrderType.MARKET = 2

        Args:
            order: 订单对象（限价单或市价单）

        Returns:
            本次撮合产生的所有 FastTrade 成交记录列表（可能为空）
        """
        # OrderType.LIMIT = 1
        if order.order_type == 1:
            return self.match_limit_order(order)
        else:
            # OrderType.MARKET = 2
            return self.match_market_order(order)

    cpdef bint cancel_order(self, int order_id):
        """
        撤单

        从订单簿中撤销订单。

        Args:
            order_id: 订单ID

        Returns:
            是否成功撤单
        """
        cdef object result = self.orderbook.cancel_order(order_id)
        return result is not None

    @property
    def tick_size(self) -> float:
        """获取最小变动单位"""
        return self._tick_size

    @property
    def next_trade_id(self) -> int:
        """获取下一个成交ID"""
        return self._next_trade_id

    @property
    def fee_rates(self) -> dict:
        """获取费率配置"""
        return self._fee_rates
