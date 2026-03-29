# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language = c++
"""
快速撮合引擎（Cython + C++ 加速）

本模块实现了撮合引擎的核心撮合逻辑，使用 C++ 容器和 COrder 结构体
直接操作订单簿，避免 Python 对象创建开销。
"""

from libc.stdlib cimport free
from libcpp.map cimport map as cppmap
from libcpp.unordered_map cimport unordered_map
from cpython.ref cimport Py_DECREF, PyObject
from cython.operator cimport dereference as deref, preincrement as inc, predecrement as dec

from src.market.orderbook.orderbook cimport OrderBook, PriceLevel, COrder


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
                 long long buyer_id, long long seller_id,
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


cdef class FastMatchingEngine:
    """
    快速撮合引擎（Cython + C++）

    交易市场的核心组件，负责：
    - 管理订单簿
    - 处理订单提交和撤销
    - 执行撮合逻辑，生成成交记录

    核心路径使用 process_order_raw 直接操作 C++ 数据结构，
    避免创建 Python Order 对象。

    Attributes:
        _orderbook: OrderBook 实例
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
        from src.market.orderbook.orderbook import OrderBook

        self._tick_size = config.tick_size
        self._orderbook = OrderBook(tick_size=self._tick_size)
        self._next_trade_id = 1
        self._fee_rates = {}

    cpdef void register_agent(self, long long agent_id, double maker_rate, double taker_rate):
        """
        注册/更新 Agent 的费率配置

        Args:
            agent_id: Agent ID
            maker_rate: 挂单费率（如 0.0002 表示万2）
            taker_rate: 吃单费率（如 0.0005 表示万5）
        """
        self._fee_rates[agent_id] = (maker_rate, taker_rate)

    cpdef double calculate_fee(self, long long agent_id, double amount, bint is_maker):
        """
        计算手续费

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
            rate = <double>rates[0]
        else:
            rate = <double>rates[1]
        return amount * rate

    cpdef list process_order_raw(self, long long order_id, long long agent_id,
                                 int side, int order_type, double price, int quantity):
        """不创建 Order 对象的快速撮合

        直接操作 C++ 数据结构进行撮合，避免 Python 对象创建开销。

        Args:
            order_id: 订单 ID
            agent_id: Agent ID
            side: 1=BUY, -1=SELL
            order_type: 1=LIMIT, 2=MARKET
            price: 价格
            quantity: 数量

        Returns:
            本次撮合产生的所有 FastTrade 成交记录列表
        """
        cdef OrderBook orderbook = <OrderBook>self._orderbook
        cdef list trades = []
        cdef int remaining = quantity
        cdef bint is_buy = (side == 1)
        cdef bint is_limit = (order_type == 1)
        cdef int filled_quantity = 0

        # 预计算 taker 费率
        cdef tuple taker_rates = self._fee_rates.get(agent_id, (0.0002, 0.0005))
        cdef double taker_rate = <double>taker_rates[1]

        # 价格归一化
        cdef double normalized_price = round(price / self._tick_size) * self._tick_size
        normalized_price = round(normalized_price, 10)

        # C++ map 引用
        cdef cppmap[double, PyObject*]* side_map
        cdef cppmap[double, PyObject*].iterator best_it
        cdef double best_price
        cdef PriceLevel price_level
        cdef COrder* maker
        cdef COrder* next_maker
        cdef int maker_remaining, trade_qty, maker_remaining_after
        cdef double trade_price, trade_amount
        cdef long long maker_agent_id
        cdef long long buyer_id_val, seller_id_val
        cdef double maker_fee, taker_fee_amount, buyer_fee_val, seller_fee_val
        cdef tuple maker_rates

        while remaining > 0:
            # 获取对手盘
            if is_buy:
                if orderbook.asks_map.empty():
                    break
                best_it = orderbook.asks_map.begin()
                best_price = deref(best_it).first
                if is_limit and normalized_price < best_price:
                    break
                side_map = &orderbook.asks_map
            else:
                if orderbook.bids_map.empty():
                    break
                best_it = orderbook.bids_map.end()
                dec(best_it)
                best_price = deref(best_it).first
                if is_limit and normalized_price > best_price:
                    break
                side_map = &orderbook.bids_map

            price_level = <PriceLevel>deref(best_it).second

            # 遍历链表匹配
            maker = price_level.head
            while maker != NULL and remaining > 0:
                next_maker = maker.next

                # 僵尸订单检测
                if orderbook.order_map_cpp.count(maker.order_id) == 0:
                    price_level.remove_order(maker.order_id)
                    free(maker)
                    maker = next_maker
                    continue

                maker_remaining = maker.quantity - maker.filled_quantity
                trade_qty = min(remaining, maker_remaining)
                trade_price = maker.price
                trade_amount = trade_price * trade_qty

                # 计算手续费
                taker_fee_amount = trade_amount * taker_rate
                maker_agent_id = maker.agent_id
                maker_rates = self._fee_rates.get(maker_agent_id, (0.0002, 0.0005))
                maker_fee = trade_amount * <double>maker_rates[0]

                # 确定买卖方
                if is_buy:
                    buyer_id_val = agent_id
                    seller_id_val = maker_agent_id
                    buyer_fee_val = taker_fee_amount
                    seller_fee_val = maker_fee
                else:
                    buyer_id_val = maker_agent_id
                    seller_id_val = agent_id
                    buyer_fee_val = maker_fee
                    seller_fee_val = taker_fee_amount

                # 创建 FastTrade
                trade = FastTrade(
                    trade_id=self._next_trade_id,
                    price=trade_price,
                    quantity=trade_qty,
                    buyer_id=buyer_id_val,
                    seller_id=seller_id_val,
                    buyer_fee=buyer_fee_val,
                    seller_fee=seller_fee_val,
                    is_buyer_taker=is_buy
                )
                self._next_trade_id += 1
                trades.append(trade)

                # 更新 last_price
                orderbook.last_price = trade_price

                # 更新已成交数量
                filled_quantity += trade_qty
                maker.filled_quantity += trade_qty
                remaining -= trade_qty
                price_level.total_quantity -= trade_qty

                # maker 完全成交则移除
                maker_remaining_after = maker.quantity - maker.filled_quantity
                if maker_remaining_after == 0:
                    orderbook.order_map_cpp.erase(maker.order_id)
                    price_level.remove_order(maker.order_id)
                    free(maker)

                maker = next_maker

            # 空档位清理
            if price_level.is_empty():
                Py_DECREF(<object>deref(best_it).second)
                side_map.erase(best_it)
                orderbook._depth_dirty = True

        # 限价单剩余部分挂入订单簿
        if is_limit and remaining > 0:
            orderbook.add_order_raw(order_id, agent_id, side, order_type,
                                    normalized_price, remaining)

        return trades

    cpdef list process_order(self, object order):
        """
        处理 Python Order 对象，执行撮合

        兼容旧接口，内部委托给 process_order_raw。

        Args:
            order: 订单对象（限价单或市价单）

        Returns:
            本次撮合产生的所有 FastTrade 成交记录列表
        """
        return self.process_order_raw(
            order.order_id, order.agent_id,
            int(order.side), int(order.order_type),
            order.price, order.quantity
        )

    cpdef bint cancel_order(self, long long order_id):
        """
        撤单

        从订单簿中撤销订单。

        Args:
            order_id: 订单ID

        Returns:
            是否成功撤单
        """
        return (<OrderBook>self._orderbook).cancel_order_fast(order_id)

    @property
    def orderbook(self):
        """获取订单簿（兼容旧接口）"""
        return self._orderbook

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
