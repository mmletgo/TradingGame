# cython: language_level=3
"""快速账户模块（Cython 加速）

本模块实现了 FastAccount cdef class，提供高性能的账户管理功能。
"""

from libc.math cimport INFINITY

from src.market.account.position cimport Position
from src.market.matching.fast_matching cimport FastTrade


# Agent 类型常量（与 AgentType 枚举对应）
DEF RETAIL_PRO = 0
DEF MARKET_MAKER = 1


cdef class FastAccount:
    """快速账户类（Cython 加速）

    记录 Agent 的交易账户状态，包括余额、持仓、杠杆倍数和手续费率等。
    使用 Cython cdef class 实现，提供比纯 Python 更高的性能。

    Attributes:
        agent_id: Agent ID
        agent_type: Agent 类型（整数：0=RETAIL_PRO, 1=MARKET_MAKER）
        initial_balance: 初始余额
        balance: 当前余额
        position: 持仓对象（Position cdef class）
        leverage: 杠杆倍数
        maintenance_margin_rate: 维持保证金率
        maker_fee_rate: 挂单手续费率
        taker_fee_rate: 吃单手续费率
        pending_order_id: 当前挂单 ID（-1 表示无挂单）
        maker_volume: 作为 maker 的累计成交量
        volatility_contribution: 作为 taker 的价格冲击累计
    """

    cdef public int agent_id
    cdef public int agent_type
    cdef public double initial_balance
    cdef public double balance
    cdef public Position position
    cdef public double leverage
    cdef public double maintenance_margin_rate
    cdef public double maker_fee_rate
    cdef public double taker_fee_rate
    cdef public int pending_order_id
    cdef public int maker_volume
    cdef public int total_volume
    cdef public int trade_count
    cdef public double volatility_contribution

    def __init__(
        self,
        int agent_id,
        int agent_type,
        double initial_balance,
        double leverage,
        double maintenance_margin_rate,
        double maker_fee_rate,
        double taker_fee_rate
    ):
        """创建快速账户

        Args:
            agent_id: Agent ID
            agent_type: Agent 类型整数（0=RETAIL_PRO, 1=MARKET_MAKER）
            initial_balance: 初始余额
            leverage: 杠杆倍数
            maintenance_margin_rate: 维持保证金率
            maker_fee_rate: 挂单手续费率
            taker_fee_rate: 吃单手续费率
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = Position()
        self.leverage = leverage
        self.maintenance_margin_rate = maintenance_margin_rate
        self.maker_fee_rate = maker_fee_rate
        self.taker_fee_rate = taker_fee_rate
        self.pending_order_id = -1  # -1 表示 None
        self.maker_volume = 0
        self.total_volume = 0
        self.trade_count = 0
        self.volatility_contribution = 0.0

    cpdef double get_equity(self, double current_price):
        """计算净值

        净值 = 余额 + 浮动盈亏，用于评估账户总资产状态。

        Args:
            current_price: 当前市场价格

        Returns:
            账户净值
        """
        return self.balance + self.position.get_unrealized_pnl(current_price)

    cpdef double get_margin_ratio(self, double current_price):
        """计算保证金率

        保证金率 = 净值 / 持仓市值，用于评估账户强平风险。
        无持仓时返回 INFINITY，表示无强平风险。

        Args:
            current_price: 当前市场价格

        Returns:
            保证金率（无持仓时为 INFINITY）
        """
        cdef double equity = self.get_equity(current_price)
        cdef int abs_quantity = abs(self.position.quantity)
        cdef double position_value = <double>abs_quantity * current_price

        if abs_quantity == 0:
            return INFINITY

        return equity / position_value

    cpdef bint check_liquidation(self, double current_price):
        """检查是否需要强制平仓

        当保证金率低于维持保证金率时，需要强制平仓以控制风险。

        Args:
            current_price: 当前市场价格

        Returns:
            True 表示需要强制平仓，False 表示不需要
        """
        cdef double margin_ratio = self.get_margin_ratio(current_price)
        return margin_ratio < self.maintenance_margin_rate

    cpdef void on_trade(self, FastTrade trade, bint is_buyer):
        """处理成交回报

        根据成交记录更新持仓、扣除手续费、更新余额、累计maker成交量。

        Args:
            trade: FastTrade 成交记录对象
            is_buyer: 是否为买方（True=买方，False=卖方）
        """
        cdef int side
        cdef double fee
        cdef double realized_pnl

        # 确定成交方向和手续费
        if is_buyer:
            side = 1  # BUY
            fee = trade.buyer_fee
        else:
            side = -1  # SELL
            fee = trade.seller_fee

        # 累加成交量和成交次数（无条件）
        self.total_volume += trade.quantity
        self.trade_count += 1

        # 判断是否为 maker 并累加成交量
        # is_buyer_taker=True: 买方是 taker，卖方是 maker
        # is_buyer_taker=False: 卖方是 taker，买方是 maker
        if trade.is_buyer_taker:
            if not is_buyer:  # 当前是卖方，即 maker
                self.maker_volume += trade.quantity
        else:
            if is_buyer:  # 当前是买方，即 maker
                self.maker_volume += trade.quantity

        # 更新持仓，获取已实现盈亏
        realized_pnl = self.position.update(side, trade.quantity, trade.price)

        # 将已实现盈亏加到余额，并扣除手续费
        self.balance += realized_pnl - fee

    cpdef double on_adl_trade(self, int quantity, double price, bint is_taker):
        """处理 ADL 成交

        ADL 成交不收取手续费。

        Args:
            quantity: 成交数量（正数）
            price: 成交价格
            is_taker: 是否为被强平方（True=被强平方，False=ADL对手方）

        Returns:
            已实现盈亏
        """
        cdef int abs_position = abs(self.position.quantity)
        cdef int actual_quantity = min(quantity, abs_position)
        cdef double realized_pnl

        if actual_quantity <= 0:
            return 0.0

        if self.position.quantity > 0:
            # 多头平仓/被减仓
            realized_pnl = (price - self.position.avg_price) * <double>actual_quantity
            self.position.quantity -= actual_quantity
        else:
            # 空头平仓/被减仓
            realized_pnl = (self.position.avg_price - price) * <double>actual_quantity
            self.position.quantity += actual_quantity

        # 仓位清零时重置均价
        if self.position.quantity == 0:
            self.position.avg_price = 0.0

        # 更新余额（ADL 不收手续费）
        self.balance += realized_pnl
        self.position.realized_pnl += realized_pnl

        return realized_pnl

    cpdef void reset(self):
        """重置账户到初始状态

        将余额恢复到初始值，清空持仓，重置所有统计数据。
        """
        self.balance = self.initial_balance
        self.position = Position()
        self.pending_order_id = -1
        self.maker_volume = 0
        self.total_volume = 0
        self.trade_count = 0
        self.volatility_contribution = 0.0
