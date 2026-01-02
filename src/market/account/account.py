"""账户模块

本模块定义账户(Account)类，用于记录 Agent 的交易账户状态。
"""

import time

from src.config.config import AgentConfig, AgentType
from src.core.event_engine.events import Event, EventType
from src.core.event_engine.event_bus import EventBus
from src.market.account.position import Position
from src.market.matching.trade import Trade
from src.market.orderbook.order import OrderSide


class Account:
    """账户类

    记录 Agent 的交易账户状态，包括余额、持仓、杠杆倍数和手续费率等。

    Attributes:
        agent_id: Agent ID
        agent_type: Agent 类型
        initial_balance: 初始余额
        balance: 余额
        position: 持仓对象
        leverage: 杠杆倍数
        maintenance_margin_rate: 维持保证金率
        maker_fee_rate: 挂单费率
        taker_fee_rate: 吃单费率
        pending_order_id: 当前挂单ID
    """

    def __init__(self, agent_id: int, agent_type: AgentType, config: AgentConfig) -> None:
        """创建账户

        初始化余额、持仓、杠杆、费率等账户属性。

        Args:
            agent_id: Agent ID
            agent_type: Agent 类型
            config: Agent 配置对象
        """
        self.agent_id: int = agent_id
        self.agent_type: AgentType = agent_type
        self.initial_balance: float = config.initial_balance
        self.balance: float = config.initial_balance
        self.position: Position = Position()
        self.leverage: float = config.leverage
        self.maintenance_margin_rate: float = config.maintenance_margin_rate
        self.maker_fee_rate: float = config.maker_fee_rate
        self.taker_fee_rate: float = config.taker_fee_rate
        self.pending_order_id: int | None = None

    def get_equity(self, current_price: float) -> float:
        """计算净值

        净值 = 余额 + 浮动盈亏，用于评估账户总资产状态。

        Args:
            current_price: 当前市场价格，用于计算持仓的浮动盈亏

        Returns:
            账户净值（余额 + 浮动盈亏）
        """
        return self.balance + self.position.get_unrealized_pnl(current_price)

    def get_margin_ratio(self, current_price: float) -> float:
        """计算保证金率

        保证金率 = 净值 / 持仓市值，用于评估账户强平风险。
        无持仓时返回无穷大，表示无强平风险。

        Args:
            current_price: 当前市场价格，用于计算净值和持仓市值

        Returns:
            保证金率（无持仓时为无穷大）
        """
        equity = self.get_equity(current_price)
        position_value = abs(self.position.quantity) * current_price
        if position_value == 0:
            return float("inf")
        return equity / position_value

    def check_liquidation(self, current_price: float) -> bool:
        """检查是否需要强制平仓

        当保证金率低于维持保证金率时，需要强制平仓以控制风险。

        Args:
            current_price: 当前市场价格，用于计算保证金率

        Returns:
            True 表示需要强制平仓，False 表示不需要
        """
        margin_ratio = self.get_margin_ratio(current_price)
        return margin_ratio < self.maintenance_margin_rate

    def check_elimination(self, current_price: float, threshold: float = 0.1) -> bool:
        """检查是否需要淘汰（资金不足）

        当 当前净值/初始资金 < threshold 时，返回 True 表示需要淘汰。

        Args:
            current_price: 当前市场价格，用于计算净值
            threshold: 淘汰阈值，默认 0.1（即资金剩余不足 10%）

        Returns:
            True 表示需要淘汰，False 表示不需要
        """
        equity = self.get_equity(current_price)
        if self.initial_balance <= 0:
            return True  # 防止除零
        ratio = equity / self.initial_balance
        return ratio < threshold

    def liquidate(self, current_price: float, event_bus: EventBus) -> None:
        """执行强制平仓

        发布强平事件，通知训练引擎提交市价单平仓。

        Args:
            current_price: 当前市场价格
            event_bus: 事件总线，用于发布强平事件
        """
        event_data = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "current_price": current_price,
            "position_quantity": self.position.quantity,
            "position_avg_price": self.position.avg_price,
        }
        liquidation_event = Event(EventType.LIQUIDATION, time.time(), event_data)
        event_bus.publish(liquidation_event)

    def on_trade(self, trade: Trade, is_buyer: bool) -> None:
        """处理成交回报

        根据成交记录更新持仓、扣除手续费、更新余额。

        Args:
            trade: 成交记录对象
            is_buyer: 是否为买方（True=买方，False=卖方）
        """
        # 确定成交方向和手续费
        if is_buyer:
            side = OrderSide.BUY
            fee = trade.buyer_fee
        else:
            side = OrderSide.SELL
            fee = trade.seller_fee

        # 更新持仓，获取已实现盈亏
        realized_pnl = self.position.update(side.value, trade.quantity, trade.price)

        # 将已实现盈亏加到余额，并扣除手续费
        self.balance += realized_pnl - fee
