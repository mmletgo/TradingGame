"""FastAccount 类型存根文件"""

from src.market.account.position import Position
from src.market.matching.fast_matching import FastTrade


# Agent 类型常量
RETAIL_PRO: int
MARKET_MAKER: int


class FastAccount:
    """快速账户类（Cython 加速）"""

    agent_id: int
    agent_type: int
    initial_balance: float
    balance: float
    position: Position
    leverage: float
    maintenance_margin_rate: float
    maker_fee_rate: float
    taker_fee_rate: float
    pending_order_id: int
    maker_volume: int
    total_volume: int
    trade_count: int
    volatility_contribution: float

    def __init__(
        self,
        agent_id: int,
        agent_type: int,
        initial_balance: float,
        leverage: float,
        maintenance_margin_rate: float,
        maker_fee_rate: float,
        taker_fee_rate: float,
    ) -> None: ...

    def get_equity(self, current_price: float) -> float:
        """计算净值"""
        ...

    def get_margin_ratio(self, current_price: float) -> float:
        """计算保证金率（无持仓时返回 inf）"""
        ...

    def check_liquidation(self, current_price: float) -> bool:
        """检查是否需要强制平仓"""
        ...

    def on_trade(self, trade: FastTrade, is_buyer: bool) -> None:
        """处理成交回报"""
        ...

    def on_adl_trade(self, quantity: int, price: float, is_taker: bool) -> float:
        """处理 ADL 成交，返回已实现盈亏"""
        ...

    def reset(self) -> None:
        """重置账户到初始状态"""
        ...
