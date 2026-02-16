"""ADL (Auto-Deleveraging) 自动减仓管理器"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from src.bio.agents.base import Agent
    from src.market.noise_trader.noise_trader import NoiseTrader


@dataclass
class ADLCandidate:
    """ADL 候选者信息"""

    participant: Union["Agent", "NoiseTrader"]  # 支持 Agent 或噪声交易者
    position_qty: int  # 持仓数量（正=多头，负=空头）
    pnl_percent: float  # 盈亏百分比
    effective_leverage: float  # 有效杠杆
    adl_score: float  # ADL 排名分数（越高越优先）

    @property
    def agent(self) -> "Agent":
        """兼容旧代码（仅当 participant 是 Agent 时可用）"""
        from src.bio.agents.base import Agent

        if not isinstance(self.participant, Agent):
            raise TypeError("This candidate is a NoiseTrader, not an Agent")
        return self.participant

    @property
    def is_noise_trader(self) -> bool:
        """检查候选者是否为噪声交易者"""
        from src.market.noise_trader.noise_trader import NoiseTrader

        return isinstance(self.participant, NoiseTrader)


class ADLManager:
    """ADL 自动减仓管理器

    当强平订单无法完全成交时，触发 ADL 机制。
    按照 ADL 排名（盈利百分比 * 有效杠杆）选择对手方强制减仓。
    """

    def __init__(self) -> None:
        from src.core.log_engine.logger import get_logger

        self.logger = get_logger("adl")

    def get_adl_price(
        self,
        current_price: float,
    ) -> float:
        """获取 ADL 成交价格

        ADL 机制简化：直接使用当前市场价格成交。

        强平 ≠ 破产：被强平时 Agent 可能还有正的净值（只是保证金率过低），
        因此不应该用"破产价格"来计算 ADL 成交价。

        使用当前市场价格的好处：
        - 简单公平：双方都以市场价成交
        - 避免异常：不会因为穿仓导致价格异常
        - 符合直觉：流动性不足时强制以当前价成交

        Args:
            current_price: 当前市场价格

        Returns:
            ADL 成交价格（即当前市场价格）
        """
        return current_price

    def calculate_adl_score(
        self,
        agent: "Agent",
        current_price: float,
    ) -> ADLCandidate | None:
        """计算单个 Agent 的 ADL 排名分数

        排名公式：
        - 如果 PnL% > 0：排名 = PnL% * 有效杠杆
        - 如果 PnL% <= 0：排名 = PnL% / 有效杠杆

        其中：
        - PnL% = 浮动盈亏 / |开仓成本|
        - 有效杠杆 = |持仓市值| / 净值

        Args:
            agent: Agent 对象
            current_price: 当前市场价格

        Returns:
            ADLCandidate 对象，无持仓时返回 None
        """
        position = agent.account.position
        quantity = position.quantity

        # 无持仓，不参与 ADL
        if quantity == 0:
            return None

        avg_price = position.avg_price
        abs_quantity = abs(quantity)

        # 计算开仓成本
        entry_cost = abs_quantity * avg_price
        if entry_cost <= 0:
            return None

        # 计算浮动盈亏
        unrealized_pnl = position.get_unrealized_pnl(current_price)

        # 计算盈亏百分比
        pnl_percent = unrealized_pnl / entry_cost

        # 计算净值
        equity = agent.account.get_equity(current_price)
        if equity <= 0:
            # 净值为非正数，有效杠杆设为极大值
            effective_leverage = float("inf")
        else:
            # 有效杠杆 = |持仓市值| / 净值
            position_value = abs_quantity * current_price
            effective_leverage = position_value / equity

        # 计算 ADL 排名分数
        if pnl_percent > 0:
            # 盈利者：分数 = 盈利百分比 * 有效杠杆
            adl_score = pnl_percent * effective_leverage
        else:
            # 亏损者：分数 = 盈利百分比 / 有效杠杆
            # 有效杠杆越高，分数越低（越不优先被选中）
            if effective_leverage == float("inf"):
                adl_score = float("-inf")
            elif effective_leverage > 0:
                adl_score = pnl_percent / effective_leverage
            else:
                adl_score = float("-inf")

        return ADLCandidate(
            participant=agent,
            position_qty=quantity,
            pnl_percent=pnl_percent,
            effective_leverage=effective_leverage,
            adl_score=adl_score,
        )
